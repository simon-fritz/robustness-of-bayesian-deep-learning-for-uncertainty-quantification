"""First-layer Laplace approximation via ``laplace-torch``.

Two phases:
    1. MAP training — delegated to the deterministic ``Trainer``.
    2. Laplace fit on the first convolutional layer (``conv1``) via
       the :func:`laplace.Laplace` factory with ``subset_of_weights="subnetwork"``
       (dispatches to :class:`laplace.SubnetLaplace` internally); optional
       marginal-likelihood optimization of the prior precision.

At predict time draws ``n_samples`` MC samples from the Gaussian posterior over
the ``conv1`` weights and returns post-softmax probabilities, both averaged and
per-sample.

Model requirement: the network must expose the first convolution as a top-level
attribute named ``conv1`` of type :class:`torch.nn.Conv2d`. Torchvision ResNets
(``PretrainedResNet18``) satisfy this. ``SmallCNN`` wraps its first conv in a
``Sequential`` (``layer1``) and would need a different subnetwork-mask setup.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from laplace import Laplace
from laplace.utils import ModuleNameSubnetMask
from torch.utils.data import DataLoader

from ..training.trainer import Trainer
from .base import BayesianMethod


class FirstLayerLaplace(BayesianMethod):
    """MAP training + first-layer Laplace posterior."""

    def __init__(
        self,
        train_cfg=None,
        laplace_cfg=None,
        ckpt_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
        device: str | None = None,
    ) -> None:
        n_predictive = int(getattr(laplace_cfg, "n_predictive_samples", 100))
        super().__init__(bayesian_layers=["conv1"], n_samples=n_predictive)
        self.train_cfg = train_cfg
        self.laplace_cfg = laplace_cfg
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self.log_dir = log_dir
        self.class_weights = class_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.la = None

    @property
    def laplace_path(self) -> Path | None:
        if self.ckpt_path is None:
            return None
        return self.ckpt_path.with_suffix(".laplace.pt")

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> nn.Module:
        # Phase 1: MAP training (re-use deterministic trainer).
        trainer = Trainer(
            self.train_cfg,
            device=self.device,
            log_dir=self.log_dir,
            ckpt_path=self.ckpt_path,
            class_weights=self.class_weights,
        )
        self.model = trainer.fit(model, train_loader, val_loader)
        self.model.eval()

        # First-layer Laplace targets the first convolution by the name ``conv1``.
        conv1 = getattr(self.model, "conv1", None)
        if not isinstance(conv1, nn.Conv2d):
            raise AttributeError(
                f"first_layer_laplace requires the model to expose an nn.Conv2d "
                f"attribute named 'conv1' (got {type(conv1).__name__}). "
                f"PretrainedResNet18 satisfies this; SmallCNN wraps its first "
                f"conv in a Sequential and would need a different subnet mask."
            )

        # Phase 2: Laplace approximation.
        cfg = self.laplace_cfg
        subset = str(getattr(cfg, "subset_of_weights", "subnetwork"))
        hess = str(getattr(cfg, "hessian_structure", "diag"))
        prior_method = getattr(cfg, "prior_precision_method", "marglik")
        module_name = str(getattr(cfg, "first_layer_module", "conv1"))

        print(f"fitting Laplace (subset={subset}, hessian={hess}, module={module_name})...", flush=True)

        # Build the subnetwork mask: mark all parameters of the named module
        # (conv1) as the Bayesian subnetwork. ``select()`` walks the model and
        # produces flat parameter indices consumed by SubnetLaplace.
        subnet_mask = ModuleNameSubnetMask(self.model, module_names=[module_name])
        subnet_mask.select(train_loader)
        raw_indices = subnet_mask.indices
        # SubnetLaplace strictly requires a non-empty 1-D torch.LongTensor.
        # ModuleNameSubnetMask can return int32 / non-flat tensors — cast explicitly.
        indices = torch.as_tensor(raw_indices, dtype=torch.long).flatten().cpu()
        print(
            f"subnetwork_indices: {int(indices.numel())} parameters selected from "
            f"'{module_name}' (dtype={indices.dtype}, shape={tuple(indices.shape)})",
            flush=True,
        )

        # Use the Laplace() factory: with subset_of_weights="subnetwork" it
        # dispatches to SubnetLaplace under the hood. Same pattern as LLL.
        self.la = Laplace(
            self.model,
            likelihood="classification",
            subset_of_weights=subset,
            hessian_structure=hess,
            subnetwork_indices=indices,
        )

        # torch.func.jacrev in laplace-torch computes the FULL-network Jacobian
        # even for subnetwork Laplace (only the sub-slice is kept afterwards).
        # For a Conv2d subnetwork on ResNet-18 at 224x224 that is both memory-
        # heavy (bs=128 → OOM) and slow (~1h on the full training set).
        #
        # Two mitigations, both configurable:
        #   * fit_batch_size:   reduce batch size for the fit only (default 16).
        #   * fit_max_samples:  subsample the training data for the fit
        #                       (default 512). Laplace posteriors from ~500
        #                       samples are effectively identical to the full
        #                       fit for our purposes and take minutes not hours.
        from torch.utils.data import Subset
        max_fit = int(getattr(cfg, "fit_max_samples", 512))
        full_ds = train_loader.dataset
        if len(full_ds) > max_fit:
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(len(full_ds), generator=g)[:max_fit].tolist()
            fit_dataset = Subset(full_ds, perm)
        else:
            fit_dataset = full_ds

        fit_batch = int(getattr(cfg, "fit_batch_size", 16))
        laplace_loader = DataLoader(
            fit_dataset,
            batch_size=fit_batch,
            shuffle=False,
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
        )
        print(
            f"Laplace fit: {len(fit_dataset)}/{len(full_ds)} samples "
            f"@ batch_size={fit_batch} (MAP was {train_loader.batch_size})",
            flush=True,
        )
        self.la.fit(laplace_loader)
        print("Laplace fit: done", flush=True)

        if isinstance(prior_method, str):
            try:
                self.la.optimize_prior_precision(method=prior_method)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: optimize_prior_precision({prior_method}) failed "
                      f"({exc!r}); falling back to prior_precision=1.0", flush=True)
                self.la.prior_precision = torch.tensor(1.0)
        else:
            self.la.prior_precision = torch.tensor(float(prior_method))
        print(f"Laplace prior_precision = {float(self.la.prior_precision.flatten()[0]):.4f}", flush=True)

        if self.laplace_path is not None:
            payload = {
                "state_dict": self.la.state_dict(),
                "subset_of_weights": subset,
                "hessian_structure": hess,
                "subnetwork_indices": indices.detach().cpu(),
                "first_layer_module": module_name,
            }
            torch.save(payload, self.laplace_path)
            print(f"laplace saved to {self.laplace_path}", flush=True)
        return self.model

    @torch.no_grad()
    def predictive_samples(
        self, x: torch.Tensor, n_samples: int | None = None
    ) -> torch.Tensor:
        """Return MC samples of softmax probabilities, shape ``(S, B, C)``."""
        if self.la is None:
            raise RuntimeError("FirstLayerLaplace.fit must be called before predict.")
        n = int(n_samples) if n_samples is not None else self.n_samples
        x = x.to(self.device)
        return self.la.predictive_samples(x, pred_type="nn", n_samples=n)

    @torch.no_grad()
    def glm_logit_distribution(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytical Gaussian over the logits via ``pred_type="glm"``.

        Returns ``(logit_mean, logit_var)`` of shape ``(B, C)`` each — the mean
        and per-class (diagonal) variance of the Gaussian posterior over the
        logits induced by the Gaussian posterior over the first-layer conv
        weights. Sampling-free; this is the natural "variance of a Gaussian"
        uncertainty.

        Note: GLM linearizes the network output around the MAP estimate w.r.t.
        the subnetwork weights. For a Conv2d subnet the Jacobian is far larger
        than for a linear last-layer subnet — expect slower predictions than
        the last-layer variant.
        """
        if self.la is None:
            raise RuntimeError("FirstLayerLaplace.fit must be called before predict.")
        x = x.to(self.device)
        f_mu, f_var = self.la._glm_predictive_distribution(x, diagonal_output=True)
        return f_mu, f_var

    @torch.no_grad()
    def glm_logit_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Standard deviation (sigma) of the analytical Gaussian logit posterior.

        ``sigma = sqrt(logit_var)`` — the per-class standard deviation of the
        same Gaussian returned by :meth:`glm_logit_distribution`, on the
        natural ("how many logits wide is one standard deviation") scale.
        Sampling-free, like the variance it derives from. Shape ``(B, C)``.
        """
        _, f_var = self.glm_logit_distribution(x)
        return f_var.clamp_min(0).sqrt()

    @torch.no_grad()
    def predict_modes(
        self,
        x: torch.Tensor,
        n_samples: int | None = None,
        modes: tuple[str, ...] = ("mc", "glm"),
    ) -> dict[str, torch.Tensor | None]:
        """Run one or both prediction modes and return a structured result.

        * ``"mc"``  → MC samples of softmax probabilities (``softmax_samples``).
        * ``"glm"`` → analytical Gaussian over logits (``logit_mean``/
          ``logit_var``/``logit_sigma``).

        Returns a dict with keys ``softmax_samples`` ``(S, B, C)``,
        ``logit_mean`` ``(B, C)``, ``logit_var`` ``(B, C)``, ``logit_sigma``
        ``(B, C)`` (= ``sqrt(logit_var)``) — each ``None`` if the corresponding
        mode was not requested.
        """
        out: dict[str, torch.Tensor | None] = {
            "softmax_samples": None, "logit_mean": None, "logit_var": None,
            "logit_sigma": None,
        }
        if "mc" in modes:
            out["softmax_samples"] = self.predictive_samples(x, n_samples)
        if "glm" in modes:
            out["logit_mean"], out["logit_var"] = self.glm_logit_distribution(x)
            out["logit_sigma"] = out["logit_var"].clamp_min(0).sqrt()
        return out

    def sigma_summary(self, train_size: int | None = None) -> dict:
        """Summary statistics of the posterior covariance for the first layer.

        Returns mean/max of diagonal(Σ) and Frobenius norm of Σ — metrics that
        should shrink monotonically as training size grows (posterior collapse
        signal). Saved to ``sigma_summary.json`` in the run directory by
        ``scripts/train.py``.

        Handles both matrix (full Hessian) and vector (diag Hessian) posterior
        covariance representations returned by SubnetLaplace.
        """
        if self.la is None:
            raise RuntimeError("fit must be called before sigma_summary.")
        cov = self.la.posterior_covariance
        if not torch.is_tensor(cov):
            cov = torch.as_tensor(cov)
        cov = cov.detach().cpu()

        if cov.dim() == 1:
            # Diagonal-only representation (hessian_structure="diag").
            diag = cov
            sigma_norm = float(torch.norm(diag, p=2))
        elif cov.dim() == 2:
            diag = torch.diagonal(cov)
            sigma_norm = float(torch.norm(cov, p="fro"))
        else:
            raise RuntimeError(
                f"unexpected posterior_covariance shape {tuple(cov.shape)}"
            )

        result: dict = {
            "mean_sigma": float(diag.mean()),
            "max_sigma": float(diag.max()),
            "sigma_norm": sigma_norm,
        }
        if train_size is not None:
            result["train_size"] = int(train_size)
        return result

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        return self.predictive_samples(x, n_samples).mean(dim=0)
