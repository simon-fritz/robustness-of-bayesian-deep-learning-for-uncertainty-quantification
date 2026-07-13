"""Last-layer Laplace approximation via ``laplace-torch``.

Two phases:
    1. MAP training — delegated to the deterministic ``Trainer``.
    2. Laplace fit on the final classifier layer (``fc``); optional marginal-
       likelihood optimization of the prior precision.

At predict time draws ``n_samples`` MC samples from the Gaussian posterior
and returns post-softmax probabilities, both averaged and per-sample.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from laplace import Laplace
from torch.utils.data import DataLoader

from ..training.trainer import Trainer
from .base import BayesianMethod


class LastLayerLaplace(BayesianMethod):
    """MAP training + last-layer Laplace posterior."""

    #: Default MC predictive type. ``"nn"`` samples network weights from the
    #: posterior; fine for the last layer whose posterior is well-constrained.
    #: Subclasses over wide/prior-dominated posteriors (e.g. first-layer) should
    #: override to ``"glm"`` — see :class:`FirstLayerLaplace`.
    _default_pred_type: str = "nn"

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
        super().__init__(bayesian_layers=["fc"], n_samples=n_predictive)
        # Predictive type: "glm" (linearized, function-space) or "nn" (weight
        # sampling). Config may override; else the class default applies.
        self.pred_type = str(
            getattr(laplace_cfg, "pred_type", None) or self._default_pred_type
        )
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

        # Last-layer Laplace targets the final classifier by the name ``fc``.
        # This works for both SmallCNN and PretrainedResNet18 because both expose
        # their classifier head as a top-level ``fc`` (torchvision ResNets use
        # that exact name). Assert it loudly so a future architecture without an
        # ``fc`` Linear fails here rather than deep inside laplace-torch.
        fc = getattr(self.model, "fc", None)
        if not isinstance(fc, nn.Linear):
            raise AttributeError(
                f"last_layer_laplace requires the model to expose an nn.Linear "
                f"attribute named 'fc' (got {type(fc).__name__}). "
                f"Models: SmallCNN, PretrainedResNet18 both satisfy this."
            )

        # Phase 2: Laplace approximation.
        cfg = self.laplace_cfg
        subset = str(getattr(cfg, "subset_of_weights", "last_layer"))
        hess = str(getattr(cfg, "hessian_structure", "full"))
        prior_method = getattr(cfg, "prior_precision_method", "marglik")

        print(f"fitting Laplace (subset={subset}, hessian={hess})...", flush=True)
        self.la = Laplace(
            self.model,
            likelihood="classification",
            subset_of_weights=subset,
            hessian_structure=hess,
        )
        self.la.fit(train_loader)

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
            raise RuntimeError("LastLayerLaplace.fit must be called before predict.")
        n = int(n_samples) if n_samples is not None else self.n_samples
        x = x.to(self.device)
        # detach: the GLM predictive builds an autograd graph (enable_grad); not
        # detaching leaks GPU memory across eval batches on large OOD sets.
        return self.la.predictive_samples(
            x, pred_type=self.pred_type, n_samples=n
        ).detach()

    @torch.no_grad()
    def glm_logit_distribution(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytical Gaussian over the logits via ``pred_type="glm"``.

        Returns ``(logit_mean, logit_var)`` of shape ``(B, C)`` each — the mean
        and per-class (diagonal) variance of the Gaussian posterior over the
        logits induced by the Gaussian posterior over the last-layer weights.
        Sampling-free; this is the natural "variance of a Gaussian" uncertainty.
        """
        if self.la is None:
            raise RuntimeError("LastLayerLaplace.fit must be called before predict.")
        x = x.to(self.device)
        f_mu, f_var = self.la._glm_predictive_distribution(x, diagonal_output=True)
        # detach: _glm_predictive_distribution runs under enable_grad, so its
        # outputs carry the autograd graph; keeping it leaks GPU memory across
        # eval batches (OOM on large OOD sets).
        return f_mu.detach(), f_var.detach()

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
        """Summary statistics of the posterior covariance for the last layer.

        Returns mean/max of diagonal(Σ) and Frobenius norm of Σ — metrics that
        should shrink monotonically as training size grows (posterior collapse
        signal). Saved to ``sigma_summary.json`` in the run directory by
        ``scripts/train.py``.
        """
        if self.la is None:
            raise RuntimeError("fit must be called before sigma_summary.")
        cov = self.la.posterior_covariance
        if not torch.is_tensor(cov):
            cov = torch.as_tensor(cov)
        cov = cov.detach().cpu()
        diag = torch.diagonal(cov)
        result: dict = {
            "mean_sigma": float(diag.mean()),
            "max_sigma": float(diag.max()),
            "sigma_norm": float(torch.norm(cov, p="fro")),
        }
        if train_size is not None:
            result["train_size"] = int(train_size)
        return result

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        return self.predictive_samples(x, n_samples).mean(dim=0)
