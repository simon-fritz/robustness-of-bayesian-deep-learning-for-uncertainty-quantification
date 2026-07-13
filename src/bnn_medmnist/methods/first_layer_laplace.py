"""First-layer Laplace approximation via ``laplace-torch`` subnetwork Laplace.

Counterpart to :class:`LastLayerLaplace` at the other end of the network: the
Gaussian posterior sits over the *first* conv layer's weights (ResNet:
``conv1``, 9,408 params) while everything else stays at its MAP value.

Two phases:
    1. MAP training — byte-identical to last-layer Laplace (delegated to the
       deterministic ``Trainer``). Because the Laplace fit is purely post-hoc,
       ``map_checkpoint`` can point at an existing MAP checkpoint (e.g. from a
       finished LLL run with the same config/seed) to skip training entirely.
    2. Subnetwork Laplace fit over the modules named in ``bayesian_modules``.

Implementation note — why ``requires_grad`` is flipped before the fit:
laplace-torch's subnetwork path differentiates w.r.t. *all* trainable
parameters and only then column-selects the subnetwork Jacobian
(``Js[:, :, subnetwork_indices]``). For a ResNet-18 that would materialize
``(batch, classes, 11.7M)`` Jacobians per batch. Disabling gradients for
everything except the target modules (explicitly supported by laplace-torch,
see ``BaseLaplace.is_subset_params``) shrinks that to ``(batch, classes,
n_subnet)``; the ``subnetwork_indices`` then simply cover the whole remaining
parameter vector. The same pattern must be re-applied before
``load_state_dict`` at eval time — :meth:`FirstLayerLaplace.load_laplace`
handles that.

The predictive API (``predictive_samples`` / ``glm_logit_distribution`` /
``predict_modes`` / ``sigma_summary``) is inherited unchanged from
:class:`LastLayerLaplace` — everything operates on ``self.la``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from laplace import Laplace
from torch.utils.data import DataLoader

from ..training.trainer import Trainer
from .last_layer_laplace import LastLayerLaplace


def mark_bayesian_submodules(model: nn.Module, module_names: Sequence[str]) -> int:
    """Enable gradients only for the named submodules; return their param count.

    ``module_names`` are resolved via ``model.get_submodule`` and therefore
    support dotted paths (e.g. ``"layer1.0"`` for the first conv inside
    SmallCNN's Sequential block, or plain ``"conv1"`` on the ResNets).
    """
    model.requires_grad_(False)
    n_subnet = 0
    for name in module_names:
        module = model.get_submodule(name)  # raises AttributeError if missing
        params = list(module.parameters())
        if not params:
            raise ValueError(f"module '{name}' has no parameters.")
        for p in params:
            p.requires_grad_(True)
            n_subnet += p.numel()
    return n_subnet


def _rebatch(loader: DataLoader, batch_size: int) -> DataLoader:
    """Same dataset, different batch size (no-op for ``batch_size <= 0``)."""
    if batch_size <= 0 or batch_size == loader.batch_size:
        return loader
    return DataLoader(
        loader.dataset, batch_size=batch_size, shuffle=False,
        num_workers=loader.num_workers,
    )


class FirstLayerLaplace(LastLayerLaplace):
    """MAP training (or checkpoint reuse) + first-layer subnetwork Laplace.

    Predictions default to the **GLM** (linearized, function-space) predictive
    rather than ``"nn"`` weight sampling. conv1's Laplace posterior is wide
    (its GGN curvature is tiny, so the posterior is essentially the prior, with
    per-weight std >> the weight magnitude); sampling those weights and running
    them forward scrambles the stem conv and collapses predictions to the
    majority class (observed: MC acc 0.62 vs GLM acc 0.92 on PneumoniaMNIST).
    The GLM predictive linearizes around the MAP, preserving accuracy while
    still propagating the posterior logit variance for uncertainty.
    """

    _default_pred_type: str = "glm"

    def __init__(
        self,
        train_cfg=None,
        laplace_cfg=None,
        ckpt_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
        device: str | None = None,
        bayesian_modules: Sequence[str] = ("conv1",),
        map_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__(
            train_cfg=train_cfg, laplace_cfg=laplace_cfg, ckpt_path=ckpt_path,
            log_dir=log_dir, class_weights=class_weights, device=device,
        )
        self.bayesian_layers = list(bayesian_modules)
        self.map_checkpoint = (
            Path(map_checkpoint) if map_checkpoint is not None else None
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> nn.Module:
        # Phase 1: MAP — either reuse an existing checkpoint (the fit is
        # post-hoc, so any MAP model trained under the same config/seed works)
        # or train exactly like the deterministic / last-layer-Laplace runs.
        if self.map_checkpoint is not None:
            state = torch.load(self.map_checkpoint, map_location="cpu")
            model.load_state_dict(state)
            self.model = model.to(self.device)
            print(f"reusing MAP checkpoint {self.map_checkpoint} "
                  f"(skipping MAP training)", flush=True)
            if self.ckpt_path is not None:
                # Persist a copy so the run directory stays self-contained.
                self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state, self.ckpt_path)
        else:
            trainer = Trainer(
                self.train_cfg,
                device=self.device,
                log_dir=self.log_dir,
                ckpt_path=self.ckpt_path,
                class_weights=self.class_weights,
            )
            self.model = trainer.fit(model, train_loader, val_loader)
        self.model.eval()

        # Phase 2: subnetwork Laplace over the first layer.
        cfg = self.laplace_cfg
        hess = str(getattr(cfg, "hessian_structure", "full"))
        prior_method = getattr(cfg, "prior_precision_method", "marglik")

        n_subnet = mark_bayesian_submodules(self.model, self.bayesian_layers)
        # The full-GGN fit builds a (batch, classes, p, p)-ish einsum
        # intermediate, so peak memory scales ~ batch * n_subnet^2. For conv1
        # (p=9408) a large batch OOMs; a dedicated small-batch loader keeps the
        # peak bounded (total fit work is ~independent of batch size).
        fit_loader = _rebatch(
            train_loader, int(getattr(cfg, "fit_batch_size", 0) or 0)
        )

        print(f"fitting subnetwork Laplace over {self.bayesian_layers} "
              f"({n_subnet} params, hessian={hess}, "
              f"fit_batch_size={fit_loader.batch_size})...", flush=True)
        self.la = Laplace(
            self.model,
            likelihood="classification",
            subset_of_weights="subnetwork",
            hessian_structure=hess,
            # requires_grad already restricts the parameter vector to the
            # target modules; the indices just cover all of it. Must stay a
            # CPU LongTensor — SubnetLaplace type-checks torch.LongTensor,
            # which a CUDA tensor fails.
            subnetwork_indices=torch.arange(n_subnet, dtype=torch.long),
        )
        self.la.fit(fit_loader)

        if isinstance(prior_method, str):
            try:
                self.la.optimize_prior_precision(method=prior_method)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: optimize_prior_precision({prior_method}) failed "
                      f"({exc!r}); falling back to prior_precision=1.0", flush=True)
                self.la.prior_precision = torch.tensor(1.0)
        else:
            self.la.prior_precision = torch.tensor(float(prior_method))
        print(f"Laplace prior_precision = "
              f"{float(self.la.prior_precision.flatten()[0]):.4f}", flush=True)

        if self.laplace_path is not None:
            payload = {
                "state_dict": self.la.state_dict(),
                "subset_of_weights": "subnetwork",
                "hessian_structure": hess,
                # Needed at load time to re-derive the requires_grad pattern
                # and subnetwork indices (laplace-torch requires both to match
                # the fit exactly).
                "bayesian_modules": list(self.bayesian_layers),
            }
            torch.save(payload, self.laplace_path)
            print(f"laplace saved to {self.laplace_path}", flush=True)
        return self.model

    @classmethod
    def load_laplace(cls, model: nn.Module, la_path: str | Path, device: str):
        """Rebuild the fitted subnetwork Laplace for evaluation.

        Re-applies the exact ``requires_grad`` pattern and subnetwork indices
        used at fit time (both derived from the payload's
        ``bayesian_modules``) before ``load_state_dict`` — laplace-torch
        refuses to load otherwise.
        """
        model.eval()  # BN must use running stats — sampling/GLM assume it
        payload = torch.load(la_path, map_location=device, weights_only=False)
        n_subnet = mark_bayesian_submodules(
            model, list(payload["bayesian_modules"])
        )
        la = Laplace(
            model,
            likelihood="classification",
            subset_of_weights="subnetwork",
            hessian_structure=payload["hessian_structure"],
            subnetwork_indices=torch.arange(n_subnet, dtype=torch.long),
        )
        la.load_state_dict(payload["state_dict"])
        return la
