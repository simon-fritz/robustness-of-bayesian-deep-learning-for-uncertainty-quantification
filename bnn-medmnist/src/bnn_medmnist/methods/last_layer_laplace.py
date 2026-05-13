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
        return self.la.predictive_samples(x, pred_type="nn", n_samples=n)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        return self.predictive_samples(x, n_samples).mean(dim=0)
