"""Last-layer Laplace approximation.

Trains a deterministic network and fits a Gaussian posterior over the final
classifier-layer weights via the Laplace approximation (``laplace-torch``).
Predictive distribution at test time is the MC integral over samples drawn
from this Gaussian.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BayesianMethod


class LastLayerLaplace(BayesianMethod):
    """Last-layer Laplace approximation via ``laplace-torch``."""

    def __init__(
        self,
        n_samples: int = 100,
        hessian_structure: str = "kron",
        prior_precision: float = 1.0,
        optimize_prior_precision: str | None = "marglik",
    ) -> None:
        super().__init__(bayesian_layers=["fc"], n_samples=n_samples)
        self.hessian_structure = hessian_structure
        self.prior_precision = prior_precision
        self.optimize_prior_precision = optimize_prior_precision
        # Populated by fit():
        self.la = None  # laplace-torch Laplace instance

    def fit(self, model: nn.Module, train_loader: DataLoader) -> None:
        # TODO: implement
        # 1. Assume `model` is already MAP-trained (or train it here).
        # 2. Wrap with laplace-torch's Laplace(model, "classification",
        #    subset_of_weights="last_layer", hessian_structure=...).
        # 3. la.fit(train_loader); optionally la.optimize_prior_precision(...).
        raise NotImplementedError

    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        # TODO: implement — la(x, pred_type="nn", n_samples=n_samples or self.n_samples)
        raise NotImplementedError
