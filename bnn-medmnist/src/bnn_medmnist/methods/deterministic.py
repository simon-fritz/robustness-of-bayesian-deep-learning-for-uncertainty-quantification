"""Deterministic (non-Bayesian) baseline.

A point-estimate softmax classifier — no posterior. Included so that all
methods share the same ``fit`` / ``predict`` interface and so that downstream
evaluation code is uniform.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BayesianMethod


class Deterministic(BayesianMethod):
    """MAP point estimate (no posterior, no MC sampling)."""

    def __init__(self) -> None:
        super().__init__(bayesian_layers=[], n_samples=1)
        self.model: nn.Module | None = None

    def fit(self, model: nn.Module, train_loader: DataLoader) -> None:
        # TODO: implement — standard supervised training loop.
        raise NotImplementedError

    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        # TODO: implement — single forward pass, softmax.
        raise NotImplementedError
