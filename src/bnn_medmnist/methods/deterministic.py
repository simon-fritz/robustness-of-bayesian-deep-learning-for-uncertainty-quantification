"""Deterministic (non-Bayesian) baseline."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..training.trainer import Trainer
from .base import BayesianMethod


class Deterministic(BayesianMethod):
    """MAP point estimate (no posterior, no MC sampling)."""

    def __init__(
        self,
        train_cfg=None,
        ckpt_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(bayesian_layers=[], n_samples=1)
        self.model: nn.Module | None = None
        self.train_cfg = train_cfg
        self.ckpt_path = ckpt_path
        self.log_dir = log_dir
        self.class_weights = class_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> nn.Module:
        trainer = Trainer(
            self.train_cfg,
            device=self.device,
            log_dir=self.log_dir,
            ckpt_path=self.ckpt_path,
            class_weights=self.class_weights,
        )
        self.model = trainer.fit(model, train_loader, val_loader)
        return self.model

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Deterministic.fit must be called before predict.")
        self.model.eval()
        device = next(self.model.parameters()).device
        return torch.softmax(self.model(x.to(device)), dim=-1)
