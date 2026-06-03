"""Deep-ensemble baseline."""



from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..training.trainer import Trainer
from .base import BayesianMethod
from bnn_medmnist.utils.seeding import set_seed


class DeepEnsemble(BayesianMethod):

    def __init__(
        self,
        train_cfg=None,
        ckpt_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        class_weights: torch.Tensor | None = None,
        device: str | None = None,
        n_members = 5
    ) -> None:
        super().__init__(bayesian_layers=[], n_samples=1)
        self.model: nn.Module | None = None
        self.train_cfg = train_cfg
        self.ckpt_path = ckpt_path
        self.log_dir = log_dir
        self.class_weights = class_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_members = n_members
        self.members = []


    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> list[nn.Module]:
        for i in range(self.n_members):
            # 1. Unique seed per member
            seed = getattr(self, 'base_seed', 42) + i
            set_seed(seed)
            
            # 2. Fresh model instance (clone the architecture)
            member_model = copy.deepcopy(model)
            
            # 3. Use existing Trainer
            # Convert ckpt_path to Path if needed and handle None
            ckpt_dir = Path(self.ckpt_path) if self.ckpt_path else Path(".")
            member_ckpt = ckpt_dir.parent / f"member_{i}.pt" if self.ckpt_path else None
            
            trainer = Trainer(
                self.train_cfg,
                device=self.device,
                ckpt_path=member_ckpt,
                log_dir=self.log_dir,
                class_weights=self.class_weights,
            )
            
            # 4. Train using the standard pipeline
            trained = trainer.fit(member_model, train_loader, val_loader)
            
            # 5. Save to list
            self.members.append(trained)
        return self.members


    @torch.no_grad()
    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        all_probs = []
        for member in self.members:
            member.eval()
            probs = torch.softmax(member(x.to(self.device)), dim=-1)
            all_probs.append(probs)
        # Stack as (S, B, C) to match Laplace output
        return torch.stack(all_probs, dim=0)
