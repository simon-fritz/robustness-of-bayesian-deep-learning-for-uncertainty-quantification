"""Generic training loop.

The trainer is method-agnostic: it produces a MAP-trained network. Method-
specific posterior fitting (Laplace, ensemble member init, etc.) happens
afterwards via ``BayesianMethod.fit``.
"""

from __future__ import annotations

import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Standard supervised trainer."""

    def __init__(self, cfg) -> None:
        # TODO: implement — store optimizer/scheduler config, device, logging.
        self.cfg = cfg

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> nn.Module:
        """Train ``model`` to MAP convergence and return it."""
        # TODO: implement
        raise NotImplementedError
