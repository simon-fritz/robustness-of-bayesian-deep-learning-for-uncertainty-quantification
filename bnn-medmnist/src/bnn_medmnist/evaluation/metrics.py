"""Classification metrics: accuracy, NLL, Brier, ECE."""

from __future__ import annotations

import torch


def accuracy(probs: torch.Tensor, targets: torch.Tensor) -> float:
    # TODO: implement
    raise NotImplementedError


def nll(probs: torch.Tensor, targets: torch.Tensor) -> float:
    # TODO: implement
    raise NotImplementedError


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    # TODO: implement
    raise NotImplementedError


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    # TODO: implement
    raise NotImplementedError
