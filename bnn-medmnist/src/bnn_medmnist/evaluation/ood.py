"""OOD detection scoring.

Given an uncertainty score (max-softmax, predictive entropy, mutual
information, ...) and labels (0 = ID, 1 = OOD), report AUROC, AUPR, and
FPR@95%TPR.
"""

from __future__ import annotations

import torch


def auroc(scores: torch.Tensor, ood_labels: torch.Tensor) -> float:
    # TODO: implement
    raise NotImplementedError


def aupr(scores: torch.Tensor, ood_labels: torch.Tensor) -> float:
    # TODO: implement
    raise NotImplementedError


def fpr_at_tpr(scores: torch.Tensor, ood_labels: torch.Tensor, tpr: float = 0.95) -> float:
    # TODO: implement
    raise NotImplementedError
