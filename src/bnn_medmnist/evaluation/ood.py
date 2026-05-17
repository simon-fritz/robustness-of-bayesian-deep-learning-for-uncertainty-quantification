"""OOD detection scoring.

Given a 1-D uncertainty score per sample for ID and OOD inputs, report AUROC,
AUPR, and FPR@95%TPR with OOD as the positive class (higher score = more
OOD-like).

Also provides :func:`evaluate_ood`, which takes raw predictive samples and
computes every uncertainty score defined in :mod:`evaluation.uncertainty`,
plus ``1 - max softmax probability``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .uncertainty import expected_entropy, mutual_information, predictive_entropy


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def _labels_and_scores(id_scores, ood_scores) -> tuple[np.ndarray, np.ndarray]:
    id_s = _to_np(id_scores)
    ood_s = _to_np(ood_scores)
    y = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])
    s = np.concatenate([id_s, ood_s])
    return y, s


def ood_auroc(id_scores, ood_scores) -> float:
    y, s = _labels_and_scores(id_scores, ood_scores)
    return float(roc_auc_score(y, s))


def ood_auprc(id_scores, ood_scores) -> float:
    y, s = _labels_and_scores(id_scores, ood_scores)
    return float(average_precision_score(y, s))


def fpr_at_95_tpr(id_scores, ood_scores) -> float:
    """False-positive rate (ID mistaken for OOD) at the threshold yielding TPR>=0.95."""
    y, s = _labels_and_scores(id_scores, ood_scores)
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, 0.95, side="left")
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def one_minus_max_softmax(probs_samples: torch.Tensor) -> torch.Tensor:
    """1 - max softmax probability of the mean predictive. Shape ``(B,)``."""
    mean = probs_samples.mean(dim=0)
    return 1.0 - mean.max(dim=-1).values


SCORE_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "predictive_entropy": predictive_entropy,
    "mutual_information": mutual_information,
    "expected_entropy": expected_entropy,
    "one_minus_max_softmax": one_minus_max_softmax,
}


def evaluate_ood(
    predictions_id: torch.Tensor,
    predictions_ood: torch.Tensor,
    score_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute AUROC/AUPRC/FPR@95 for every uncertainty score.

    ``predictions_*`` are predictive-sample tensors of shape ``(S, N, C)`` —
    ``S=1`` for deterministic models, ``S>1`` for Bayesian methods.

    Returns ``{score_name: {auroc, auprc, fpr_at_95_tpr}}``.
    """
    fns = score_fns if score_fns is not None else SCORE_FNS
    out: dict[str, dict[str, float]] = {}
    for name, fn in fns.items():
        id_s = fn(predictions_id)
        ood_s = fn(predictions_ood)
        out[name] = {
            "auroc": ood_auroc(id_s, ood_s),
            "auprc": ood_auprc(id_s, ood_s),
            "fpr_at_95_tpr": fpr_at_95_tpr(id_s, ood_s),
        }
    return out
