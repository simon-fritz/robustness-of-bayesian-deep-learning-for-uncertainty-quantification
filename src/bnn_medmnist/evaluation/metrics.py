"""Classification metrics.

All functions take ``(y_true, y_pred_probs)`` where ``y_pred_probs`` is shaped
``(N, C)``. Inputs may be torch tensors or numpy arrays.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _to_np(*arrs):
    out = []
    for a in arrs:
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        out.append(np.asarray(a))
    return out


def accuracy(y_true, y_pred_probs) -> float:
    y_true, p = _to_np(y_true, y_pred_probs)
    return float((p.argmax(axis=-1) == y_true).mean())


def balanced_accuracy(y_true, y_pred_probs) -> float:
    y_true, p = _to_np(y_true, y_pred_probs)
    return float(balanced_accuracy_score(y_true, p.argmax(axis=-1)))


def auroc(y_true, y_pred_probs) -> float:
    y_true, p = _to_np(y_true, y_pred_probs)
    if p.ndim == 2 and p.shape[1] == 2:
        return float(roc_auc_score(y_true, p[:, 1]))
    return float(roc_auc_score(y_true, p, multi_class="ovr", average="macro"))


def expected_calibration_error(y_true, y_pred_probs, n_bins: int = 15) -> float:
    y_true, p = _to_np(y_true, y_pred_probs)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(conf[mask].mean() - correct[mask].mean()) * mask.sum() / n
    return float(ece)


def nll(y_true, y_pred_probs) -> float:
    y_true, p = _to_np(y_true, y_pred_probs)
    eps = 1e-12
    idx = np.arange(len(y_true))
    return float(-np.log(np.clip(p[idx, y_true], eps, 1.0)).mean())


def brier_score(y_true, y_pred_probs) -> float:
    y_true, p = _to_np(y_true, y_pred_probs)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((p - onehot) ** 2).sum(axis=-1).mean())
