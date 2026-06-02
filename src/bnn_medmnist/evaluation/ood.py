"""OOD detection scoring.

Given a 1-D uncertainty score per sample for ID and OOD inputs, report AUROC,
AUPR, and FPR@95%TPR with OOD as the positive class (higher score = more
OOD-like).

The uncertainty scores are grouped into conceptual *families* (see
:data:`SCORE_CATEGORIES`) so the evaluation systematically compares different
ways of summarising second-order uncertainty — first-order confidence,
information-theoretic decomposition, direct statistical spread, and the
analytical Gaussian spread available to Laplace — rather than just piling up
entropy variants.

:func:`per_sample_scores` computes every applicable score from cached
predictions; :func:`evaluate_ood` turns ID/OOD predictions into the full
metric table.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .uncertainty import (
    expected_entropy,
    expected_pairwise_kl,
    logit_variance,
    mutual_information,
    predictive_entropy,
    softmax_predictive_variance,
)


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


# ---------------------------------------------------------------------------
# score families — each entry is a conceptual category of uncertainty measure
# ---------------------------------------------------------------------------
SCORE_CATEGORIES: dict[str, list[str]] = {
    "First-order (any model)": [
        "predictive_entropy",
        "one_minus_max_softmax",
    ],
    "Information-theoretic decomposition (MC samples)": [
        "mutual_information",
        "expected_entropy",
    ],
    "Statistical spread (MC samples)": [
        "softmax_variance_sum",
        "softmax_variance_max",
        "expected_pairwise_kl",
    ],
    "Analytical Gaussian spread (Laplace only)": [
        "logit_variance_sum",
        "logit_variance_max",
    ],
}

# Flat ordered list of every score name.
ALL_SCORE_NAMES: list[str] = [s for names in SCORE_CATEGORIES.values() for s in names]


def category_of(score_name: str) -> str | None:
    """Return the category a score belongs to, or ``None`` if unknown."""
    for cat, names in SCORE_CATEGORIES.items():
        if score_name in names:
            return cat
    return None


# Backwards-compatible mapping of the probs-only scores (no logit data needed).
# expected_pairwise_kl is excluded here because it may return ``None``; use
# :func:`per_sample_scores` for the full, N/A-aware set.
SCORE_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "predictive_entropy": predictive_entropy,
    "mutual_information": mutual_information,
    "expected_entropy": expected_entropy,
    "one_minus_max_softmax": one_minus_max_softmax,
    "softmax_variance_sum": lambda p: softmax_predictive_variance(p, "sum"),
    "softmax_variance_max": lambda p: softmax_predictive_variance(p, "max"),
}


def per_sample_scores(
    probs_samples: torch.Tensor,
    logit_mean: torch.Tensor | None = None,
    logit_var: torch.Tensor | None = None,
    *,
    min_kl_samples: int = 10,
) -> dict[str, torch.Tensor | None]:
    """Compute every applicable uncertainty score for one set of predictions.

    Scores that cannot be computed for the given inputs map to ``None``:

    * ``expected_pairwise_kl`` → ``None`` when ``n_samples < min_kl_samples``.
    * ``logit_variance_*``     → ``None`` when ``logit_var`` is ``None`` (e.g. a
      deterministic model with no posterior over weights).

    Args:
        probs_samples: ``(S, N, C)`` posterior-sample softmax probs (``S=1`` for
            deterministic models).
        logit_mean: ``(N, C)`` analytical logit mean, or ``None``.
        logit_var: ``(N, C)`` analytical logit variance, or ``None``.

    Returns:
        ``{score_name: per_sample_tensor (N,) or None}`` for every name in
        :data:`ALL_SCORE_NAMES`.
    """
    p = probs_samples
    has_logits = logit_var is not None
    logit_pair = (logit_mean, logit_var)
    return {
        "predictive_entropy": predictive_entropy(p),
        "one_minus_max_softmax": one_minus_max_softmax(p),
        "mutual_information": mutual_information(p),
        "expected_entropy": expected_entropy(p),
        "softmax_variance_sum": softmax_predictive_variance(p, "sum"),
        "softmax_variance_max": softmax_predictive_variance(p, "max"),
        "expected_pairwise_kl": expected_pairwise_kl(p, min_samples=min_kl_samples),
        "logit_variance_sum": logit_variance(logit_pair, "sum") if has_logits else None,
        "logit_variance_max": logit_variance(logit_pair, "max") if has_logits else None,
    }


def ood_metrics_from_scores(
    id_scores: dict[str, torch.Tensor | None],
    ood_scores: dict[str, torch.Tensor | None],
) -> dict[str, dict[str, float]]:
    """AUROC/AUPRC/FPR@95 for every score computable on *both* ID and OOD.

    Scores that are ``None`` on either side are skipped (not crashed on); the
    caller is responsible for reporting them as N/A.
    """
    out: dict[str, dict[str, float]] = {}
    for name in ALL_SCORE_NAMES:
        id_s = id_scores.get(name)
        ood_s = ood_scores.get(name)
        if id_s is None or ood_s is None:
            continue
        out[name] = {
            "auroc": ood_auroc(id_s, ood_s),
            "auprc": ood_auprc(id_s, ood_s),
            "fpr_at_95_tpr": fpr_at_95_tpr(id_s, ood_s),
        }
    return out


def evaluate_ood(
    predictions_id: torch.Tensor,
    predictions_ood: torch.Tensor,
    *,
    logit_id: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    logit_ood: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
    min_kl_samples: int = 10,
) -> dict[str, dict[str, float]]:
    """Compute AUROC/AUPRC/FPR@95 for every applicable uncertainty score.

    ``predictions_*`` are predictive-sample tensors of shape ``(S, N, C)`` —
    ``S=1`` for deterministic models, ``S>1`` for Bayesian methods. The optional
    ``logit_*`` tuples ``(logit_mean, logit_var)`` enable the analytical
    logit-variance scores (Laplace only); when omitted those scores are skipped.

    Returns ``{score_name: {auroc, auprc, fpr_at_95_tpr}}`` for every computable
    score (N/A scores are absent from the dict).
    """
    lm_id, lv_id = logit_id if logit_id is not None else (None, None)
    lm_ood, lv_ood = logit_ood if logit_ood is not None else (None, None)
    id_scores = per_sample_scores(predictions_id, lm_id, lv_id, min_kl_samples=min_kl_samples)
    ood_scores = per_sample_scores(predictions_ood, lm_ood, lv_ood, min_kl_samples=min_kl_samples)
    return ood_metrics_from_scores(id_scores, ood_scores)
