"""Uncertainty decomposition from MC samples of softmax probabilities.

Inputs are tensors of shape ``(n_samples, batch, num_classes)`` holding
posterior-sample softmax probabilities ``p(y | x, theta_s)``.

The standard decomposition (Depeweg et al. 2018; Kwon et al. 2020) is

    H[E_q p(y|x,theta)]   =   E_q H[p(y|x,theta)]   +   I[y; theta | x]
    -------------------       ----------------------     ----------------
    total predictive          aleatoric (expected)       epistemic (BALD)
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(p * p.clamp_min(_EPS).log()).sum(dim=dim)


def predictive_entropy(probs_samples: torch.Tensor) -> torch.Tensor:
    """Total predictive uncertainty: ``H[ E_q p(y|x,theta) ]``. Shape ``(B,)``."""
    mean = probs_samples.mean(dim=0)
    return _entropy(mean, dim=-1)


def expected_entropy(probs_samples: torch.Tensor) -> torch.Tensor:
    """Aleatoric proxy: ``E_q[ H[ p(y|x,theta) ] ]``. Shape ``(B,)``."""
    return _entropy(probs_samples, dim=-1).mean(dim=0)


def mutual_information(probs_samples: torch.Tensor) -> torch.Tensor:
    """Epistemic uncertainty (BALD): ``I[y; theta | x] = predictive - expected``.
    Shape ``(B,)``.
    """
    return predictive_entropy(probs_samples) - expected_entropy(probs_samples)


def predictive_variance(probs_samples: torch.Tensor) -> torch.Tensor:
    """Per-class variance of the predicted probabilities across MC samples.
    Shape ``(B, C)`` — caller can sum/mean if a scalar per example is wanted.
    """
    return probs_samples.var(dim=0, unbiased=False)
