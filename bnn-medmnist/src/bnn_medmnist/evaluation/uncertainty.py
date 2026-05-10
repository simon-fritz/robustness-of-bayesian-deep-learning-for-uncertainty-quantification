"""Uncertainty decomposition.

Splits predictive uncertainty into aleatoric (data) and epistemic (model)
components from MC samples of class probabilities.
"""

from __future__ import annotations

import torch


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Entropy of the mean predictive distribution. Shape: (batch,)."""
    # TODO: implement
    raise NotImplementedError


def expected_entropy(probs_samples: torch.Tensor) -> torch.Tensor:
    """Mean entropy across MC samples (aleatoric proxy). Shape: (batch,)."""
    # TODO: implement — input shape (n_samples, batch, num_classes).
    raise NotImplementedError


def mutual_information(probs_samples: torch.Tensor) -> torch.Tensor:
    """BALD / mutual information (epistemic). Shape: (batch,)."""
    # TODO: implement — predictive_entropy(mean) - expected_entropy(samples).
    raise NotImplementedError
