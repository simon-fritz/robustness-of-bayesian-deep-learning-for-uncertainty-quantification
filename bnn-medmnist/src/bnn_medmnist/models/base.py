"""Abstract base class for models.

All architectures expose named submodules (``layer1`` ... ``layerN``, ``fc``)
so that ``BayesianMethod.bayesian_layers`` can refer to them by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base for classification networks."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, num_classes)``."""
        raise NotImplementedError

    @abstractmethod
    def feature_layer_names(self) -> Iterable[str]:
        """Names of feature-extraction layers, in forward order.

        Used by methods to resolve ``bayesian_layers`` like
        ``["layer4", "fc"]`` into concrete submodules.
        """
        raise NotImplementedError

    @abstractmethod
    def classifier_layer_name(self) -> str:
        """Name of the final classifier (head) layer (typically ``"fc"``)."""
        raise NotImplementedError
