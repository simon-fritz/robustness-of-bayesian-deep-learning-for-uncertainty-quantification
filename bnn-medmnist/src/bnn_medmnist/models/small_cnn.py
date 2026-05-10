"""Small CNN baseline for 28x28 MedMNIST inputs.

Layout (sketch):
    layer1: Conv-BN-ReLU-Pool
    layer2: Conv-BN-ReLU-Pool
    layer3: Conv-BN-ReLU
    layer4: Conv-BN-ReLU-AdaptiveAvgPool
    fc:     Linear classifier head
"""

from __future__ import annotations

from typing import Iterable

import torch

from .base import BaseModel


class SmallCNN(BaseModel):
    """Compact 4-block CNN with named submodules."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        # TODO: implement — define layer1..layer4 and fc.
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        raise NotImplementedError

    def feature_layer_names(self) -> Iterable[str]:
        return ("layer1", "layer2", "layer3", "layer4")

    def classifier_layer_name(self) -> str:
        return "fc"
