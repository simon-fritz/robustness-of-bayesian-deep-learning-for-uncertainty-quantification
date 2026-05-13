"""Small CNN baseline for 28x28 MedMNIST inputs.

Layout:
    layer1: Conv-BN-ReLU-Pool        -> 14x14
    layer2: Conv-BN-ReLU-Pool        -> 7x7
    layer3: Conv-BN-ReLU             -> 7x7
    layer4: AdaptiveAvgPool-Flatten  -> feature vector
    fc:     Linear classifier head   -> logits
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .base import BaseModel


class SmallCNN(BaseModel):
    """Compact CNN with named submodules; ``fc`` is the last-layer Laplace target."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.0,
        channels: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        c1, c2, c3 = channels

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
        )
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(x)

    def feature_layer_names(self) -> Iterable[str]:
        return ("layer1", "layer2", "layer3", "layer4")

    def classifier_layer_name(self) -> str:
        return "fc"
