"""Network architectures.

Architectures are switchable via ``cfg.model.name``:

* ``small_cnn``          — :class:`SmallCNN`, compact from-scratch baseline.
* ``pretrained_resnet18``— :class:`PretrainedResNet18`, ImageNet-pretrained,
  fine-tuned — the project default (see ``docs/architecture_choice.md``).
* ``pretrained_resnet50``— :class:`PretrainedResNet50`, same code path, used
  only when compute/time allow.

Use :func:`build_model` so entry points stay architecture-agnostic.
"""

from __future__ import annotations

import torch.nn as nn

from .resnet import PretrainedResNet18, PretrainedResNet50
from .small_cnn import SmallCNN


def build_model(model_cfg, in_channels: int, num_classes: int) -> nn.Module:
    """Construct a model from its config block.

    Args:
        model_cfg: The ``cfg.model`` block (must have a ``name`` field).
        in_channels: Channels the data loader actually produces.
        num_classes: Number of target classes (from the data config).
    """
    name = str(model_cfg.get("name", "small_cnn")).lower()
    if name == "small_cnn":
        return SmallCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name in ("pretrained_resnet18", "pretrained_resnet50"):
        cls = (
            PretrainedResNet18
            if name == "pretrained_resnet18"
            else PretrainedResNet50
        )
        return cls(
            num_classes=num_classes,
            weights=str(model_cfg.get("weights", "imagenet")),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
            in_channels=in_channels,
        )
    raise ValueError(f"unknown model name: {name!r}")


__all__ = [
    "SmallCNN",
    "PretrainedResNet18",
    "PretrainedResNet50",
    "build_model",
]
