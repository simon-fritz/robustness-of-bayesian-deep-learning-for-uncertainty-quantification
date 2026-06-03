"""Pre-trained torchvision ResNets (ResNet-18 / ResNet-50) for MedMNIST.

A second architecture family alongside :class:`SmallCNN`, kept switchable via
config so results can be compared across architectures (see
``docs/architecture_choice.md``). ResNet-18 is the default; ResNet-50 shares the
exact same code path and is used only when compute/time allow.

The torchvision ResNet submodules are adopted as *direct* attributes
(``conv1`` ... ``layer4``, ``fc``) rather than nested under a sub-module. This
keeps the final classifier reachable as ``model.fc`` — the exact name that
``methods.last_layer_laplace`` targets — so last-layer Laplace works unchanged
on every variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from .base import BaseModel

# torchvision ResNets always consume 3-channel input; MedMNIST grayscale is
# expanded to 3 channels on the data-loader side (see data/medmnist_loader.py).
_RESNET_IN_CHANNELS = 3

# Registry of supported depths: arch name -> (constructor, ImageNet weights enum).
_RESNET_VARIANTS: dict[str, tuple[Callable[..., nn.Module], object]] = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}


class _PretrainedResNet(BaseModel):
    """torchvision ResNet with a fresh ``fc`` head for ``num_classes``.

    Subclassed by the concrete depths via the ``_arch`` class attribute; the
    body is depth-agnostic because every torchvision ResNet exposes the same
    submodule names.

    Args:
        num_classes: Size of the new classifier head.
        weights: Weight source — ``"imagenet"`` (ImageNet-pretrained, default),
            ``"random"`` (train from scratch), or a filesystem path to a custom
            checkpoint (a ``state_dict`` compatible with this module or with a
            bare torchvision ResNet backbone of the same depth).
        freeze_backbone: When ``True``, freeze everything except ``fc`` (linear
            probing). When ``False`` (default), full fine-tuning.
        in_channels: Expected input channels; must be 3.
    """

    #: torchvision architecture key; set by concrete subclasses.
    _arch: str = "resnet18"

    def __init__(
        self,
        num_classes: int = 2,
        weights: str = "imagenet",
        freeze_backbone: bool = False,
        in_channels: int = _RESNET_IN_CHANNELS,
    ) -> None:
        super().__init__()
        if int(in_channels) != _RESNET_IN_CHANNELS:
            raise ValueError(
                f"{type(self).__name__} expects {_RESNET_IN_CHANNELS}-channel "
                f"input, got in_channels={in_channels}. Configure the data "
                f"loader's image_transform with expand_channels_to: 3."
            )
        self.in_channels = _RESNET_IN_CHANNELS
        self.num_classes = num_classes
        self.weights_source = str(weights)
        self.freeze_backbone = bool(freeze_backbone)

        backbone = self._build_backbone(self.weights_source)

        # Replace the 1000-class ImageNet head with a fresh classifier.
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        # Adopt submodules as direct attributes so ``self.fc`` is top-level.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

        if self.freeze_backbone:
            self._freeze_all_but_fc()

    # ------------------------------------------------------------------
    @classmethod
    def _build_backbone(cls, weights: str) -> nn.Module:
        """Instantiate the torchvision backbone for the requested weight source."""
        ctor, imagenet_weights = _RESNET_VARIANTS[cls._arch]
        source = weights.lower()
        if source in ("imagenet", "imagenet1k_v1", "imagenet1k", "default"):
            return ctor(weights=imagenet_weights)
        if source in ("random", "none", "scratch"):
            return ctor(weights=None)
        # Otherwise treat ``weights`` as a path to a custom checkpoint.
        ckpt = Path(weights).expanduser()
        if not ckpt.is_file():
            raise ValueError(
                f"weights={weights!r} is neither a known source "
                f"('imagenet'|'random') nor an existing checkpoint path."
            )
        backbone = ctor(weights=None)
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Allow either a bare backbone state or a _PretrainedResNet state.
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[{cls.__name__}] loaded custom weights from {ckpt} "
                f"(missing={len(missing)}, unexpected={len(unexpected)} keys)",
                flush=True,
            )
        return backbone

    def _freeze_all_but_fc(self) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("fc.")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def feature_layer_names(self) -> Iterable[str]:
        return ("layer1", "layer2", "layer3", "layer4")

    def classifier_layer_name(self) -> str:
        return "fc"


class PretrainedResNet18(_PretrainedResNet):
    """ImageNet-pretrained ResNet-18, the project's default architecture."""

    _arch = "resnet18"


class PretrainedResNet50(_PretrainedResNet):
    """ImageNet-pretrained ResNet-50; used only when compute/time allow.

    Identical interface and code path to :class:`PretrainedResNet18` — see
    ``docs/architecture_choice.md`` for when we reach for the deeper variant.
    """

    _arch = "resnet50"
