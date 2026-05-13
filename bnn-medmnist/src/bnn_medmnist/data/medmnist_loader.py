"""MedMNIST dataset loader."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms

from .base import BaseDataset, DatasetMetadata


def _squeeze_target(y):
    if hasattr(y, "__len__"):
        y = y[0]
    return int(y)


class MedMNISTLoader(BaseDataset):
    """Loader for any MedMNIST 2D dataset (PneumoniaMNIST, BloodMNIST, ...)."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        flag = cfg.flag
        info = INFO[flag]
        self.info = info
        n_channels = int(info["n_channels"])
        num_classes = len(info["label"])
        DataClass = getattr(medmnist, info["python_class"])

        # Normalize to [-1, 1] regardless of channel count.
        mean = [0.5] * n_channels
        std = [0.5] * n_channels
        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        root = Path(os.path.expanduser(str(cfg.root)))
        if not root.is_absolute():
            root = PACKAGE_ROOT / root
        root = str(root)
        os.makedirs(root, exist_ok=True)
        download = bool(cfg.get("download", True)) if hasattr(cfg, "get") else True

        kwargs = dict(transform=tfm, target_transform=_squeeze_target,
                      download=download, root=root)
        self._train = DataClass(split="train", **kwargs)
        self._val = DataClass(split="val", **kwargs)
        self._test = DataClass(split="test", **kwargs)

        self._meta = DatasetMetadata(
            name=flag, num_classes=num_classes,
            in_channels=n_channels, image_size=28,
        )
        self._batch_size = int(cfg.batch_size)
        self._num_workers = int(cfg.num_workers)

    @property
    def metadata(self) -> DatasetMetadata:
        return self._meta

    def class_distribution(self) -> dict[int, int]:
        labels = np.asarray(self._train.labels).flatten().tolist()
        return dict(Counter(int(x) for x in labels))

    def class_weights(self) -> torch.Tensor:
        labels = np.asarray(self._train.labels).flatten().astype(int)
        counts = np.bincount(labels, minlength=self._meta.num_classes)
        weights = len(labels) / (self._meta.num_classes * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32)

    def _make(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=self._batch_size, shuffle=shuffle,
            num_workers=self._num_workers, pin_memory=torch.cuda.is_available(),
        )

    def train_loader(self):
        return self._make(self._train, shuffle=True)

    def val_loader(self):
        return self._make(self._val, shuffle=False)

    def test_loader(self):
        return self._make(self._test, shuffle=False)
