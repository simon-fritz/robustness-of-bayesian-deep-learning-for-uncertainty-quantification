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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .base import BaseDataset, DatasetMetadata


def _squeeze_target(y):
    if hasattr(y, "__len__"):
        y = y[0]
    return int(y)


def _labels_array(ds) -> np.ndarray:
    """Return the integer label vector for a MedMNIST dataset object."""
    return np.asarray(ds.labels).flatten().astype(int)


class MedMNISTLoader(BaseDataset):
    """Loader for any MedMNIST 2D dataset (PneumoniaMNIST, BloodMNIST, ...).

    Optional config knobs (default to current behavior when omitted):
      * ``exclude_classes: list[int]`` — drop these classes from train/val
        entirely. Test set is left untouched so they can serve as OOD samples.
      * ``class_subsampling: dict[int, float]`` — keep the given fraction of
        each listed class in train/val (random subsample under the global
        seed). Test set is left untouched.
    """

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
        train_full = DataClass(split="train", **kwargs)
        val_full = DataClass(split="val", **kwargs)
        test_full = DataClass(split="test", **kwargs)

        getter = getattr(cfg, "get", None)
        exclude_classes = getter("exclude_classes", None) if getter else None
        class_subsampling = getter("class_subsampling", None) if getter else None
        self._exclude_classes: list[int] = (
            [int(c) for c in exclude_classes] if exclude_classes else []
        )
        # OmegaConf DictConfig keys may be strings; coerce to int.
        self._class_subsampling: dict[int, float] = (
            {int(k): float(v) for k, v in dict(class_subsampling).items()}
            if class_subsampling else {}
        )

        seed = int(getter("seed", 0)) if getter else 0
        self._train = self._apply_filters(train_full, split="train", seed=seed)
        self._val = self._apply_filters(val_full, split="val", seed=seed + 1)
        self._test = test_full

        self._meta = DatasetMetadata(
            name=flag, num_classes=num_classes,
            in_channels=n_channels, image_size=28,
        )
        self._batch_size = int(cfg.batch_size)
        self._num_workers = int(cfg.num_workers)

        if self._exclude_classes or self._class_subsampling:
            print(
                f"[{flag}] exclude_classes={self._exclude_classes} "
                f"class_subsampling={self._class_subsampling}",
                flush=True,
            )
            print(f"[{flag}] train per-class counts: {self._per_class_counts(self._train)}", flush=True)
            print(f"[{flag}] val   per-class counts: {self._per_class_counts(self._val)}", flush=True)
            print(f"[{flag}] test  per-class counts: {self._per_class_counts(self._test)}", flush=True)

    # ------------------------------------------------------------------
    # filtering helpers
    # ------------------------------------------------------------------
    def _apply_filters(self, ds, split: str, seed: int):
        if not self._exclude_classes and not self._class_subsampling:
            return ds
        labels = _labels_array(ds)
        keep = np.ones(len(labels), dtype=bool)
        if self._exclude_classes:
            keep &= ~np.isin(labels, self._exclude_classes)
        if self._class_subsampling:
            rng = np.random.default_rng(seed)
            for cls, frac in self._class_subsampling.items():
                cls_idx = np.where(labels == cls)[0]
                if len(cls_idx) == 0:
                    continue
                n_keep = int(round(frac * len(cls_idx)))
                drop_idx = rng.choice(cls_idx, size=len(cls_idx) - n_keep, replace=False)
                keep[drop_idx] = False
        indices = np.where(keep)[0].tolist()
        return Subset(ds, indices)

    def _per_class_counts(self, ds) -> dict[int, int]:
        if isinstance(ds, Subset):
            labels = _labels_array(ds.dataset)[ds.indices]
        else:
            labels = _labels_array(ds)
        return dict(sorted(Counter(int(x) for x in labels).items()))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> DatasetMetadata:
        return self._meta

    @property
    def excluded_classes(self) -> list[int]:
        return list(self._exclude_classes)

    @property
    def subsampled_classes(self) -> dict[int, float]:
        return dict(self._class_subsampling)

    def held_out_or_tail_classes(self) -> list[int]:
        """Classes that should be treated as OOD/tail at evaluation time."""
        return sorted(set(self._exclude_classes) | set(self._class_subsampling.keys()))

    def class_distribution(self) -> dict[int, int]:
        return self._per_class_counts(self._train)

    def class_weights(self) -> torch.Tensor:
        if isinstance(self._train, Subset):
            labels = _labels_array(self._train.dataset)[self._train.indices]
        else:
            labels = _labels_array(self._train)
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

    # Convenience used by OOD evaluation: build a test loader containing only
    # the requested classes (or, if ``include`` is False, excluding them).
    def test_loader_filtered(self, classes: list[int], include: bool = True) -> DataLoader:
        labels = _labels_array(self._test)
        mask = np.isin(labels, classes)
        if not include:
            mask = ~mask
        idx = np.where(mask)[0].tolist()
        return self._make(Subset(self._test, idx), shuffle=False)
