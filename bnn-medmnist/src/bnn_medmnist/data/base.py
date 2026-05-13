"""Abstract base for dataset providers.

Every concrete dataset wrapper exposes train/val/test DataLoaders and basic
metadata (num_classes, in_channels, image_size). Concrete implementations live
alongside this module (see ``medmnist_loader.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetMetadata:
    """Static metadata about a dataset, independent of split."""

    name: str
    num_classes: int
    in_channels: int
    image_size: int


class BaseDataset(ABC):
    """Abstract dataset provider."""

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return static metadata for this dataset."""
        raise NotImplementedError

    @abstractmethod
    def train_loader(self) -> Any:
        """Return a DataLoader over the training split."""
        raise NotImplementedError

    @abstractmethod
    def val_loader(self) -> Any:
        """Return a DataLoader over the validation split."""
        raise NotImplementedError

    @abstractmethod
    def test_loader(self) -> Any:
        """Return a DataLoader over the test split."""
        raise NotImplementedError
