"""Smoke tests for the class-exclusion / subsampling features of MedMNISTLoader."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

pytest.importorskip("medmnist")

from bnn_medmnist.data.medmnist_loader import MedMNISTLoader, _labels_array
from torch.utils.data import Subset


def _bloodmnist_cfg(**overrides):
    cfg = {
        "name": "bloodmnist", "flag": "bloodmnist", "root": "./data",
        "download": True, "batch_size": 32, "num_workers": 0, "seed": 0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _labels_of(ds):
    if isinstance(ds, Subset):
        return _labels_array(ds.dataset)[ds.indices]
    return _labels_array(ds)


def test_exclude_classes_removes_from_train_keeps_in_test():
    loader = MedMNISTLoader(_bloodmnist_cfg(exclude_classes=[7]))
    train_labels = _labels_of(loader._train)
    val_labels = _labels_of(loader._val)
    test_labels = _labels_of(loader._test)
    assert (train_labels == 7).sum() == 0
    assert (val_labels == 7).sum() == 0
    assert (test_labels == 7).sum() > 0  # held-out class still present in test
    assert loader.held_out_or_tail_classes() == [7]


def test_class_subsampling_reduces_class_count():
    loader_full = MedMNISTLoader(_bloodmnist_cfg())
    full_train = _labels_of(loader_full._train)
    full_count_7 = int((full_train == 7).sum())
    assert full_count_7 > 0

    loader = MedMNISTLoader(_bloodmnist_cfg(class_subsampling={7: 0.02}))
    train_labels = _labels_of(loader._train)
    n_kept = int((train_labels == 7).sum())
    expected = round(0.02 * full_count_7)
    assert n_kept == expected
    # Other classes untouched.
    other = full_train[full_train != 7]
    other_after = train_labels[train_labels != 7]
    assert len(other) == len(other_after)


def test_test_loader_filtered_partitions_test_set():
    loader = MedMNISTLoader(_bloodmnist_cfg(exclude_classes=[7]))
    ood_loader = loader.test_loader_filtered([7], include=True)
    id_loader = loader.test_loader_filtered([7], include=False)
    ood_labels = []
    for _x, y in ood_loader:
        ood_labels += y.tolist()
    id_labels = []
    for _x, y in id_loader:
        id_labels += y.tolist()
    assert set(ood_labels) == {7}
    assert 7 not in set(id_labels)
    assert len(ood_labels) + len(id_labels) == len(_labels_of(loader._test))
