"""OOD scenario definitions and loader builders.

We support three conceptually different OOD scenarios, all sharing the same
basic shape of (ID test loader, dict-of-OOD-loaders):

* ``far_ood``         — external dataset, visually very different modality.
* ``near_ood``        — external dataset, similar modality / different task.
* ``held_out_class``  — within-dataset: classes excluded from training are
                        treated as OOD at test time. Requires a model trained
                        with the matching ``exclude_classes`` data setting.
* ``long_tail``       — within-dataset: classes heavily subsampled in training
                        are treated as a low-density "tail" region at test
                        time. Requires a model trained with the matching
                        ``class_subsampling`` data setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from medmnist import INFO
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .medmnist_loader import MedMNISTLoader


Scenario = Literal["far_ood", "near_ood", "held_out_class", "long_tail"]


@dataclass
class OODPair:
    """Specification of an OOD evaluation scenario."""

    scenario: Scenario
    id_dataset: str
    # for far_ood / near_ood:
    ood_datasets: list[str] | None = None
    # for held_out_class / long_tail:
    held_out_classes: list[int] | None = None
    tail_classes: list[int] | None = None
    split: str = "test"

    def __post_init__(self) -> None:
        if self.scenario in ("far_ood", "near_ood") and not self.ood_datasets:
            raise ValueError(f"scenario={self.scenario} requires ood_datasets")
        if self.scenario == "held_out_class" and not self.held_out_classes:
            raise ValueError("scenario=held_out_class requires held_out_classes")
        if self.scenario == "long_tail" and not self.tail_classes:
            raise ValueError("scenario=long_tail requires tail_classes")


def _make_data_cfg(flag: str, batch_size: int, num_workers: int, data_root: str):
    return OmegaConf.create(
        {
            "name": flag,
            "flag": flag,
            "root": data_root,
            "download": True,
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
        }
    )


def _external_ood_loader(
    flag: str,
    target_channels: int,
    batch_size: int,
    num_workers: int,
    data_root: str,
    image_transform_cfg=None,
) -> DataLoader:
    """Build a test loader for an external MedMNIST dataset, adapting channels.

    Channel mismatch (e.g. RGB BloodMNIST vs grayscale ID model) is resolved by
    converting OOD inputs to the model's expected channel count via
    ``transforms.Grayscale`` (RGB->1) or by channel-replicating (1->3).
    This is a known confound — it means we score the model on inputs that have
    been pushed through a non-trivial preprocessing step that ID inputs do not
    see. Flagged in the report.

    When ``image_transform_cfg`` is provided (e.g. the ImageNet-style pipeline
    for the pretrained ResNet-18), the *same* transform builder is used as for
    the ID loader, so ID and OOD tensors share resolution / channels /
    normalization. The transform's ``expand_channels_to`` handles channel
    adaptation in that case.
    """
    import os
    from pathlib import Path

    import medmnist
    from torchvision import transforms

    info = INFO[flag]
    n_src_channels = int(info["n_channels"])
    DataClass = getattr(medmnist, info["python_class"])

    from .medmnist_loader import PACKAGE_ROOT, _squeeze_target, build_image_transform

    if image_transform_cfg is not None:
        # Shared ImageNet-style pipeline (resize / channel-expand / normalize).
        tfm = build_image_transform(n_src_channels, image_transform_cfg)
    else:
        # Legacy [-1, 1] pipeline with explicit channel adaptation to the model.
        ops: list = []
        if n_src_channels != target_channels:
            if target_channels == 1 and n_src_channels == 3:
                ops.append(transforms.Grayscale(num_output_channels=1))
            elif target_channels == 3 and n_src_channels == 1:
                ops += [transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1))]
            else:
                raise ValueError(
                    f"unsupported channel adaptation {n_src_channels} -> {target_channels}"
                )
        if not any(isinstance(op, transforms.ToTensor) for op in ops):
            ops.append(transforms.ToTensor())
        mean = [0.5] * target_channels
        std = [0.5] * target_channels
        ops.append(transforms.Normalize(mean, std))
        tfm = transforms.Compose(ops)

    root = Path(os.path.expanduser(str(data_root)))
    if not root.is_absolute():
        root = PACKAGE_ROOT / root
    os.makedirs(root, exist_ok=True)
    ds = DataClass(
        split="test", transform=tfm, target_transform=_squeeze_target,
        download=True, root=str(root),
    )
    return DataLoader(
        ds, batch_size=int(batch_size), shuffle=False,
        num_workers=int(num_workers), pin_memory=torch.cuda.is_available(),
    )


def build_ood_loaders(
    ood_pair: OODPair,
    *,
    id_loader: MedMNISTLoader,
    batch_size: int,
    num_workers: int,
    data_root: str,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Build (id_test_loader, {ood_name: ood_loader, ...}) for the scenario.

    ``id_loader`` is the already-constructed ID loader for the trained run; we
    re-use it so any train-side filters it applied are honored, and so the
    target channel count is known.
    """
    target_channels = id_loader.metadata.in_channels
    image_transform_cfg = id_loader.image_transform_cfg

    if ood_pair.scenario in ("far_ood", "near_ood"):
        id_test = id_loader.test_loader()
        ood_loaders: dict[str, DataLoader] = {}
        for flag in ood_pair.ood_datasets or []:
            ood_loaders[flag] = _external_ood_loader(
                flag, target_channels=target_channels,
                batch_size=batch_size, num_workers=num_workers,
                data_root=data_root, image_transform_cfg=image_transform_cfg,
            )
        return id_test, ood_loaders

    if ood_pair.scenario == "held_out_class":
        classes = list(ood_pair.held_out_classes or [])
        id_test = id_loader.test_loader_filtered(classes, include=False)
        ood_test = id_loader.test_loader_filtered(classes, include=True)
        return id_test, {f"held_out_{'_'.join(map(str, classes))}": ood_test}

    if ood_pair.scenario == "long_tail":
        classes = list(ood_pair.tail_classes or [])
        id_test = id_loader.test_loader_filtered(classes, include=False)
        tail_test = id_loader.test_loader_filtered(classes, include=True)
        n_id = len(id_test.dataset)
        n_tail = len(tail_test.dataset)
        if n_id == 0 or n_tail == 0:
            raise ValueError(
                f"long_tail split is empty (ID={n_id}, tail={n_tail}) for "
                f"classes={classes}. Check that the tail classes exist in the "
                f"test split of {ood_pair.id_dataset}."
            )
        if n_id < 50 or n_tail < 50:
            print(
                f"[ood_pairs] WARNING: long_tail split is small "
                f"(ID={n_id}, tail={n_tail}); AUROC/AUPRC will be noisy.",
                flush=True,
            )
        return id_test, {f"tail_{'_'.join(map(str, classes))}": tail_test}

    raise ValueError(f"unknown scenario: {ood_pair.scenario}")


def ood_pair_from_cfg(cfg) -> OODPair:
    """Construct an :class:`OODPair` from an OmegaConf experiment config."""
    return OODPair(
        scenario=str(cfg.scenario),
        id_dataset=str(cfg.id_dataset),
        ood_datasets=list(cfg.get("ood_datasets") or []) or None,
        held_out_classes=list(cfg.get("held_out_classes") or []) or None,
        tail_classes=list(cfg.get("tail_classes") or []) or None,
        split=str(cfg.get("split", "test")),
    )
