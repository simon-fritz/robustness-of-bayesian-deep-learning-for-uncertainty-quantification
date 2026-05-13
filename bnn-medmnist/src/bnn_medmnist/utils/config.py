"""Hydra config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def load_experiment_config(path: str | Path) -> Any:
    """Load and compose an experiment config from a YAML path under ``configs/``.

    The path is expected to be ``<configs_dir>/<group>/<name>.yaml`` so that
    Hydra can resolve ``defaults`` referring to sibling groups.
    """
    path = Path(path).resolve()
    group = path.parent.name
    name = path.stem
    configs_dir = path.parent.parent

    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        cfg = compose(config_name=f"{group}/{name}")
    # Hydra wraps the composed config under the leading group name; unwrap it
    # so callers see the experiment fields at the top level.
    if group in cfg and len(cfg) == 1:
        cfg = cfg[group]
    OmegaConf.resolve(cfg)
    return cfg
