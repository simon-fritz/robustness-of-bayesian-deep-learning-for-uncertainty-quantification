"""Hydra config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def load_experiment_config(path: str | Path) -> Any:
    """Load and compose an experiment config from a YAML path under ``configs/``.

    The path must live somewhere beneath a directory named ``configs/``. The
    Hydra config name is the path's location relative to that root (without the
    ``.yaml`` suffix), so absolute ``defaults`` like ``/data`` resolve against
    sibling groups of the ``configs/`` root regardless of nesting depth.
    """
    path = Path(path).resolve()
    configs_dir = next(
        (p for p in path.parents if p.name == "configs"), path.parent.parent
    )
    rel = path.relative_to(configs_dir).with_suffix("")
    config_name = rel.as_posix()
    top_group = rel.parts[0] if len(rel.parts) > 1 else None

    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        cfg = compose(config_name=config_name)
    # Hydra wraps the composed config under the leading group name; unwrap it
    # so callers see the experiment fields at the top level.
    if top_group is not None and top_group in cfg and len(cfg) == 1:
        cfg = cfg[top_group]
        # Continue unwrapping nested groups (e.g. experiment/training/foo).
        for part in rel.parts[1:-1]:
            if part in cfg and len(cfg) == 1:
                cfg = cfg[part]
            else:
                break
    OmegaConf.resolve(cfg)
    return cfg
