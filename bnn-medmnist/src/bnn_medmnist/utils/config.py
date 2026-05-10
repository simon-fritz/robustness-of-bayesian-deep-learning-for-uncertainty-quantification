"""Hydra config loading helpers.

Thin wrappers around ``hydra.compose`` so that ``scripts/train.py`` and
``scripts/evaluate.py`` can load a single experiment YAML without bringing in
Hydra's app-decorator machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_experiment_config(path: str | Path) -> Any:
    """Load and compose an experiment config from a YAML path.

    Resolves OmegaConf interpolations (e.g. ``${oc.env:DATA_ROOT,...}``).
    """
    # TODO: implement using hydra.initialize_config_dir + hydra.compose,
    # or OmegaConf.load + OmegaConf.resolve for the simple single-file case.
    raise NotImplementedError
