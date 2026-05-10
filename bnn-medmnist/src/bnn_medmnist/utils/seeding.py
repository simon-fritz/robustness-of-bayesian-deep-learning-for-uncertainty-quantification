"""Reproducibility — set seeds for python, numpy, torch (CPU + CUDA)."""

from __future__ import annotations


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python's ``random``, numpy, and torch (CPU + CUDA).

    If ``deterministic`` is True, also enable cuDNN deterministic mode (slower).
    """
    # TODO: implement
    raise NotImplementedError
