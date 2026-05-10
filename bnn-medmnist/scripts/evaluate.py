"""Evaluation entry point.

Loads a trained method + checkpoint and computes ID metrics + OOD scores
against the configured OOD datasets.

Usage:
    python scripts/evaluate.py --config configs/experiment/pneumonia_lll.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Bayesian model.")
    parser.add_argument("--config", required=True, help="Path to a composed experiment YAML.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    args = parser.parse_args()  # noqa: F841

    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
