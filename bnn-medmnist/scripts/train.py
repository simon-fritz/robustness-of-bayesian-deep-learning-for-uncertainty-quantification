"""Training entry point.

Usage:
    python scripts/train.py --config configs/experiment/pneumonia_lll.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model under a Bayesian method.")
    parser.add_argument("--config", required=True, help="Path to a composed experiment YAML.")
    args = parser.parse_args()  # noqa: F841

    # TODO: implement
    # 1. utils.config.load_experiment_config(args.config)
    # 2. utils.seeding.set_seed(cfg.seed)
    # 3. utils.logging.log_run_start(extra={"config": args.config})
    # 4. Build dataset, model, method, trainer; train; method.fit(); save.
    raise NotImplementedError


if __name__ == "__main__":
    main()
