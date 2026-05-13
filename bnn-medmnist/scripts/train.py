"""Training entry point.

Usage:
    python scripts/train.py --config configs/experiment/pneumonia_baseline.yaml
    python scripts/train.py --config configs/experiment/pneumonia_lll.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from bnn_medmnist.data.medmnist_loader import MedMNISTLoader
from bnn_medmnist.methods.deterministic import Deterministic
from bnn_medmnist.methods.last_layer_laplace import LastLayerLaplace
from bnn_medmnist.models.small_cnn import SmallCNN
from bnn_medmnist.utils.config import load_experiment_config
from bnn_medmnist.utils.logging import log_run_start
from bnn_medmnist.utils.seeding import set_seed


def _training_block(method_cfg):
    """Return the training-config block for a given method config.

    Deterministic configs are flat (training fields live at the top); other
    methods nest training under ``method.training``.
    """
    if "training" in method_cfg:
        return method_cfg.training
    return method_cfg


def _apply_smoke_overrides(cfg) -> None:
    train_cfg = _training_block(cfg.method)
    train_cfg.epochs = 2
    train_cfg.early_stopping_patience = 0
    if "laplace" in cfg.method:
        cfg.method.laplace.n_predictive_samples = 20


def _build_method(cfg, data, ckpt_path: Path, tb_dir: Path):
    name = str(cfg.method.get("name", "deterministic")).lower()
    train_cfg = _training_block(cfg.method)

    use_cw = bool(train_cfg.get("use_class_weights", False))
    class_weights = data.class_weights() if use_cw else None
    if class_weights is not None:
        print(f"class_weights = {class_weights.tolist()}", flush=True)

    if name == "deterministic":
        return Deterministic(
            train_cfg=train_cfg, ckpt_path=ckpt_path, log_dir=tb_dir,
            class_weights=class_weights,
        )
    if name == "last_layer_laplace":
        return LastLayerLaplace(
            train_cfg=train_cfg, laplace_cfg=cfg.method.laplace,
            ckpt_path=ckpt_path, log_dir=tb_dir, class_weights=class_weights,
        )
    raise NotImplementedError(f"method '{name}' is not wired up.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model under a Bayesian method.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.smoke:
        _apply_smoke_overrides(cfg)
    if args.epochs is not None:
        _training_block(cfg.method).epochs = args.epochs

    set_seed(int(cfg.seed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name") or cfg.get("experiment_name") or "run"
    run_dir = Path("outputs") / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    tb_dir = Path("logs/tensorboard") / f"{run_name}_{timestamp}"
    ckpt_path = Path("checkpoints") / run_name / timestamp / "best.pt"

    log_run_start(
        run_dir=run_dir, config=cfg,
        extra={"run_dir": str(run_dir), "checkpoint": str(ckpt_path), "tensorboard": str(tb_dir)},
    )

    data = MedMNISTLoader(cfg.data)
    print(f"class_distribution(train) = {data.class_distribution()}", flush=True)

    model = SmallCNN(
        in_channels=data.metadata.in_channels,
        num_classes=data.metadata.num_classes,
        dropout=float(cfg.model.get("dropout", 0.0)),
    )
    method = _build_method(cfg, data, ckpt_path, tb_dir)
    method.fit(model, data.train_loader(), data.val_loader())

    (run_dir / "checkpoint_path.txt").write_text(str(ckpt_path))
    print(f"checkpoint saved to {ckpt_path}")
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
