"""Training entry point.

Usage:
    python scripts/train.py --config configs/experiment/training/pneumonia_baseline.yaml
    python scripts/train.py --config configs/experiment/training/pneumonia_lll.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from bnn_medmnist.data.medmnist_loader import MedMNISTLoader
from bnn_medmnist.methods.deterministic import Deterministic
from bnn_medmnist.methods.last_layer_laplace import LastLayerLaplace
from bnn_medmnist.methods.deep_ensemble import DeepEnsemble
from bnn_medmnist.models import build_model
from bnn_medmnist.utils.config import load_experiment_config
from bnn_medmnist.utils.logging import log_run_start
from bnn_medmnist.utils.seeding import set_seed

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


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


def _assert_input_shape(model_cfg, data) -> None:
    """Fail loudly if the data loader's output shape mismatches the model.

    Compares the model config's declared ``input_channels`` / ``input_resolution``
    (when present) against what the loader actually produces, and cross-checks a
    real batch so a misconfigured transform is caught before training starts.
    """
    produced_c = int(data.metadata.in_channels)
    produced_r = int(data.metadata.image_size)
    exp_c = model_cfg.get("input_channels", None)
    exp_r = model_cfg.get("input_resolution", None)
    name = str(model_cfg.get("name", "model"))

    if exp_c is not None and int(exp_c) != produced_c:
        raise SystemExit(
            f"[train] input-channel mismatch for model '{name}': expects "
            f"{int(exp_c)} channels but the data loader produces {produced_c}. "
            f"Fix data.image_transform.expand_channels_to."
        )
    if exp_r is not None and int(exp_r) != produced_r:
        raise SystemExit(
            f"[train] input-resolution mismatch for model '{name}': expects "
            f"{int(exp_r)}x{int(exp_r)} but the data loader produces "
            f"{produced_r}x{produced_r}. Fix data.image_transform.resize."
        )

    # Cross-check against a real batch (catches transform bugs metadata misses).
    x0, _ = next(iter(data.train_loader()))
    got_c, got_h, got_w = int(x0.shape[1]), int(x0.shape[2]), int(x0.shape[3])
    if got_c != produced_c or got_h != produced_r or got_w != produced_r:
        raise SystemExit(
            f"[train] data loader produced tensor of shape {tuple(x0.shape)} "
            f"(C={got_c}, {got_h}x{got_w}) which disagrees with metadata "
            f"(C={produced_c}, {produced_r}x{produced_r}). Check image_transform."
        )
    print(f"[train] input shape OK: model '{name}' <- C={produced_c}, "
          f"{produced_r}x{produced_r}", flush=True)


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
    if name == "deep_ensemble":
        n_members = int(cfg.method.get("n_members", 5))
        return DeepEnsemble(
            train_cfg=train_cfg, ckpt_path=ckpt_path, log_dir=tb_dir, 
            class_weights=class_weights, n_members=n_members
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
    run_dir = PACKAGE_ROOT / "outputs" / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    tb_dir = PACKAGE_ROOT / "logs" / "tensorboard" / f"{run_name}_{timestamp}"
    ckpt_path = PACKAGE_ROOT / "checkpoints" / run_name / timestamp / "best.pt"

    log_run_start(
        run_dir=run_dir, config=cfg,
        extra={"run_dir": str(run_dir), "checkpoint": str(ckpt_path), "tensorboard": str(tb_dir)},
    )

    data = MedMNISTLoader(cfg.data)
    print(f"class_distribution(train) = {data.class_distribution()}", flush=True)

    _assert_input_shape(cfg.model, data)
    model = build_model(
        cfg.model,
        in_channels=data.metadata.in_channels,
        num_classes=data.metadata.num_classes,
    )
    method = _build_method(cfg, data, ckpt_path, tb_dir)
    method.fit(model, data.train_loader(), data.val_loader())

    if hasattr(method, "sigma_summary") and getattr(method, "la", None) is not None:
        getter = getattr(cfg.data, "get", None)
        train_size = int(getter("train_size", None) or -1) if getter else None
        summary = method.sigma_summary(train_size=train_size if train_size and train_size > 0 else None)
        (run_dir / "sigma_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"sigma_summary: {summary}", flush=True)

    (run_dir / "checkpoint_path.txt").write_text(str(ckpt_path))
    print(f"checkpoint saved to {ckpt_path}")
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
