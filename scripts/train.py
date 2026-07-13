"""Training entry point.

Usage:
    python scripts/train.py --config configs/experiment/training/pneumonia_baseline.yaml
    python scripts/train.py --config configs/experiment/training/pneumonia_lll.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from bnn_medmnist.data.medmnist_loader import MedMNISTLoader
from bnn_medmnist.methods.deterministic import Deterministic
from bnn_medmnist.methods.first_layer_laplace import FirstLayerLaplace
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


def _readable_file(path: Path) -> bool:
    """``path`` exists, is a file, and is readable by us.

    ``Path.is_file()`` *raises* ``PermissionError`` when a parent directory is
    unreadable (e.g. a teammate's checkpoint referenced from a shared repo),
    so it can't be used as a plain boolean — hence the explicit guard. Also
    checks ``os.access`` so we don't return a path we can't actually open.
    """
    try:
        return path.is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def _resolve_map_checkpoint(cfg) -> Path | None:
    """Resolve a reusable MAP checkpoint for post-hoc methods.

    Two config keys on ``cfg.method`` (both optional):
      * ``map_checkpoint`` — explicit path to a ``best.pt``.
      * ``reuse_map_from`` — run name under ``outputs/``; the newest finished
        run whose saved config matches ``cfg.seed`` *and whose checkpoint we
        can actually read* is used (its ``checkpoint_path.txt`` points at the
        checkpoint). Runs whose checkpoint is missing or unreadable (e.g. a
        teammate's run committed into a shared ``outputs/`` but stored under
        their private home) are skipped, not fatal.

    Errors out loudly rather than silently retraining — remove the key from
    the config if retraining is intended.
    """
    explicit = cfg.method.get("map_checkpoint", None)
    if explicit:
        ckpt = Path(explicit).expanduser()
        if not _readable_file(ckpt):
            raise SystemExit(
                f"[train] map_checkpoint={ckpt} does not exist or is unreadable."
            )
        return ckpt

    source = cfg.method.get("reuse_map_from", None)
    if not source:
        return None
    seed = int(cfg.seed)
    root = PACKAGE_ROOT / "outputs" / str(source)
    skipped: list[str] = []
    # Newest timestamp first, so re-runs pick the latest checkpoint per seed.
    for cfg_file in sorted(root.glob("*/config.yaml"), reverse=True):
        try:
            run_cfg = OmegaConf.load(cfg_file)
            if int(run_cfg.seed) != seed:
                continue
            ckpt_file = cfg_file.parent / "checkpoint_path.txt"
            if not _readable_file(ckpt_file):
                continue  # run crashed before saving, or pointer unreadable
            ckpt = Path(ckpt_file.read_text().strip())
        except OSError:  # unreadable run dir / pointer -> skip this candidate
            continue
        if _readable_file(ckpt):
            print(f"[train] reuse_map_from={source}: found seed={seed} "
                  f"checkpoint at {ckpt}", flush=True)
            return ckpt
        skipped.append(str(ckpt))
    hint = ""
    if skipped:
        hint = (" Skipped matching runs whose checkpoint was missing/unreadable"
                f" (e.g. {skipped[0]}) — likely another user's run in a shared"
                " outputs/ dir.")
    raise SystemExit(
        f"[train] reuse_map_from={source}: no readable seed={seed} checkpoint "
        f"under {root}.{hint} Train the MAP model first, point map_checkpoint "
        f"at an explicit path, or remove the key to retrain from scratch."
    )


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
    if name == "first_layer_laplace":
        return FirstLayerLaplace(
            train_cfg=train_cfg, laplace_cfg=cfg.method.laplace,
            ckpt_path=ckpt_path, log_dir=tb_dir, class_weights=class_weights,
            bayesian_modules=list(cfg.method.get("bayesian_layers", ["conv1"])),
            map_checkpoint=_resolve_map_checkpoint(cfg),
        )
    if name == "deep_ensemble":
        n_members = int(cfg.method.get("n_members", 5))
        return DeepEnsemble(
            train_cfg=train_cfg, ckpt_path=ckpt_path, log_dir=tb_dir, 
            class_weights=class_weights, n_members=n_members, base_seed=int(cfg.seed)
        )
    raise NotImplementedError(f"method '{name}' is not wired up.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model under a Bayesian method.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in config (for multi-seed sweeps).")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if args.smoke:
        _apply_smoke_overrides(cfg)
    if args.epochs is not None:
        _training_block(cfg.method).epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(int(cfg.seed))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name") or cfg.get("experiment_name") or "run"
    run_dir = PACKAGE_ROOT / "outputs" / run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # TensorBoard logs default to the repo's logs/ dir, but that may sit on a
    # shared network mount that throws OSError on append. TENSORBOARD_DIR lets a
    # SLURM job redirect them to node-local scratch ($TMPDIR), or disable them
    # entirely with TENSORBOARD_DIR=off / "" / none.
    tb_env = os.environ.get("TENSORBOARD_DIR")
    if tb_env is None:
        tb_dir = PACKAGE_ROOT / "logs" / "tensorboard" / f"{run_name}_{timestamp}"
    elif tb_env.strip().lower() in ("", "off", "none"):
        tb_dir = None
    else:
        tb_dir = Path(tb_env) / f"{run_name}_{timestamp}"
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
    
    name = str(cfg.method.get("name", "deterministic")).lower()
    if name == "deep_ensemble":
        method.fit(model, data)
    else:
        method.fit(model, data.train_loader(), data.val_loader())

    (run_dir / "checkpoint_path.txt").write_text(str(ckpt_path))
    print(f"checkpoint saved to {ckpt_path}")

    if hasattr(method, "sigma_summary") and getattr(method, "la", None) is not None:
        getter = getattr(cfg.data, "get", None)
        train_size = int(getter("train_size", None) or -1) if getter else None
        try:
            summary = method.sigma_summary(train_size=train_size if train_size and train_size > 0 else None)
            (run_dir / "sigma_summary.json").write_text(json.dumps(summary, indent=2))
            print(f"sigma_summary: {summary}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: sigma_summary failed ({exc!r}); skipping.", flush=True)
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
