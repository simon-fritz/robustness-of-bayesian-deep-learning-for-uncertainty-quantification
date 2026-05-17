"""OOD evaluation entry point.

Pure post-hoc against a trained run for far_ood / near_ood scenarios. For
held_out_class and long_tail scenarios, the run's training config must already
match the OOD eval config (the script asserts this).

Usage:
    python scripts/evaluate_ood.py \\
        --run-dir outputs/pneumonia_baseline/<timestamp> \\
        --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve

from bnn_medmnist.data.medmnist_loader import MedMNISTLoader
from bnn_medmnist.data.ood_pairs import build_ood_loaders, ood_pair_from_cfg
from bnn_medmnist.evaluation.ood import SCORE_FNS, evaluate_ood
from bnn_medmnist.models.small_cnn import SmallCNN


# ---------------------------------------------------------------------------
# model + prediction helpers (mirror scripts/evaluate.py)
# ---------------------------------------------------------------------------
def _load_model(cfg, ckpt_path, device, num_classes, in_channels) -> SmallCNN:
    model = SmallCNN(
        in_channels=in_channels, num_classes=num_classes,
        dropout=float(cfg.model.get("dropout", 0.0)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _deterministic_samples(model, loader, device) -> torch.Tensor:
    all_p = []
    for x, _y in loader:
        all_p.append(torch.softmax(model(x.to(device)), dim=-1).cpu())
    return torch.cat(all_p).unsqueeze(0)  # (1, N, C)


@torch.no_grad()
def _laplace_samples(la, loader, device, n_samples: int) -> torch.Tensor:
    all_p = []
    for x, _y in loader:
        s = la.predictive_samples(x.to(device), pred_type="nn", n_samples=n_samples).cpu()
        all_p.append(s)
    return torch.cat(all_p, dim=1)  # (S, N, C)


def _predict(method_name, model, loader, device, n_samples, la=None) -> torch.Tensor:
    if method_name == "deterministic":
        return _deterministic_samples(model, loader, device)
    if method_name == "last_layer_laplace":
        return _laplace_samples(la, loader, device, n_samples)
    raise NotImplementedError(f"method '{method_name}' not supported by evaluate_ood")


# ---------------------------------------------------------------------------
# sanity check for within-dataset scenarios
# ---------------------------------------------------------------------------
def _assert_training_matches(run_cfg, ood_cfg) -> None:
    scenario = str(ood_cfg.scenario)
    data = run_cfg.data
    train_excl = list(data.get("exclude_classes") or [])
    train_sub = dict(data.get("class_subsampling") or {})
    # Normalize keys to int.
    train_sub = {int(k): float(v) for k, v in train_sub.items()}

    if scenario == "held_out_class":
        expected = sorted(int(c) for c in ood_cfg.held_out_classes)
        actual = sorted(int(c) for c in train_excl)
        if expected != actual:
            raise SystemExit(
                f"[evaluate_ood] held_out_class mismatch: trained run has "
                f"exclude_classes={actual} but OOD config requests {expected}. "
                f"Train a model with exclude_classes={expected} first."
            )
    elif scenario == "long_tail":
        expected = sorted(int(c) for c in ood_cfg.tail_classes)
        actual = sorted(train_sub.keys())
        if expected != actual:
            raise SystemExit(
                f"[evaluate_ood] long_tail mismatch: trained run has "
                f"class_subsampling={train_sub} but OOD config requests tail_classes={expected}. "
                f"Train a model with class_subsampling on these classes first."
            )
    elif scenario in ("far_ood", "near_ood"):
        # Pure post-hoc — but warn if training itself had filtering, since
        # then the ID distribution differs from the dataset's nominal one.
        if train_excl or train_sub:
            print(
                f"[evaluate_ood] WARNING: trained run has data filters "
                f"(exclude={train_excl}, subsample={train_sub}) — its ID test "
                f"loader will be filtered to those classes only.",
                flush=True,
            )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_histograms(scores_by_name, ood_name: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(scores_by_name), figsize=(4 * len(scores_by_name), 3.2))
    if len(scores_by_name) == 1:
        axes = [axes]
    for ax, (name, (id_s, ood_s)) in zip(axes, scores_by_name.items()):
        ax.hist(id_s, bins=40, alpha=0.55, label="ID", density=True)
        ax.hist(ood_s, bins=40, alpha=0.55, label="OOD", density=True)
        ax.set_title(name)
        ax.set_xlabel("score")
        ax.legend()
    fig.suptitle(f"score histograms — {ood_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"hist_{ood_name}.png", dpi=150)
    plt.close(fig)


def _plot_roc(scores_by_name, ood_name: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    for name, (id_s, ood_s) in scores_by_name.items():
        y = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])
        s = np.concatenate([id_s, ood_s])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {ood_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{ood_name}.png", dpi=150)
    plt.close(fig)


def _plot_auroc_bars(metrics_by_ood, out_dir: Path) -> None:
    score_names = list(SCORE_FNS.keys())
    ood_names = list(metrics_by_ood.keys())
    x = np.arange(len(score_names))
    width = 0.8 / max(len(ood_names), 1)
    fig, ax = plt.subplots(figsize=(1.8 * len(score_names) + 2, 3.5))
    for i, ood_name in enumerate(ood_names):
        vals = [metrics_by_ood[ood_name][s]["auroc"] for s in score_names]
        ax.bar(x + i * width, vals, width, label=ood_name)
    ax.set_xticks(x + width * (len(ood_names) - 1) / 2)
    ax.set_xticklabels(score_names, rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "auroc_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run OOD evaluation against a trained model.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ood-config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override n_predictive_samples from the OOD config.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.resolve(run_cfg)
    ood_cfg = OmegaConf.load(args.ood_config)
    OmegaConf.resolve(ood_cfg)

    method_name = str(run_cfg.method.get("name", "deterministic")).lower()
    scenario = str(ood_cfg.scenario)
    _assert_training_matches(run_cfg, ood_cfg)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(
        (run_dir / "checkpoint_path.txt").read_text().strip()
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ID loader uses the trained run's data config exactly.
    id_loader = MedMNISTLoader(run_cfg.data)
    model = _load_model(
        run_cfg, ckpt_path, device,
        id_loader.metadata.num_classes, id_loader.metadata.in_channels,
    )

    la = None
    n_samples = int(args.n_samples or ood_cfg.get("n_predictive_samples", 100))
    if method_name == "last_layer_laplace":
        from laplace import Laplace
        la_path = ckpt_path.with_suffix(".laplace.pt")
        payload = torch.load(la_path, map_location=device, weights_only=False)
        la = Laplace(
            model, likelihood="classification",
            subset_of_weights=payload["subset_of_weights"],
            hessian_structure=payload["hessian_structure"],
        )
        la.load_state_dict(payload["state_dict"])

    pair = ood_pair_from_cfg(ood_cfg)
    batch_size = int(ood_cfg.get("batch_size", 256))
    num_workers = int(ood_cfg.get("num_workers", 4))
    data_root = str(run_cfg.data.get("root", "./data"))
    id_loader_t, ood_loaders = build_ood_loaders(
        pair, id_loader=id_loader,
        batch_size=batch_size, num_workers=num_workers, data_root=data_root,
    )

    scenario_dir = run_dir / "ood" / scenario
    fig_dir = scenario_dir / "figures"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[evaluate_ood] scenario={scenario} method={method_name} run={run_dir.name}", flush=True)

    if scenario == "long_tail" and id_loader.metadata.num_classes == 2:
        print(
            "\nNOTE: long-tail OOD on a binary dataset compares uncertainty on the\n"
            "majority class vs. the under-represented class. Results may also\n"
            "reflect the inherent difficulty of the two classes, not only the\n"
            "training data imbalance. Interpret with care; cross-check against\n"
            "multi-class long-tail experiments (e.g. on BloodMNIST).\n",
            flush=True,
        )

    print("[evaluate_ood] predicting on ID test set...", flush=True)
    preds_id = _predict(method_name, model, id_loader_t, device, n_samples, la=la)
    np.savez(scenario_dir / "id_predictions.npz",
             probs_samples=preds_id.numpy().astype(np.float32))

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for ood_name, ood_loader in ood_loaders.items():
        print(f"[evaluate_ood] predicting on OOD '{ood_name}'...", flush=True)
        preds_ood = _predict(method_name, model, ood_loader, device, n_samples, la=la)
        np.savez(scenario_dir / f"{ood_name}_predictions.npz",
                 probs_samples=preds_ood.numpy().astype(np.float32))

        metrics = evaluate_ood(preds_id, preds_ood)
        all_metrics[ood_name] = metrics

        scores_by_name = {
            name: (fn(preds_id).cpu().numpy(), fn(preds_ood).cpu().numpy())
            for name, fn in SCORE_FNS.items()
        }
        _plot_histograms(scores_by_name, ood_name, fig_dir)
        _plot_roc(scores_by_name, ood_name, fig_dir)

    _plot_auroc_bars(all_metrics, fig_dir)
    (scenario_dir / "ood_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    # Pretty summary
    score_names = list(SCORE_FNS.keys())
    name_w = max(len(n) for n in score_names)
    print(f"\nOOD metrics — scenario={scenario}, method={method_name}")
    print("-" * (name_w + 60))
    for ood_name, scores in all_metrics.items():
        print(f"\n  OOD set: {ood_name}")
        print(f"  {'score':<{name_w}}  {'AUROC':>7}  {'AUPRC':>7}  {'FPR@95':>7}")
        for s in score_names:
            m = scores[s]
            print(f"  {s:<{name_w}}  {m['auroc']:>7.4f}  {m['auprc']:>7.4f}  {m['fpr_at_95_tpr']:>7.4f}")
    print(f"\nsaved: {scenario_dir / 'ood_metrics.json'}")
    print(f"figures: {fig_dir}")


if __name__ == "__main__":
    main()
