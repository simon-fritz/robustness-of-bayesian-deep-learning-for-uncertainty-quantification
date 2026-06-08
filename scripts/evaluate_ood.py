"""OOD evaluation entry point.

Pure post-hoc against a trained run for far_ood / near_ood scenarios. For
held_out_class and long_tail scenarios, the run's training config must already
match the OOD eval config (the script asserts this).

For Laplace runs both prediction modes are computed and cached: MC softmax
samples (entropy / spread scores) and the analytical Gaussian over the logits
(``pred_type="glm"``, the sampling-free logit-variance scores). The score
table groups every measure by conceptual family (see
``evaluation.ood.SCORE_CATEGORIES``).

Usage:
    python scripts/evaluate_ood.py \\
        --run-dir outputs/pneumonia_baseline/<timestamp> \\
        --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve

from bnn_medmnist.data.medmnist_loader import MedMNISTLoader
from bnn_medmnist.data.ood_pairs import build_ood_loaders, ood_pair_from_cfg
from bnn_medmnist.evaluation.ood import (
    SCORE_CATEGORIES,
    ood_metrics_from_scores,
    per_sample_scores,
)
from bnn_medmnist.evaluation.plots import (
    plot_auroc_bar_chart,
    plot_confidence_histogram_on_ood,
    plot_failure_modes,
    plot_roc_curves,
    plot_score_category_bar_chart,
    plot_score_correlation_scatter,
    plot_uncertainty_histogram,
    plot_uncertainty_scatter,
)
from bnn_medmnist.evaluation.uncertainty import (
    expected_entropy,
    mutual_information,
    predictive_entropy,
)
from bnn_medmnist.models import build_model


# ---------------------------------------------------------------------------
# model + prediction helpers (mirror scripts/evaluate.py)
# ---------------------------------------------------------------------------
def _load_model(cfg, ckpt_path, device, num_classes, in_channels):
    model = build_model(cfg.model, in_channels=in_channels, num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _assert_id_ood_shapes(id_loader_t, ood_loaders) -> None:
    """Fail loudly unless ID and every OOD loader yield same-shape, same-dtype tensors.

    With ResNet-18 both must be 3-channel 224x224; the transforms are applied on
    the loader side, so a misconfigured OOD transform would otherwise surface as
    a confusing runtime error mid-evaluation.
    """
    x_id, _ = next(iter(id_loader_t))
    id_shape, id_dtype = tuple(x_id.shape[1:]), x_id.dtype
    for name, loader in ood_loaders.items():
        x_ood, _ = next(iter(loader))
        if tuple(x_ood.shape[1:]) != id_shape or x_ood.dtype != id_dtype:
            raise SystemExit(
                f"[evaluate_ood] ID/OOD tensor mismatch for '{name}': "
                f"ID={id_shape}/{id_dtype} vs OOD={tuple(x_ood.shape[1:])}/{x_ood.dtype}. "
                f"OOD loaders must apply the same image_transform as the ID data."
            )
    print(f"[evaluate_ood] ID/OOD shape OK: {id_shape}, dtype={id_dtype}", flush=True)


@torch.no_grad()
def _collect(model, loader, device, *, method_name: str, predictor=None, n_samples: int = 1):
    """Run inference and collect predictions, images, labels, and logit moments.

    Returns ``(probs[S, N, C], images[N, ...], labels[N], logit_mean, logit_var)``
    where the logit moments are ``(N, C)`` tensors for Laplace runs and ``None``
    for deterministic runs (no posterior over weights).
    """
    all_p, all_x, all_y, all_lm, all_lv = [], [], [], [], []
    for x, y in loader:
        all_x.append(x.cpu())
        all_y.append(y.cpu() if isinstance(y, torch.Tensor) else torch.as_tensor(y))
        if method_name == "deterministic":
            p = torch.softmax(model(x.to(device)), dim=-1).cpu().unsqueeze(0)
        elif method_name == "last_layer_laplace":
            res = predictor.predict_modes(x.to(device), n_samples=n_samples, modes=("mc", "glm"))
            p = res["softmax_samples"].cpu()
            all_lm.append(res["logit_mean"].cpu())
            all_lv.append(res["logit_var"].cpu())
        elif method_name == "deep_ensemble":
            batch_probs = [torch.softmax(m(x.to(device)), dim=-1).cpu() for m in model]
            p = torch.stack(batch_probs, dim=0)
        else:
            raise NotImplementedError(f"method '{method_name}' not supported")
        all_p.append(p)
    probs = torch.cat(all_p, dim=1)
    images = torch.cat(all_x, dim=0)
    labels = torch.cat(all_y, dim=0)
    logit_mean = torch.cat(all_lm, dim=0) if all_lm else None
    logit_var = torch.cat(all_lv, dim=0) if all_lv else None
    return probs, images, labels, logit_mean, logit_var


# ---------------------------------------------------------------------------
# sanity check for within-dataset scenarios
# ---------------------------------------------------------------------------
def _assert_training_matches(run_cfg, ood_cfg) -> None:
    scenario = str(ood_cfg.scenario)
    data = run_cfg.data
    train_excl = list(data.get("exclude_classes") or [])
    train_sub = dict(data.get("class_subsampling") or {})
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
        if train_excl or train_sub:
            print(
                f"[evaluate_ood] WARNING: trained run has data filters "
                f"(exclude={train_excl}, subsample={train_sub}) — its ID test "
                f"loader will be filtered to those classes only.",
                flush=True,
            )


# ---------------------------------------------------------------------------
# plotting helpers — thin wrappers over evaluation.plots
# ---------------------------------------------------------------------------
def _render_scenario_plots(
    *,
    fig_dir: Path,
    preds_id: torch.Tensor,
    preds_ood: torch.Tensor,
    id_scores: dict[str, torch.Tensor | None],
    ood_scores: dict[str, torch.Tensor | None],
    images_id: torch.Tensor,
    labels_id: torch.Tensor,
    images_ood: torch.Tensor,
    ood_name: str,
    scenario: str,
    metrics: dict[str, dict[str, float]],
    class_names: list[str] | None,
) -> list[Path]:
    """Render and save every per-scenario figure. Returns saved-figure paths."""
    written: list[Path] = []

    # per-score histograms + ROC overlay — over every *computable* score.
    score_results: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name in metrics:  # metrics already excludes N/A scores
        id_s = id_scores[name].cpu().numpy()
        ood_s = ood_scores[name].cpu().numpy()
        auroc = float(metrics[name]["auroc"])
        path = fig_dir / f"hist_{ood_name}_{name}"
        plot_uncertainty_histogram(id_s, ood_s, name, ood_name, auroc, path)
        written.append(path.with_suffix(".png"))

        y = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])
        s = np.concatenate([id_s, ood_s])
        fpr, tpr, _ = roc_curve(y, s)
        score_results[name] = (fpr, tpr, auroc)

    roc_path = fig_dir / f"roc_{ood_name}"
    plot_roc_curves(score_results, ood_name, roc_path)
    written.append(roc_path.with_suffix(".png"))

    # AUROC bars grouped by conceptual family.
    cat_path = fig_dir / f"auroc_by_category_{ood_name}"
    plot_score_category_bar_chart(
        {name: m["auroc"] for name, m in metrics.items()},
        SCORE_CATEGORIES, cat_path,
        title=f"OOD-detection AUROC by score family — ID vs {ood_name}",
    )
    written.append(cat_path.with_suffix(".png"))

    # epistemic vs aleatoric scatter
    epis_id = mutual_information(preds_id).cpu().numpy()
    alea_id = expected_entropy(preds_id).cpu().numpy()
    epis_ood = mutual_information(preds_ood).cpu().numpy()
    alea_ood = expected_entropy(preds_ood).cpu().numpy()
    scatter_path = fig_dir / f"scatter_{ood_name}"
    plot_uncertainty_scatter(
        epistemic=np.concatenate([epis_id, epis_ood]),
        aleatoric=np.concatenate([alea_id, alea_ood]),
        is_ood=np.concatenate([np.zeros_like(epis_id, dtype=bool),
                               np.ones_like(epis_ood, dtype=bool)]),
        save_path=scatter_path, ood_name=ood_name,
    )
    written.append(scatter_path.with_suffix(".png"))

    # MI vs logit-variance scatter (Hüllermeier: same info or not?) — Laplace only.
    if id_scores.get("logit_variance_sum") is not None:
        mi_all = np.concatenate([epis_id, epis_ood])
        lv_all = np.concatenate([
            id_scores["logit_variance_sum"].cpu().numpy(),
            ood_scores["logit_variance_sum"].cpu().numpy(),
        ])
        is_ood = np.concatenate([np.zeros_like(epis_id, dtype=bool),
                                 np.ones_like(epis_ood, dtype=bool)])
        mivlv_path = fig_dir / f"mi_vs_logitvar_{ood_name}"
        _, r = plot_score_correlation_scatter(
            mi_all, lv_all, is_ood, mivlv_path,
            x_label="Mutual Information", y_label="Logit Variance (sum)",
            title=f"MI vs Logit Variance — ID vs {ood_name}",
        )
        written.append(mivlv_path.with_suffix(".png"))
        print(f"  [scatter] MI vs logit-variance Pearson r = {r:.3f}", flush=True)

    # OOD confidence histogram (1 - MSP, but plotted as max softmax)
    mean_ood = preds_ood.mean(dim=0)
    msp_ood = mean_ood.max(dim=-1).values.cpu().numpy()
    conf_path = fig_dir / f"ood_confidence_{ood_name}"
    plot_confidence_histogram_on_ood(msp_ood, conf_path, ood_name)
    written.append(conf_path.with_suffix(".png"))

    # failure-modes grid: use mutual information when available (S>1),
    # otherwise predictive entropy (always defined).
    if preds_id.shape[0] > 1:
        u_id = mutual_information(preds_id).cpu().numpy()
        u_ood = mutual_information(preds_ood).cpu().numpy()
    else:
        u_id = predictive_entropy(preds_id).cpu().numpy()
        u_ood = predictive_entropy(preds_ood).cpu().numpy()
    pred_id = preds_id.mean(dim=0).argmax(dim=-1).cpu().numpy()
    pred_ood = preds_ood.mean(dim=0).argmax(dim=-1).cpu().numpy()
    fm_path = fig_dir / f"failure_modes_{ood_name}"
    plot_failure_modes(
        id_images=images_id.numpy(), id_uncertainty=u_id,
        id_preds=pred_id, id_labels=labels_id.numpy(),
        ood_images=images_ood.numpy(), ood_uncertainty=u_ood,
        ood_preds=pred_ood, save_path=fm_path,
        scenario_name=f"{scenario} / {ood_name}",
        class_names=class_names,
    )
    written.append(fm_path.with_suffix(".png"))
    return written


def _render_summary(metrics_by_ood, method_name: str, scenario: str, fig_dir: Path) -> Path:
    """One AUROC bar chart summarising all OOD sets for this scenario."""
    rows = []
    for ood_name, scores in metrics_by_ood.items():
        for score, vals in scores.items():
            rows.append({
                "method": method_name, "scenario": f"{scenario}:{ood_name}",
                "score": score, "auroc": vals["auroc"],
            })
    path = fig_dir / "auroc_summary"
    plot_auroc_bar_chart(rows, path)
    return path.with_suffix(".png")


# ---------------------------------------------------------------------------
# grouped console summary
# ---------------------------------------------------------------------------
def _print_grouped_summary(all_metrics, scenario: str, method_name: str) -> None:
    """Print the AUROC table grouped by conceptual score family."""
    name_w = max(len(n) for n in (s for names in SCORE_CATEGORIES.values() for s in names))
    print(f"\nOOD metrics — scenario={scenario}, method={method_name}")
    for ood_name, scores in all_metrics.items():
        print(f"\n  OOD set: {ood_name}")
        for category, names in SCORE_CATEGORIES.items():
            print(f"\n    {category}")
            print(f"    {'score':<{name_w}}  {'AUROC':>7}  {'AUPRC':>7}  {'FPR@95':>7}")
            for s in names:
                if s in scores:
                    m = scores[s]
                    print(f"    {s:<{name_w}}  {m['auroc']:>7.4f}  {m['auprc']:>7.4f}  {m['fpr_at_95_tpr']:>7.4f}")
                else:
                    print(f"    {s:<{name_w}}  {'N/A — requires posterior over weights':>7}")


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

    id_loader = MedMNISTLoader(run_cfg.data)
    
    if method_name == "deep_ensemble":
        ckpt_dir = ckpt_path.parent
        member_files = sorted(ckpt_dir.glob("member_*.pt"))
        model = []
        for f in member_files:
            model.append(_load_model(run_cfg, f, device, id_loader.metadata.num_classes, id_loader.metadata.in_channels))
    else:
        model = _load_model(
            run_cfg, ckpt_path, device,
            id_loader.metadata.num_classes, id_loader.metadata.in_channels,
        )

    predictor = None
    n_samples = int(args.n_samples or ood_cfg.get("n_predictive_samples", 100))
    if method_name == "last_layer_laplace":
        from laplace import Laplace

        from bnn_medmnist.methods.last_layer_laplace import LastLayerLaplace
        la_path = ckpt_path.with_suffix(".laplace.pt")
        payload = torch.load(la_path, map_location=device, weights_only=False)
        la = Laplace(
            model, likelihood="classification",
            subset_of_weights=payload["subset_of_weights"],
            hessian_structure=payload["hessian_structure"],
        )
        la.load_state_dict(payload["state_dict"])
        # Wrap the fitted Laplace so we can use predict_modes (mc + glm).
        predictor = LastLayerLaplace(device=device)
        predictor.model = model
        predictor.la = la

    pair = ood_pair_from_cfg(ood_cfg)
    batch_size = int(ood_cfg.get("batch_size", 256))
    num_workers = int(ood_cfg.get("num_workers", 4))
    data_root = str(run_cfg.data.get("root", "./data"))
    id_loader_t, ood_loaders = build_ood_loaders(
        pair, id_loader=id_loader,
        batch_size=batch_size, num_workers=num_workers, data_root=data_root,
    )
    _assert_id_ood_shapes(id_loader_t, ood_loaders)

    scenario_dir = run_dir / "ood" / scenario
    fig_dir = scenario_dir / "figures"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Cache the OOD config so regenerate_plots.py can reproduce class names.
    OmegaConf.save(ood_cfg, scenario_dir / "ood_config.yaml")

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

    class_names = [v for _, v in sorted(id_loader.info["label"].items(),
                                        key=lambda kv: int(kv[0]))]

    print("[evaluate_ood] predicting on ID test set...", flush=True)
    preds_id, images_id, labels_id, lm_id, lv_id = _collect(
        model, id_loader_t, device,
        method_name=method_name, predictor=predictor, n_samples=n_samples,
    )
    id_save = {
        "probs_samples": preds_id.numpy().astype(np.float32),
        "images": images_id.numpy().astype(np.float32),
        "labels": labels_id.numpy().astype(np.int64),
    }
    if lm_id is not None:
        id_save["logit_mean"] = lm_id.numpy().astype(np.float32)
        id_save["logit_var"] = lv_id.numpy().astype(np.float32)
    np.savez(scenario_dir / "id_predictions.npz", **id_save)

    id_scores = per_sample_scores(preds_id, lm_id, lv_id)

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    written_figs: list[Path] = []
    for ood_name, ood_loader in ood_loaders.items():
        print(f"[evaluate_ood] predicting on OOD '{ood_name}'...", flush=True)
        preds_ood, images_ood, _, lm_ood, lv_ood = _collect(
            model, ood_loader, device,
            method_name=method_name, predictor=predictor, n_samples=n_samples,
        )
        ood_save = {
            "probs_samples": preds_ood.numpy().astype(np.float32),
            "images": images_ood.numpy().astype(np.float32),
        }
        if lm_ood is not None:
            ood_save["logit_mean"] = lm_ood.numpy().astype(np.float32)
            ood_save["logit_var"] = lv_ood.numpy().astype(np.float32)
        np.savez(scenario_dir / f"{ood_name}_predictions.npz", **ood_save)

        ood_scores = per_sample_scores(preds_ood, lm_ood, lv_ood)
        metrics = ood_metrics_from_scores(id_scores, ood_scores)
        all_metrics[ood_name] = metrics

        written_figs += _render_scenario_plots(
            fig_dir=fig_dir, preds_id=preds_id, preds_ood=preds_ood,
            id_scores=id_scores, ood_scores=ood_scores,
            images_id=images_id, labels_id=labels_id, images_ood=images_ood,
            ood_name=ood_name, scenario=scenario, metrics=metrics,
            class_names=class_names,
        )

    written_figs.append(_render_summary(all_metrics, method_name, scenario, fig_dir))
    (scenario_dir / "ood_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    _print_grouped_summary(all_metrics, scenario, method_name)
    print(f"\nsaved: {scenario_dir / 'ood_metrics.json'}")
    print("figures written:")
    for p in written_figs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
