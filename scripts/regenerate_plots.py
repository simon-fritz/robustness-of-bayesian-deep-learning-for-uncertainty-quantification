"""Regenerate every plot for a trained run from cached predictions.

Reads ``test_predictions.npz`` (from ``scripts/evaluate.py``) and any
``ood/<scenario>/{id,<ood_name>}_predictions.npz`` files (from
``scripts/evaluate_ood.py``), then re-runs every plotting helper in
:mod:`bnn_medmnist.evaluation.plots`. No inference, no metric recomputation
beyond what the plots themselves derive from the cached arrays.

Usage:
    python scripts/regenerate_plots.py --run-dir outputs/<name>/<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from medmnist import INFO
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve

from bnn_medmnist.evaluation.ood import SCORE_FNS, evaluate_ood
from bnn_medmnist.evaluation.plots import (
    plot_auroc_bar_chart,
    plot_confidence_histogram_on_ood,
    plot_failure_modes,
    plot_reliability_diagram,
    plot_roc_curves,
    plot_uncertainty_histogram,
    plot_uncertainty_scatter,
)
from bnn_medmnist.evaluation.uncertainty import (
    expected_entropy,
    mutual_information,
    predictive_entropy,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _regen_test_plots(run_dir: Path, method_name: str) -> list[Path]:
    pred_path = run_dir / "test_predictions.npz"
    if not pred_path.exists():
        return []
    data = _load_npz(pred_path)
    probs = data["probs_samples"]
    mean_probs = probs.mean(axis=0)
    labels = data["labels"]
    fig_dir = run_dir / "figures"
    rel_path = fig_dir / "reliability_diagram"
    plot_reliability_diagram(labels, mean_probs, method_name=method_name, save_path=rel_path)
    return [rel_path.with_suffix(".png")]


def _class_names_for(dataset_flag: str) -> list[str]:
    info = INFO.get(dataset_flag.lower())
    if not info:
        return []
    return [v for _, v in sorted(info["label"].items(), key=lambda kv: int(kv[0]))]


def _regen_ood_plots(run_dir: Path, scenario_dir: Path, run_cfg) -> list[Path]:
    scenario = scenario_dir.name
    fig_dir = scenario_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    id_path = scenario_dir / "id_predictions.npz"
    if not id_path.exists():
        print(f"[regen] skipping {scenario}: no id_predictions.npz", flush=True)
        return []
    id_data = _load_npz(id_path)
    preds_id = torch.from_numpy(id_data["probs_samples"])
    images_id = id_data["images"]
    labels_id = id_data["labels"]

    class_names = _class_names_for(str(run_cfg.data.flag))

    ood_files = sorted(
        p for p in scenario_dir.glob("*_predictions.npz") if p.name != "id_predictions.npz"
    )
    written: list[Path] = []
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for ood_file in ood_files:
        ood_name = ood_file.stem.removesuffix("_predictions")
        ood_data = _load_npz(ood_file)
        preds_ood = torch.from_numpy(ood_data["probs_samples"])
        images_ood = ood_data["images"]
        metrics = evaluate_ood(preds_id, preds_ood)
        all_metrics[ood_name] = metrics

        score_results = {}
        for name, fn in SCORE_FNS.items():
            id_s = fn(preds_id).cpu().numpy()
            ood_s = fn(preds_ood).cpu().numpy()
            auroc = float(metrics[name]["auroc"])
            p = fig_dir / f"hist_{ood_name}_{name}"
            plot_uncertainty_histogram(id_s, ood_s, name, ood_name, auroc, p)
            written.append(p.with_suffix(".png"))
            y = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])
            s = np.concatenate([id_s, ood_s])
            fpr, tpr, _ = roc_curve(y, s)
            score_results[name] = (fpr, tpr, auroc)

        roc_path = fig_dir / f"roc_{ood_name}"
        plot_roc_curves(score_results, ood_name, roc_path)
        written.append(roc_path.with_suffix(".png"))

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

        msp_ood = preds_ood.mean(dim=0).max(dim=-1).values.cpu().numpy()
        conf_path = fig_dir / f"ood_confidence_{ood_name}"
        plot_confidence_histogram_on_ood(msp_ood, conf_path, ood_name)
        written.append(conf_path.with_suffix(".png"))

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
            id_images=images_id, id_uncertainty=u_id,
            id_preds=pred_id, id_labels=labels_id,
            ood_images=images_ood, ood_uncertainty=u_ood,
            ood_preds=pred_ood, save_path=fm_path,
            scenario_name=f"{scenario} / {ood_name}",
            class_names=class_names,
        )
        written.append(fm_path.with_suffix(".png"))

    if all_metrics:
        method_name = str(run_cfg.method.get("name", "deterministic")).lower()
        rows = [
            {"method": method_name, "scenario": f"{scenario}:{ood_name}",
             "score": s, "auroc": v["auroc"]}
            for ood_name, scores in all_metrics.items()
            for s, v in scores.items()
        ]
        summary = fig_dir / "auroc_summary"
        plot_auroc_bar_chart(rows, summary)
        written.append(summary.with_suffix(".png"))
        (scenario_dir / "ood_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all plots for a run from cached predictions.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.resolve(run_cfg)
    method_name = str(run_cfg.method.get("name", "deterministic")).lower()

    written: list[Path] = []
    written += _regen_test_plots(run_dir, method_name)
    ood_root = run_dir / "ood"
    if ood_root.exists():
        for scenario_dir in sorted(p for p in ood_root.iterdir() if p.is_dir()):
            written += _regen_ood_plots(run_dir, scenario_dir, run_cfg)

    if not written:
        print("[regen] no cached predictions found — nothing to plot.")
        return
    print(f"[regen] regenerated {len(written)} figure(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
