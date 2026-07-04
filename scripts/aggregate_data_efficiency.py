"""Aggregate data-efficiency sweep results into a summary CSV and plots.

Scans outputs/ for ALL runs matching each pneumonia_{method}_n{size} pattern,
reads ood_metrics.json and sigma_summary.json, computes mean ± std across seeds,
and generates plots with error bars.

Usage:
    python scripts/aggregate_data_efficiency.py
    python scripts/aggregate_data_efficiency.py --outputs-dir outputs --out-dir results

Output files:
    results/data_efficiency_raw.csv       — one row per (method, n, seed)
    results/data_efficiency_summary.csv   — one row per (method, n) with mean±std
    results/plots/auroc_vs_train_size.png
    results/plots/sigma_vs_train_size.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

RUN_PATTERNS = [
    ("lll",      "pneumonia_lll_n100",       100),
    ("lll",      "pneumonia_lll_n1000",      1000),
    ("lll",      "pneumonia_lll_n10000",     10000),
    ("map",      "pneumonia_map_n100",       100),
    ("map",      "pneumonia_map_n1000",      1000),
    ("map",      "pneumonia_map_n10000",     10000),
    ("ensemble", "pneumonia_ensemble_n100",  100),
    ("ensemble", "pneumonia_ensemble_n1000", 1000),
    ("ensemble", "pneumonia_ensemble_n10000",10000),
]

FAR_OOD_DATASET  = "bloodmnist"
NEAR_OOD_DATASET = "organamnist"

SCORES_LLL      = ["mutual_information", "logit_variance_sum", "expected_pairwise_kl",
                   "softmax_variance_sum"]
SCORES_MAP      = ["predictive_entropy", "one_minus_max_softmax"]
SCORES_ENSEMBLE = ["mutual_information", "softmax_variance_sum"]


def _all_runs(outputs_dir: Path, run_name: str) -> list[Path]:
    """Return all completed run dirs for a given run_name, sorted oldest-first."""
    base = outputs_dir / run_name
    if not base.exists():
        return []
    return sorted(
        [c for c in base.iterdir() if c.is_dir() and (c / "config.yaml").exists()]
    )


def _read_seed(run_dir: Path) -> int | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
        return int(cfg.seed)
    except Exception:
        return None


def _read_ood_auroc(run_dir: Path, scenario: str, ood_dataset: str, score: str) -> float | None:
    p = run_dir / "ood" / scenario / "ood_metrics.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get(ood_dataset, {}).get(score, {}).get("auroc", None)


def _read_sigma(run_dir: Path) -> dict | None:
    p = run_dir / "sigma_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_raw_table(outputs_dir: Path) -> pd.DataFrame:
    """One row per (method, train_size, seed run)."""
    rows = []
    for method, run_name, train_size in RUN_PATTERNS:
        run_dirs = _all_runs(outputs_dir, run_name)
        if not run_dirs:
            print(f"  [skip] {run_name} — no completed run found")
            continue
        print(f"  [found] {run_name} — {len(run_dirs)} run(s)")

        scores = {"lll": SCORES_LLL, "map": SCORES_MAP, "ensemble": SCORES_ENSEMBLE}[method]

        for run_dir in run_dirs:
            seed = _read_seed(run_dir)
            row: dict = {
                "method": method,
                "run_name": run_name,
                "train_size": train_size,
                "seed": seed,
                "run_dir": str(run_dir),
            }

            for score in scores:
                row[f"far_ood_{score}"]  = _read_ood_auroc(run_dir, "far_ood",  FAR_OOD_DATASET,  score)
                row[f"near_ood_{score}"] = _read_ood_auroc(run_dir, "near_ood", NEAR_OOD_DATASET, score)

            sigma = _read_sigma(run_dir)
            if sigma:
                row["mean_sigma"] = sigma.get("mean_sigma")
                row["max_sigma"]  = sigma.get("max_sigma")
                row["sigma_norm"] = sigma.get("sigma_norm")

            rows.append(row)

    return pd.DataFrame(rows).sort_values(["method", "train_size", "seed"]).reset_index(drop=True)


def build_summary_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw rows to mean ± std per (method, train_size)."""
    numeric_cols = [c for c in raw.columns if c not in ("method", "run_name", "train_size", "seed", "run_dir")]
    agg: list[dict] = []
    for (method, train_size), group in raw.groupby(["method", "train_size"]):
        row: dict = {"method": method, "train_size": train_size, "n_seeds": len(group)}
        for col in numeric_cols:
            vals = group[col].dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"]  = float("nan")
            else:
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"]  = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        agg.append(row)
    return pd.DataFrame(agg).sort_values(["method", "train_size"]).reset_index(drop=True)


def _plot_series(ax, sizes, means, stds, label, color, marker):
    means = np.array(means, dtype=float)
    stds  = np.array(stds,  dtype=float)
    mask  = ~np.isnan(means)
    if not mask.any():
        return
    xs = np.array(sizes)[mask]
    ys = means[mask]
    es = stds[mask]
    ax.plot(xs, ys, marker=marker, label=label, color=color)
    if es.any():
        ax.fill_between(xs, ys - es, ys + es, alpha=0.15, color=color)


def plot_auroc(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("OOD Detection AUROC vs Training Size", fontsize=13)

    scenarios = [("far_ood",  f"Far-OOD ({FAR_OOD_DATASET})",  axes[0]),
                 ("near_ood", f"Near-OOD ({NEAR_OOD_DATASET})", axes[1])]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map: dict[str, str] = {}
    ci = 0

    for scenario_key, title, ax in scenarios:
        ci_local = ci
        for method, group in df.groupby("method"):
            group = group.sort_values("train_size")
            sizes = group["train_size"].tolist()

            if method == "lll":
                plot_scores = [
                    ("mutual_information",   "LLL — Mutual Information",   "o"),
                    ("logit_variance_sum",   "LLL — Logit Variance",       "o"),
                    ("expected_pairwise_kl", "LLL — Exp. Pairwise KL",    "o"),
                ]
            elif method == "ensemble":
                plot_scores = [
                    ("mutual_information",   "Ensemble — Mutual Information",  "s"),
                    ("softmax_variance_sum", "Ensemble — Softmax Variance",    "s"),
                ]
            else:
                plot_scores = [
                    ("predictive_entropy",    "MAP — Predictive Entropy",  "^"),
                    ("one_minus_max_softmax", "MAP — Max Softmax",         "^"),
                ]

            for score_key, label, marker in plot_scores:
                mean_col = f"{scenario_key}_{score_key}_mean"
                std_col  = f"{scenario_key}_{score_key}_std"
                if mean_col not in group.columns:
                    continue
                color = color_map.setdefault(label, colors[ci_local % len(colors)])
                if label not in color_map:
                    ci_local += 1
                _plot_series(
                    ax, sizes,
                    group[mean_col].tolist(),
                    group[std_col].tolist() if std_col in group.columns else [0.0] * len(sizes),
                    label, color, marker,
                )

        ax.set_xscale("log")
        ax.set_xticks([100, 1000, 10000])
        ax.set_xticklabels(["100", "1000", "10000"])
        ax.set_xlabel("Training size")
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "auroc_vs_train_size.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def plot_sigma(df: pd.DataFrame, out_dir: Path) -> None:
    lll = df[df["method"] == "lll"].sort_values("train_size")
    if lll.empty or "mean_sigma_mean" not in lll.columns:
        print("  [skip] sigma plot — no LLL sigma data found")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Posterior Sigma vs Training Size (Last-Layer Laplace)", fontsize=12)
    sizes = lll["train_size"].tolist()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col, label, marker, color in [
        ("mean_sigma", "mean diagonal(Σ)", "o", colors[0]),
        ("max_sigma",  "max diagonal(Σ)",  "s", colors[1]),
    ]:
        mean_col = f"{col}_mean"
        std_col  = f"{col}_std"
        if mean_col in lll.columns and lll[mean_col].notna().any():
            _plot_series(ax, sizes, lll[mean_col].tolist(),
                         lll[std_col].tolist() if std_col in lll.columns else [0.0]*len(sizes),
                         label, color, marker)

    ax2 = ax.twinx()
    if "sigma_norm_mean" in lll.columns and lll["sigma_norm_mean"].notna().any():
        _plot_series(ax2, sizes, lll["sigma_norm_mean"].tolist(),
                     lll["sigma_norm_std"].tolist() if "sigma_norm_std" in lll.columns else [0.0]*len(sizes),
                     "‖Σ‖_F (right axis)", "tab:green", "^")
        ax2.set_ylabel("Frobenius norm ‖Σ‖_F", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

    ax.set_xscale("log")
    ax.set_xticks([100, 1000, 10000])
    ax.set_xticklabels(["100", "1000", "10000"])
    ax.set_xlabel("Training size")
    ax.set_ylabel("Sigma (posterior variance diagonal)")
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.tight_layout()
    path = out_dir / "sigma_vs_train_size.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    outputs_dir = (PACKAGE_ROOT / args.outputs_dir).resolve()
    out_dir = (PACKAGE_ROOT / args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning for data-efficiency runs...")
    raw = build_raw_table(outputs_dir)

    if raw.empty:
        print("No completed runs found. Run the sweep first.")
        return

    raw_csv = out_dir / "data_efficiency_raw.csv"
    raw.to_csv(raw_csv, index=False)
    print(f"\nsaved raw: {raw_csv}")

    summary = build_summary_table(raw)
    summary_csv = out_dir / "data_efficiency_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"saved summary: {summary_csv}")

    # Print a readable table: mean (±std) for key AUROC columns
    print("\n=== Summary (mean ± std across seeds) ===")
    display_cols = ["method", "train_size", "n_seeds"]
    key_scores = [
        ("lll",      "far_ood_mutual_information"),
        ("lll",      "near_ood_mutual_information"),
        ("map",      "far_ood_predictive_entropy"),
        ("map",      "near_ood_predictive_entropy"),
        ("ensemble", "far_ood_mutual_information"),
        ("ensemble", "near_ood_mutual_information"),
    ]
    for method, score in key_scores:
        mean_col = f"{score}_mean"
        std_col  = f"{score}_std"
        if mean_col in summary.columns:
            display_cols += [mean_col, std_col]

    avail = [c for c in display_cols if c in summary.columns]
    print(summary[avail].to_string(index=False))

    print("\nGenerating plots...")
    plot_auroc(summary, plots_dir)
    plot_sigma(summary, plots_dir)


if __name__ == "__main__":
    main()
