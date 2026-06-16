"""Aggregate data-efficiency sweep results into a summary CSV and plots.

Scans outputs/ for runs matching pneumonia_lll_n* and pneumonia_map_n*,
reads ood_metrics.json and sigma_summary.json, builds a summary table, and
generates two plots:

  Plot 1 — AUROC vs Training Size (far-OOD and near-OOD subplots)
  Plot 2 — Posterior sigma vs Training Size (LLL only)

Usage:
    python scripts/aggregate_data_efficiency.py
    python scripts/aggregate_data_efficiency.py --outputs-dir outputs --out-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

RUN_PATTERNS = [
    ("lll",      "pneumonia_lll_n100",      100),
    ("lll",      "pneumonia_lll_n1000",     1000),
    ("lll",      "pneumonia_lll_n10000",    10000),
    ("map",      "pneumonia_map_n100",      100),
    ("map",      "pneumonia_map_n1000",     1000),
    ("map",      "pneumonia_map_n10000",    10000),
    ("ensemble", "pneumonia_ensemble_n100",  100),
    ("ensemble", "pneumonia_ensemble_n1000", 1000),
    ("ensemble", "pneumonia_ensemble_n10000",10000),
]

FAR_OOD_DATASET  = "bloodmnist"
NEAR_OOD_DATASET = "organamnist"

SCORES_LLL      = ["mutual_information", "logit_variance_sum", "expected_pairwise_kl",
                   "softmax_variance_sum"]
SCORES_MAP      = ["predictive_entropy", "one_minus_max_softmax"]
SCORES_ENSEMBLE = ["mutual_information", "expected_pairwise_kl", "softmax_variance_sum"]


def _latest_run(outputs_dir: Path, run_name: str) -> Path | None:
    base = outputs_dir / run_name
    if not base.exists():
        return None
    candidates = sorted(base.iterdir(), reverse=True)
    for c in candidates:
        if c.is_dir() and (c / "config.yaml").exists():
            return c
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


def build_table(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for method, run_name, train_size in RUN_PATTERNS:
        run_dir = _latest_run(outputs_dir, run_name)
        if run_dir is None:
            print(f"  [skip] {run_name} — no completed run found in {outputs_dir}")
            continue
        print(f"  [found] {run_name} → {run_dir.name}")

        row: dict = {"method": method, "run_name": run_name, "train_size": train_size}

        scores = {"lll": SCORES_LLL, "map": SCORES_MAP, "ensemble": SCORES_ENSEMBLE}[method]
        for score in scores:
            row[f"far_ood_{score}"]  = _read_ood_auroc(run_dir, "far_ood",  FAR_OOD_DATASET,  score)
            row[f"near_ood_{score}"] = _read_ood_auroc(run_dir, "near_ood", NEAR_OOD_DATASET, score)

        sigma = _read_sigma(run_dir)
        if sigma:
            row["mean_sigma"] = sigma.get("mean_sigma")
            row["max_sigma"]  = sigma.get("max_sigma")
            row["sigma_norm"] = sigma.get("sigma_norm")

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["method", "train_size"]).reset_index(drop=True)


def plot_auroc(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("OOD Detection AUROC vs Training Size", fontsize=13)

    scenarios = [("far_ood",  f"Far-OOD ({FAR_OOD_DATASET})",  axes[0]),
                 ("near_ood", f"Near-OOD ({NEAR_OOD_DATASET})", axes[1])]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map: dict[str, str] = {}

    for scenario_key, title, ax in scenarios:
        ci = 0
        for method, group in df.groupby("method"):
            group = group.sort_values("train_size")
            sizes = group["train_size"].tolist()

            if method == "lll":
                plot_scores = [
                    ("mutual_information",   "LLL — Mutual Information"),
                    ("logit_variance_sum",   "LLL — Logit Variance"),
                    ("expected_pairwise_kl", "LLL — Exp. Pairwise KL"),
                ]
            elif method == "ensemble":
                plot_scores = [
                    ("mutual_information",   "Ensemble — Mutual Information"),
                    ("expected_pairwise_kl", "Ensemble — Exp. Pairwise KL"),
                ]
            else:
                plot_scores = [
                    ("predictive_entropy",    "MAP — Predictive Entropy"),
                    ("one_minus_max_softmax", "MAP — Max Softmax"),
                ]

            for score_key, label in plot_scores:
                col = f"{scenario_key}_{score_key}"
                if col not in group.columns:
                    continue
                vals = group[col].tolist()
                if all(v is None for v in vals):
                    continue
                color = color_map.setdefault(label, colors[ci % len(colors)])
                ci += 1 if label not in color_map else 0
                marker = "o" if method == "lll" else "s"
                ax.plot(sizes, vals, marker=marker, label=label, color=color)

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
    if lll.empty or "mean_sigma" not in lll.columns:
        print("  [skip] sigma plot — no LLL sigma data found")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Posterior Sigma vs Training Size (Last-Layer Laplace)", fontsize=12)
    sizes = lll["train_size"].tolist()

    for col, label, marker in [
        ("mean_sigma", "mean diagonal(Σ)", "o"),
        ("max_sigma",  "max diagonal(Σ)",  "s"),
    ]:
        if col in lll.columns and lll[col].notna().any():
            ax.plot(sizes, lll[col].tolist(), marker=marker, label=label)

    ax2 = ax.twinx()
    if "sigma_norm" in lll.columns and lll["sigma_norm"].notna().any():
        ax2.plot(sizes, lll["sigma_norm"].tolist(), marker="^", color="tab:green",
                 linestyle="--", label="‖Σ‖_F (right axis)")
        ax2.set_ylabel("Frobenius norm ‖Σ‖_F", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

    ax.set_xscale("log")
    ax.set_xticks([100, 1000, 10000])
    ax.set_xticklabels(["100", "1000", "10000"])
    ax.set_xlabel("Training size")
    ax.set_ylabel("Sigma (posterior variance diagonal)")
    ax.legend(loc="upper right", fontsize=9)
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
    df = build_table(outputs_dir)

    if df.empty:
        print("No completed runs found. Run the sweep first.")
        return

    csv_path = out_dir / "data_efficiency_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nsaved: {csv_path}")
    print(df.to_string(index=False))

    print("\nGenerating plots...")
    plot_auroc(df, plots_dir)
    plot_sigma(df, plots_dir)


if __name__ == "__main__":
    main()
