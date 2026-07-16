"""Aggregate ALL experiment results into unified CSVs and a summary table.

Covers three experiment groups:
  1. full_data   — balanced PneumoniaMNIST, ResNet-18 (LLL / FLL / MAP / Ensemble)
  2. longtail    — class_subsampling 2% normal, ResNet-18 (LLL / FLL / MAP / Ensemble)
  3. data_eff    — ResNet-18, train_size 100/1000/10000 (LLL / MAP / Ensemble)

Note: FLL (First-Layer Laplace) is single-seed (seed=42), exploratory.
      Run with --seeds 42 to include only its seed.

Usage:
    python scripts/aggregate_all.py --seeds 0 1 2 3 4        # 5-seed methods only
    python scripts/aggregate_all.py --seeds 0 1 2 3 4 42     # include FLL (seed=42) and legacy runs
    python scripts/aggregate_all.py --seeds 42               # FLL only

Output:
    results/all_experiments_raw.csv      — one row per run
    results/all_experiments_summary.csv  — mean ± std per (group, method, train_size)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

FAR_OOD_DATASET  = "bloodmnist"
NEAR_OOD_DATASET = "organamnist"

# Primary OOD score per method (used in the summary display table)
PRIMARY_SCORE = {
    "lll":      "mutual_information",
    "fll":      "mutual_information",
    "map":      "predictive_entropy",
    "ensemble": "mutual_information",
}

# All experiments: (group, method, run_name, train_size, ood_scenarios)
# ood_scenarios: list of (scenario_key, ood_dataset_key)
EXPERIMENTS = [
    # --- full-data balanced ---
    ("full_data", "lll",      "pneumonia_resnet18_lll",               None,  ["far_ood", "near_ood"]),
    ("full_data", "fll",      "pneumonia_resnet18_fll",               None,  ["far_ood", "near_ood"]),
    ("full_data", "map",      "pneumonia_resnet18_baseline",           None,  ["far_ood", "near_ood"]),
    ("full_data", "ensemble", "pneumonia_deep_ensemble",               None,  ["far_ood", "near_ood"]),
    # --- long-tail ---
    ("longtail",  "lll",      "pneumonia_resnet18_longtail_normal2pct_lll", None, ["long_tail", "far_ood", "near_ood"]),
    ("longtail",  "fll",      "pneumonia_resnet18_longtail_normal2pct_fll", None, ["long_tail", "far_ood", "near_ood"]),
    ("longtail",  "map",      "pneumonia_resnet18_longtail_normal2pct_det", None, ["long_tail", "far_ood", "near_ood"]),
    ("longtail",  "ensemble", "pneumonia_resnet18_longtail_normal2pct_de",  None, ["long_tail", "far_ood", "near_ood"]),
    # --- data-efficiency sweep ---
    ("data_eff",  "lll",      "pneumonia_lll_n100",     100,   ["far_ood", "near_ood"]),
    ("data_eff",  "lll",      "pneumonia_lll_n1000",    1000,  ["far_ood", "near_ood"]),
    ("data_eff",  "lll",      "pneumonia_lll_n10000",   10000, ["far_ood", "near_ood"]),
    ("data_eff",  "map",      "pneumonia_map_n100",     100,   ["far_ood", "near_ood"]),
    ("data_eff",  "map",      "pneumonia_map_n1000",    1000,  ["far_ood", "near_ood"]),
    ("data_eff",  "map",      "pneumonia_map_n10000",   10000, ["far_ood", "near_ood"]),
    ("data_eff",  "ensemble", "pneumonia_ensemble_n100",     100,   ["far_ood", "near_ood"]),
    ("data_eff",  "ensemble", "pneumonia_ensemble_n1000",    1000,  ["far_ood", "near_ood"]),
    ("data_eff",  "ensemble", "pneumonia_ensemble_n10000",   10000, ["far_ood", "near_ood"]),
]

# OOD dataset key per scenario (long_tail uses PneumoniaMNIST tail class)
SCENARIO_DATASET = {
    "far_ood":   FAR_OOD_DATASET,
    "near_ood":  NEAR_OOD_DATASET,
    "long_tail": "tail_0",
}

# All uncertainty scores each method can produce
SCORES = {
    "lll":      ["mutual_information", "logit_variance_sum", "expected_pairwise_kl", "softmax_variance_sum"],
    "fll":      ["mutual_information", "logit_variance_sum", "expected_pairwise_kl", "softmax_variance_sum"],
    "map":      ["predictive_entropy", "one_minus_max_softmax"],
    "ensemble": ["mutual_information", "softmax_variance_sum"],
}


def _all_runs(outputs_dir: Path, run_name: str) -> list[Path]:
    base = outputs_dir / run_name
    if not base.exists():
        return []
    return sorted(c for c in base.iterdir() if c.is_dir() and (c / "config.yaml").exists())


def _read_seed(run_dir: Path) -> int | None:
    try:
        cfg = OmegaConf.load(run_dir / "config.yaml")
        return int(cfg.seed)
    except Exception:
        return None


def _read_ood_auroc(run_dir: Path, scenario: str, ood_dataset: str, score: str) -> float | None:
    p = run_dir / "ood" / scenario / "ood_metrics.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get(ood_dataset, {}).get(score, {}).get("auroc", None)
    except Exception:
        return None


def _read_id_metrics(run_dir: Path) -> dict:
    p = run_dir / "test_metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def build_raw_table(outputs_dir: Path, seed_filter: list[int] | None) -> pd.DataFrame:
    rows = []
    for group, method, run_name, train_size, ood_scenarios in EXPERIMENTS:
        run_dirs = _all_runs(outputs_dir, run_name)
        if not run_dirs:
            print(f"  [skip] {run_name} — no runs found")
            continue

        kept = []
        for rd in run_dirs:
            s = _read_seed(rd)
            if seed_filter is None or s in seed_filter:
                kept.append((rd, s))

        if not kept:
            print(f"  [skip] {run_name} — no runs matching seed filter")
            continue
        print(f"  [found] {run_name} — {len(kept)} run(s)")

        for run_dir, seed in kept:
            row: dict = {
                "group": group,
                "method": method,
                "run_name": run_name,
                "train_size": train_size,
                "seed": seed,
                "run_dir": str(run_dir),
            }

            id_m = _read_id_metrics(run_dir)
            row["id_accuracy"] = id_m.get("accuracy", None)
            row["id_auroc"]    = id_m.get("auroc",    None)

            for scenario in ood_scenarios:
                ood_ds = SCENARIO_DATASET[scenario]
                for score in SCORES[method]:
                    col = f"{scenario}_{score}"
                    row[col] = _read_ood_auroc(run_dir, scenario, ood_ds, score)

            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["group", "method", "train_size", "seed"], na_position="last"
    ).reset_index(drop=True)


def build_summary_table(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["group", "method", "train_size"]
    numeric_cols = [c for c in raw.columns if c not in group_cols + ["run_name", "seed", "run_dir"]]
    agg = []
    for keys, grp in raw.groupby(group_cols, dropna=False):
        row: dict = dict(zip(group_cols, keys))
        row["n_seeds"] = len(grp)
        for col in numeric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{col}_std"]  = float(vals.std(ddof=0)) if len(vals) > 1 else (float(vals.iloc[0]) if len(vals) == 1 else float("nan"))
        agg.append(row)
    return pd.DataFrame(agg).sort_values(group_cols, na_position="last").reset_index(drop=True)


def print_summary(summary: pd.DataFrame) -> None:
    print("\n=== Summary (mean ± std across seeds) ===\n")
    for group, gdf in summary.groupby("group"):
        print(f"--- {group} ---")
        rows = []
        for _, r in gdf.iterrows():
            method = r["method"]
            score  = PRIMARY_SCORE.get(method, "mutual_information")
            train_size = r.get("train_size")
            label = f"{method}" if pd.isna(train_size) else f"{method} n={int(train_size)}"

            def _fmt(scenario):
                col = f"{scenario}_{score}_mean"
                std_col = f"{scenario}_{score}_std"
                if col not in r or pd.isna(r[col]):
                    return "  —  "
                std = r.get(std_col, 0.0)
                return f"{r[col]:.3f}±{std:.3f}"

            row = {"experiment": label, "n_seeds": int(r["n_seeds"])}
            row["id_acc"] = f"{r['id_accuracy_mean']:.3f}" if "id_accuracy_mean" in r and not pd.isna(r.get("id_accuracy_mean", float("nan"))) else "—"
            row["far_OOD"] = _fmt("far_ood")
            row["near_OOD"] = _fmt("near_ood")
            if group == "longtail":
                row["long_tail"] = _fmt("long_tail")
            rows.append(row)

        print(pd.DataFrame(rows).to_string(index=False))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, metavar="S",
        help="Only include runs with these seeds. Default: all runs.",
    )
    args = parser.parse_args()

    outputs_dir = (PACKAGE_ROOT / args.outputs_dir).resolve()
    out_dir     = (PACKAGE_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning for all experiment runs...")
    raw = build_raw_table(outputs_dir, seed_filter=args.seeds)

    if raw.empty:
        print("No runs found.")
        return

    if args.seeds is not None:
        print(f"  [filter] seeds={args.seeds}: {len(raw)} rows kept")

    raw_csv = out_dir / "all_experiments_raw.csv"
    raw.to_csv(raw_csv, index=False)
    print(f"\nsaved raw: {raw_csv}")

    summary = build_summary_table(raw)
    summary_csv = out_dir / "all_experiments_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"saved summary: {summary_csv}")

    print_summary(summary)


if __name__ == "__main__":
    main()
