"""Centralized plotting for OOD/uncertainty figures.

Every figure-producing helper:

* Takes already-computed data (no inference, no metric recomputation).
* Accepts ``save_path`` and writes both ``<stem>.png`` (300 dpi) and
  ``<stem>.pdf``.
* Returns the :class:`matplotlib.figure.Figure` for optional display.

Importing this module applies a project-wide rcParams update once. All
project figures should be produced through these helpers so that a single
restyle changes everything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# project-wide style (applied at import time)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

COLORS = {
    "id": "#2E86AB",
    "ood": "#E63946",
    "deterministic": "#6C757D",
    "laplace": "#2E86AB",
    "mc_dropout": "#F4A261",
    "ensemble": "#2A9D8F",
    "diagonal": "#BBBBBB",
}

DATASET_COLORS = {
    "PneumoniaMNIST": "#2E86AB",
    "BloodMNIST": "#E63946",
    "OrganAMNIST": "#F4A261",
    "PathMNIST": "#2A9D8F",
}

# Distinguishable hues for uncertainty scores (used in ROC curves, bar charts).
SCORE_PALETTE = {
    "predictive_entropy": "#2E86AB",
    "mutual_information": "#E63946",
    "expected_entropy": "#F4A261",
    "one_minus_max_softmax": "#2A9D8F",
    "softmax_variance_sum": "#9B5DE5",
    "softmax_variance_max": "#C77DFF",
    "expected_pairwise_kl": "#F15BB5",
    "logit_variance_sum": "#00BBF9",
    "logit_variance_max": "#48CAE4",
}

# Colors for the four conceptual score families (category-grouped bar chart).
CATEGORY_COLORS = {
    "First-order (any model)": "#2A9D8F",
    "Information-theoretic decomposition (MC samples)": "#E63946",
    "Statistical spread (MC samples)": "#9B5DE5",
    "Analytical Gaussian spread (Laplace only)": "#00BBF9",
}


def _pretty(name: str) -> str:
    """Snake-case score name -> Title Case label."""
    overrides = {
        "predictive_entropy": "Predictive Entropy",
        "mutual_information": "Mutual Information",
        "expected_entropy": "Expected Entropy",
        "one_minus_max_softmax": "1 - Max Softmax",
        "softmax_variance_sum": "Softmax Var (sum)",
        "softmax_variance_max": "Softmax Var (max)",
        "expected_pairwise_kl": "Expected Pairwise KL",
        "logit_variance_sum": "Logit Var (sum)",
        "logit_variance_max": "Logit Var (max)",
    }
    return overrides.get(name, name.replace("_", " ").title())


def _save(fig, save_path: Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    png = save_path.with_suffix(".png")
    pdf = save_path.with_suffix(".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    return png


# ---------------------------------------------------------------------------
# 1. uncertainty histogram (ID vs OOD)
# ---------------------------------------------------------------------------
def plot_uncertainty_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    score_name: str,
    ood_name: str,
    auroc: float,
    save_path: Path,
):
    """Overlay density histograms of one uncertainty score for ID vs OOD.

    Uses ``density=True`` so unequal sample sizes don't visually dominate;
    sample counts are recorded in the legend instead.
    """
    id_scores = np.asarray(id_scores).ravel()
    ood_scores = np.asarray(ood_scores).ravel()

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bins = np.linspace(
        float(min(id_scores.min(), ood_scores.min())),
        float(max(id_scores.max(), ood_scores.max())),
        41,
    )
    ax.hist(id_scores, bins=bins, density=True, alpha=0.55,
            color=COLORS["id"], label=f"ID (n={len(id_scores)})")
    ax.hist(ood_scores, bins=bins, density=True, alpha=0.55,
            color=COLORS["ood"], label=f"OOD (n={len(ood_scores)})")
    ax.set_xlabel(_pretty(score_name))
    ax.set_ylabel("Density")
    ax.set_title(f"{_pretty(score_name)} — ID vs {ood_name}")
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.97, 0.95, f"AUROC = {auroc:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9),
    )
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. ROC curves overlaid (one per uncertainty score)
# ---------------------------------------------------------------------------
def plot_roc_curves(
    score_results: dict[str, tuple[np.ndarray, np.ndarray, float]],
    ood_name: str,
    save_path: Path,
):
    """All score ROC curves on one axis. Diagonal reference, square aspect."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, (fpr, tpr, auroc) in score_results.items():
        color = SCORE_PALETTE.get(name)
        ax.plot(fpr, tpr, label=f"{_pretty(name)} (AUROC = {auroc:.3f})",
                color=color, linewidth=1.6)
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["diagonal"], linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"OOD detection: ID test vs {ood_name}")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. AUROC bar chart across methods/scenarios/scores
# ---------------------------------------------------------------------------
def plot_auroc_bar_chart(results_table, save_path: Path):
    """Grouped AUROC bar chart.

    ``results_table`` is a long-format pandas DataFrame with columns
    ``method, scenario, score, auroc`` and an optional ``auroc_std`` column
    (used to draw error bars). One subplot row per ``method``; bars grouped
    by score with one bar per scenario.
    """
    import pandas as pd  # local import — pandas is a project dep
    df = pd.DataFrame(results_table) if not isinstance(results_table, pd.DataFrame) else results_table

    methods = list(df["method"].unique())
    scores = list(df["score"].unique())
    scenarios = list(df["scenario"].unique())
    has_err = "auroc_std" in df.columns

    fig, axes = plt.subplots(
        len(methods), 1, figsize=(1.6 * len(scores) + 2.5, 2.8 * len(methods)),
        sharex=True, sharey=True, squeeze=False,
    )
    x = np.arange(len(scores))
    n_sc = max(len(scenarios), 1)
    width = 0.8 / n_sc
    cmap = plt.get_cmap("tab10")

    for row, method in enumerate(methods):
        ax = axes[row, 0]
        sub = df[df["method"] == method]
        for j, scen in enumerate(scenarios):
            ssub = sub[sub["scenario"] == scen].set_index("score").reindex(scores)
            vals = ssub["auroc"].to_numpy(dtype=float)
            err = ssub["auroc_std"].to_numpy(dtype=float) if has_err else None
            ax.bar(
                x + j * width, vals, width, label=scen,
                color=cmap(j % 10),
                yerr=err, capsize=3 if has_err else 0,
            )
        ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="_random")
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUROC")
        ax.set_title(method)
        if row == 0:
            ax.legend(loc="upper right", frameon=False, ncol=min(n_sc, 4))

    axes[-1, 0].set_xticks(x + width * (n_sc - 1) / 2)
    axes[-1, 0].set_xticklabels([_pretty(s) for s in scores], rotation=15, ha="right")
    axes[-1, 0].set_xlabel("Uncertainty Score")
    # Tag the random reference line in the bottom subplot for clarity.
    axes[-1, 0].annotate(
        "random (0.5)", xy=(len(scores) - 0.6, 0.5), xycoords="data",
        xytext=(0, 5), textcoords="offset points", fontsize=8, color="k",
    )
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. epistemic vs aleatoric scatter (with marginals)
# ---------------------------------------------------------------------------
def plot_uncertainty_scatter(
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    is_ood: np.ndarray,
    save_path: Path,
    ood_name: str,
):
    """Per-sample epistemic-vs-aleatoric scatter with marginal histograms."""
    epistemic = np.asarray(epistemic).ravel()
    aleatoric = np.asarray(aleatoric).ravel()
    is_ood = np.asarray(is_ood).ravel().astype(bool)

    fig = plt.figure(figsize=(5.5, 5.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        hspace=0.04, wspace=0.04,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    for mask, label, color in [
        (~is_ood, "ID", COLORS["id"]),
        (is_ood, "OOD", COLORS["ood"]),
    ]:
        ax_main.scatter(epistemic[mask], aleatoric[mask],
                        s=10, alpha=0.3, color=color, label=label,
                        edgecolors="none")

    lo = float(min(epistemic.min(), aleatoric.min()))
    hi = float(max(epistemic.max(), aleatoric.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lim = (lo - pad, hi + pad)
    ax_main.set_xlim(lim)
    ax_main.set_ylim(lim)
    ax_main.plot(lim, lim, linestyle="--", color=COLORS["diagonal"], linewidth=1.0)

    ax_main.set_xlabel("Epistemic uncertainty (MI)")
    ax_main.set_ylabel("Aleatoric uncertainty (expected entropy)")
    ax_main.legend(loc="upper left", frameon=False)

    bins = np.linspace(*lim, 41)
    ax_top.hist(epistemic[~is_ood], bins=bins, density=True,
                color=COLORS["id"], alpha=0.55)
    ax_top.hist(epistemic[is_ood], bins=bins, density=True,
                color=COLORS["ood"], alpha=0.55)
    ax_right.hist(aleatoric[~is_ood], bins=bins, density=True,
                  color=COLORS["id"], alpha=0.55, orientation="horizontal")
    ax_right.hist(aleatoric[is_ood], bins=bins, density=True,
                  color=COLORS["ood"], alpha=0.55, orientation="horizontal")
    for a in (ax_top, ax_right):
        a.tick_params(axis="both", which="both",
                      bottom=False, left=False, labelbottom=False, labelleft=False)
        for s in a.spines.values():
            s.set_visible(False)

    fig.suptitle(f"Epistemic vs aleatoric — ID vs {ood_name}", y=0.94)
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. confidence histogram on OOD only ("model is confidently wrong")
# ---------------------------------------------------------------------------
def plot_confidence_histogram_on_ood(
    ood_max_softmax: np.ndarray,
    save_path: Path,
    ood_name: str,
):
    """Max softmax distribution on OOD samples; highlights confident-OOD region.

    If a working uncertainty signal existed, OOD confidence would concentrate
    near 1/C (uniform). A spike near 1.0 means the model is "confidently
    wrong" on OOD — the Li-et-al-style failure mode.
    """
    p = np.asarray(ood_max_softmax).ravel()
    above = float((p > 0.9).mean()) * 100.0

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(p, bins=bins, color=COLORS["ood"], alpha=0.85)
    ax.axvspan(0.9, 1.0, color=COLORS["ood"], alpha=0.12, label="conf > 0.9")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Max Softmax Probability on OOD")
    ax.set_ylabel("Count")
    ax.set_title(f"Model confidence on {ood_name} (OOD) — should be low if uncertainty works")
    ax.text(
        0.97, 0.95, f"P(conf > 0.9) = {above:.1f}%",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9),
    )
    ax.legend(loc="upper left", frameon=False)
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. reliability diagram for ID test set
# ---------------------------------------------------------------------------
def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    method_name: str,
    save_path: Path,
    n_bins: int = 15,
):
    """Standard reliability diagram (confidence on x, accuracy on y).

    Bars show per-bin accuracy; the diagonal is perfect calibration. ECE
    annotated in the top-left corner.
    """
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(y_pred_probs)
    conf = p.max(axis=-1)
    pred = p.argmax(axis=-1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    weights = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        accs[i] = correct[mask].mean()
        confs[i] = conf[mask].mean()
        weights[i] = mask.mean()
    valid = ~np.isnan(accs)
    ece = float(np.nansum(weights[valid] * np.abs(confs[valid] - accs[valid])))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    width = 1.0 / n_bins
    ax.bar(centers[valid], accs[valid], width=width * 0.95,
           color=COLORS["id"], alpha=0.85, label="Accuracy", edgecolor="white")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["diagonal"],
            linewidth=1.0, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Calibration: {method_name}")
    ax.text(
        0.04, 0.95, f"ECE = {ece:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9),
    )
    ax.legend(loc="lower right", frameon=False)
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. failure-mode grid (confidently wrong OOD + uncertain ID)
# ---------------------------------------------------------------------------
def _denorm_for_display(img: np.ndarray) -> np.ndarray:
    """Reverse the [-1, 1] normalization used by the loaders and shape to HWC."""
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * 0.5 + 0.5
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def plot_failure_modes(
    id_images: np.ndarray,
    id_uncertainty: np.ndarray,
    id_preds: np.ndarray,
    id_labels: np.ndarray,
    ood_images: np.ndarray,
    ood_uncertainty: np.ndarray,
    ood_preds: np.ndarray,
    save_path: Path,
    scenario_name: str,
    n_examples: int = 4,
    class_names: list[str] | None = None,
):
    """Two-row image grid of the most embarrassing predictions.

    * Row 1: ``n_examples`` OOD samples with the **lowest** uncertainty.
    * Row 2: ``n_examples`` ID samples with the **highest** uncertainty.

    Caption per cell: uncertainty value, predicted class, and (for ID) the
    true class.
    """
    id_uncertainty = np.asarray(id_uncertainty).ravel()
    ood_uncertainty = np.asarray(ood_uncertainty).ravel()

    # Row 1: OOD with lowest uncertainty (confident wrong) — ascending.
    ood_order = np.argsort(ood_uncertainty)[:n_examples]
    # Row 2: ID with highest uncertainty (confused on home turf) — descending.
    id_order = np.argsort(id_uncertainty)[::-1][:n_examples]

    fig, axes = plt.subplots(2, n_examples, figsize=(2.2 * n_examples, 5.0))
    if n_examples == 1:
        axes = np.array(axes).reshape(2, 1)

    def _label(idx: int) -> str:
        if class_names and 0 <= idx < len(class_names):
            return class_names[idx]
        return f"class {idx}"

    for k, idx in enumerate(ood_order):
        ax = axes[0, k]
        ax.imshow(_denorm_for_display(ood_images[idx]),
                  cmap="gray" if ood_images[idx].ndim < 3 or ood_images[idx].shape[0] == 1 else None)
        ax.set_title(
            f"u={ood_uncertainty[idx]:.3f}\npred: {_label(int(ood_preds[idx]))}",
            fontsize=9,
        )
        ax.axis("off")

    for k, idx in enumerate(id_order):
        ax = axes[1, k]
        ax.imshow(_denorm_for_display(id_images[idx]),
                  cmap="gray" if id_images[idx].ndim < 3 or id_images[idx].shape[0] == 1 else None)
        ax.set_title(
            f"u={id_uncertainty[idx]:.3f}\npred: {_label(int(id_preds[idx]))}"
            f"  (true: {_label(int(id_labels[idx]))})",
            fontsize=9,
        )
        ax.axis("off")

    axes[0, 0].annotate("OOD,\nlow uncertainty", xy=(-0.18, 0.5),
                        xycoords="axes fraction", ha="right", va="center",
                        fontsize=10, rotation=90)
    axes[1, 0].annotate("ID,\nhigh uncertainty", xy=(-0.18, 0.5),
                        xycoords="axes fraction", ha="right", va="center",
                        fontsize=10, rotation=90)
    fig.suptitle(f"Failure modes — {scenario_name}", y=1.0)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. AUROC bar chart grouped by conceptual score *family*
# ---------------------------------------------------------------------------
def plot_score_category_bar_chart(
    auroc_by_score: dict[str, float],
    categories: dict[str, list[str]],
    save_path: Path,
    title: str | None = None,
):
    """AUROC bars grouped by conceptual score family.

    ``auroc_by_score`` maps score name -> AUROC (scores absent / N/A are simply
    skipped). ``categories`` maps category label -> ordered list of score names
    (e.g. :data:`evaluation.ood.SCORE_CATEGORIES`). Bars are clustered by
    category and colored per score, making it visually clear which *family* of
    measures is useful for OOD detection.
    """
    # Short labels for the (long) category names, used above each group.
    short = {
        "First-order (any model)": "First-order",
        "Information-theoretic decomposition (MC samples)": "Information-theoretic",
        "Statistical spread (MC samples)": "Statistical spread",
        "Analytical Gaussian spread (Laplace only)": "Analytical Gaussian",
    }

    # Build the flat plotting order, keeping only scores we actually have.
    positions, heights, bar_colors = [], [], []
    group_spans, group_labels = [], []
    tick_positions, tick_labels = [], []
    x = 0.0
    gap = 1.0  # gap between categories
    for cat, names in categories.items():
        present = [n for n in names if n in auroc_by_score and auroc_by_score[n] is not None]
        if not present:
            continue
        start = x
        for n in present:
            positions.append(x)
            heights.append(float(auroc_by_score[n]))
            bar_colors.append(SCORE_PALETTE.get(n, "#888888"))
            tick_positions.append(x)
            tick_labels.append(_pretty(n))
            x += 1.0
        group_spans.append((start, x - 1.0))
        group_labels.append(short.get(cat, cat.split(" (")[0]))
        x += gap

    fig, ax = plt.subplots(figsize=(max(6.0, 0.85 * len(positions) + 2.0), 5.0))
    ax.bar(positions, heights, width=0.9, color=bar_colors, edgecolor="white")
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.set_ylabel("AUROC")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)

    # Category bands: a bracket + label above each group, below the title.
    for (lo, hi), label in zip(group_spans, group_labels):
        center = (lo + hi) / 2.0
        ax.plot([lo - 0.45, hi + 0.45], [1.005, 1.005], transform=ax.get_xaxis_transform(),
                color="0.4", linewidth=1.0, clip_on=False)
        ax.text(center, 1.02, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", clip_on=False)
    ax.annotate(
        "random (0.5)", xy=(positions[-1], 0.5), xytext=(4, 3),
        textcoords="offset points", fontsize=8, color="k", va="bottom",
    )
    ax.set_title(title or "OOD-detection AUROC by score family", pad=22)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 9. MI vs logit-variance scatter (Hüllermeier: do they carry the same info?)
# ---------------------------------------------------------------------------
def plot_score_correlation_scatter(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    is_ood: np.ndarray,
    save_path: Path,
    x_label: str,
    y_label: str,
    title: str | None = None,
):
    """Per-sample scatter of two uncertainty scores, colored by ID/OOD.

    Annotates the Pearson correlation. If the two scores are highly correlated
    they carry the same information; if not, they pick up different aspects of
    the posterior — the direct visualization of Hüllermeier's point that the
    choice of second-order summary statistic matters.
    """
    x_scores = np.asarray(x_scores).ravel()
    y_scores = np.asarray(y_scores).ravel()
    is_ood = np.asarray(is_ood).ravel().astype(bool)

    finite = np.isfinite(x_scores) & np.isfinite(y_scores)
    if finite.sum() >= 2 and np.std(x_scores[finite]) > 0 and np.std(y_scores[finite]) > 0:
        pearson = float(np.corrcoef(x_scores[finite], y_scores[finite])[0, 1])
    else:
        pearson = float("nan")

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for mask, label, color in [
        (~is_ood, "ID", COLORS["id"]),
        (is_ood, "OOD", COLORS["ood"]),
    ]:
        ax.scatter(x_scores[mask], y_scores[mask], s=10, alpha=0.3,
                   color=color, label=label, edgecolors="none")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"{x_label} vs {y_label}")
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.97, 0.04, f"Pearson r = {pearson:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9),
    )
    _save(fig, save_path)
    return fig, pearson


__all__ = [
    "COLORS",
    "DATASET_COLORS",
    "SCORE_PALETTE",
    "CATEGORY_COLORS",
    "plot_uncertainty_histogram",
    "plot_roc_curves",
    "plot_auroc_bar_chart",
    "plot_uncertainty_scatter",
    "plot_confidence_histogram_on_ood",
    "plot_reliability_diagram",
    "plot_failure_modes",
    "plot_score_category_bar_chart",
    "plot_score_correlation_scatter",
]
