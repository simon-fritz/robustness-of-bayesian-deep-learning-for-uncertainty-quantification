"""Evaluation entry point.

For deterministic runs: single forward pass, softmax, ID metrics.
For Bayesian runs (last-layer Laplace etc.): MC predictive samples, mean
metrics + uncertainty stats, raw samples saved for downstream OOD analysis.

Usage:
    python scripts/evaluate.py --run-dir outputs/<run_name>/<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from bnn_medmnist.data.medmnist_loader import MedMNISTLoader
from bnn_medmnist.evaluation.metrics import (
    accuracy,
    auroc,
    balanced_accuracy,
    brier_score,
    expected_calibration_error,
    nll,
)
from bnn_medmnist.evaluation.plots import plot_reliability_diagram
from bnn_medmnist.evaluation.uncertainty import (
    expected_entropy,
    mutual_information,
    predictive_entropy,
)
from bnn_medmnist.models import build_model


def _load_model(cfg, ckpt_path: Path, device: str, num_classes: int, in_channels: int):
    model = build_model(cfg.model, in_channels=in_channels, num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _deterministic_samples(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(probs[1, N, C], y[N])`` so the downstream code is sample-agnostic."""
    all_p, all_y = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=-1).cpu()
        all_p.append(p)
        all_y.append(y)
    probs = torch.cat(all_p).unsqueeze(0)  # (1, N, C)
    return probs, torch.cat(all_y)


@torch.no_grad()
def _laplace_samples(
    la, loader, device, n_samples: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(probs[S, N, C], y[N], logit_mean[N, C], logit_var[N, C], logit_sigma[N, C])``.

    The MC softmax samples drive the entropy/spread scores; the analytical
    Gaussian over logits (``pred_type="glm"``) gives the sampling-free
    logit-variance scores. ``logit_sigma = sqrt(logit_var)`` is the same
    Gaussian's standard deviation, on the natural (same-units-as-logits) scale.
    """
    all_p, all_y, all_lm, all_lv, all_ls = [], [], [], [], []
    for x, y in loader:
        xb = x.to(device)
        s = la.predictive_samples(xb, pred_type="nn", n_samples=n_samples).cpu()
        f_mu, f_var = la._glm_predictive_distribution(xb, diagonal_output=True)
        all_p.append(s)
        all_y.append(y)
        all_lm.append(f_mu.cpu())
        all_lv.append(f_var.cpu())
        all_ls.append(f_var.clamp_min(0).sqrt().cpu())
    return (
        torch.cat(all_p, dim=1),
        torch.cat(all_y),
        torch.cat(all_lm, dim=0),
        torch.cat(all_lv, dim=0),
        torch.cat(all_ls, dim=0),
    )


@torch.no_grad()
def _deep_ensemble_samples(members, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(probs[S, N, C], y[N])`` from the ensemble."""
    all_p, all_y = [], []
    for x, y in loader:
        batch_probs = []
        for member in members:
            probs = torch.softmax(member(x.to(device)), dim=-1).cpu()
            batch_probs.append(probs)
        
        s = torch.stack(batch_probs, dim=0)
        all_p.append(s)
        all_y.append(y)
    
    return torch.cat(all_p, dim=1), torch.cat(all_y)

# Pretty labels + "better" direction for the printed overview. Anything not
# listed falls back to its raw key with no arrow.
_METRIC_META: dict[str, tuple[str, str]] = {
    "accuracy": ("accuracy", "↑"),
    "balanced_accuracy": ("balanced accuracy", "↑"),
    "auroc": ("AUROC", "↑"),
    "ece": ("ECE (calibration error)", "↓"),
    "nll": ("NLL (-log p of true label)", "↓"),
    "brier": ("Brier score", "↓"),
    "mean_predictive_entropy": ("predictive entropy (total)", " "),
    "mean_expected_entropy": ("expected entropy (aleatoric)", " "),
    "mean_mutual_information": ("mutual information (epistemic)", " "),
    "max_mutual_information": ("max mutual information", " "),
}


def _print_metrics(
    method_name: str,
    n_samples: int,
    metrics: dict[str, float],
    groups: list[tuple[str, list[str]]],
) -> None:
    """Print metrics grouped into labelled sections with a direction hint."""
    label_w = max(len(_METRIC_META.get(k, (k, ""))[0]) for k in metrics)
    header = f" Test metrics — {method_name} "
    bar = "═" * max(len(header), label_w + 14)
    print(f"\n{bar}\n{header}\n{bar}")
    for title, keys in groups:
        print(f"\n{title}")
        for k in keys:
            if k not in metrics:
                continue
            label, arrow = _METRIC_META.get(k, (k, " "))
            print(f"  {arrow} {label:<{label_w}}  {metrics[k]:>8.4f}")
    print(f"\n{bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override predictive sample count for Bayesian methods.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.resolve(cfg)
    method_name = str(cfg.method.get("name", "deterministic")).lower()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path((run_dir / "checkpoint_path.txt").read_text().strip())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = MedMNISTLoader(cfg.data)

    logit_mean = logit_var = logit_sigma = None
    if method_name == "deterministic":
        model = _load_model(cfg, ckpt_path, device, data.metadata.num_classes, data.metadata.in_channels)
        probs_samples, y = _deterministic_samples(model, data.test_loader(), device)
    elif method_name == "last_layer_laplace":
        model = _load_model(cfg, ckpt_path, device, data.metadata.num_classes, data.metadata.in_channels)
        from laplace import Laplace
        la_path = ckpt_path.with_suffix(".laplace.pt")
        payload = torch.load(la_path, map_location=device, weights_only=False)
        la = Laplace(
            model, likelihood="classification",
            subset_of_weights=payload["subset_of_weights"],
            hessian_structure=payload["hessian_structure"],
        )
        la.load_state_dict(payload["state_dict"])
        n_samples = int(args.n_samples or cfg.method.laplace.n_predictive_samples)
        print(f"drawing {n_samples} predictive samples per test example...", flush=True)
        probs_samples, y, logit_mean, logit_var, logit_sigma = _laplace_samples(
            la, data.test_loader(), device, n_samples
        )

    elif method_name == "first_layer_laplace":
        model = _load_model(cfg, ckpt_path, device, data.metadata.num_classes, data.metadata.in_channels)
        from laplace import Laplace
        la_path = ckpt_path.with_suffix(".laplace.pt")
        payload = torch.load(la_path, map_location=device, weights_only=False)
        # SubnetLaplace requires subnetwork_indices at reload — fit() stashes
        # them in the payload. The library's isinstance check demands a CPU
        # torch.LongTensor (NOT torch.cuda.LongTensor), so keep on CPU.
        la = Laplace(
            model, likelihood="classification",
            subset_of_weights=payload["subset_of_weights"],
            hessian_structure=payload["hessian_structure"],
            subnetwork_indices=payload["subnetwork_indices"].cpu().long(),
        )
        la.load_state_dict(payload["state_dict"])
        n_samples = int(args.n_samples or cfg.method.laplace.n_predictive_samples)
        # GLM (_glm_predictive_distribution) uses the same expensive
        # torch.func.jacrev as the fit — hours on the test set. Skip it: MC
        # samples only, no analytical logit moments (identical treatment to
        # deep_ensemble). logit_variance scores end up N/A downstream.
        print(f"drawing {n_samples} MC predictive samples (GLM skipped for Conv2d subnet)...", flush=True)
        all_p, all_y = [], []
        for x, y_b in data.test_loader():
            xb = x.to(device)
            s = la.predictive_samples(xb, pred_type="nn", n_samples=n_samples).cpu()
            all_p.append(s)
            all_y.append(y_b)
        probs_samples = torch.cat(all_p, dim=1)
        y = torch.cat(all_y)
        logit_mean = logit_var = logit_sigma = None

    elif method_name == "deep_ensemble":
        # Discover ensemble member checkpoints in the same directory as the main checkpoint.
        ckpt_dir = ckpt_path.parent
        member_files = sorted(ckpt_dir.glob("member_*.pt"))
        if not member_files:
            # Fallback: try a configured n_members or default to 5
            try:
                n_members = int(cfg.method.get("n_members", 5))
            except Exception:
                n_members = 5
            member_files = [ckpt_dir / f"member_{i}.pt" for i in range(n_members)]

        members = []
        for member_ckpt in member_files:
            if not Path(member_ckpt).exists():
                raise FileNotFoundError(f"Ensemble member checkpoint not found: {member_ckpt}")
            member = _load_model(cfg, member_ckpt, device, data.metadata.num_classes, data.metadata.in_channels)
            members.append(member)

        probs_samples, y = _deep_ensemble_samples(members, data.test_loader(), device)
            
    else:
        raise NotImplementedError(f"evaluate not implemented for method '{method_name}'")

    mean_probs = probs_samples.mean(dim=0)

    # Metrics grouped by what they tell you. The dict written to JSON stays flat
    # (one key per metric); ``groups`` only drives the printed overview.
    metrics: dict[str, float] = {
        "accuracy": accuracy(y, mean_probs),
        "balanced_accuracy": balanced_accuracy(y, mean_probs),
        "auroc": auroc(y, mean_probs),
        "ece": expected_calibration_error(y, mean_probs),
        "nll": nll(y, mean_probs),
        "brier": brier_score(y, mean_probs),
    }
    groups: list[tuple[str, list[str]]] = [
        ("Classification", ["accuracy", "balanced_accuracy", "auroc"]),
        ("Calibration / proper scores", ["ece", "nll", "brier"]),
    ]

    if probs_samples.shape[0] > 1:
        pe = predictive_entropy(probs_samples)
        ee = expected_entropy(probs_samples)
        mi = mutual_information(probs_samples)
        metrics.update({
            "mean_predictive_entropy": float(pe.mean()),
            "mean_expected_entropy": float(ee.mean()),
            "mean_mutual_information": float(mi.mean()),
            "max_mutual_information": float(mi.max()),
        })
        groups.append((
            f"Uncertainty (mean over test set, {probs_samples.shape[0]} samples)",
            [
                "mean_predictive_entropy",
                "mean_expected_entropy",
                "mean_mutual_information",
                "max_mutual_information",
            ],
        ))

    _print_metrics(method_name, probs_samples.shape[0], metrics, groups)

    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    save_arrays = {
        "probs_samples": probs_samples.numpy().astype(np.float32),
        "labels": y.numpy().astype(np.int64),
    }
    # Analytical Gaussian over logits (Laplace only); deterministic runs have
    # no posterior over weights, so these fields are simply omitted.
    if logit_mean is not None and logit_var is not None:
        save_arrays["logit_mean"] = logit_mean.numpy().astype(np.float32)
        save_arrays["logit_var"] = logit_var.numpy().astype(np.float32)
        save_arrays["logit_sigma"] = logit_sigma.numpy().astype(np.float32)
    np.savez(run_dir / "test_predictions.npz", **save_arrays)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel_path = fig_dir / "reliability_diagram"
    plot_reliability_diagram(
        y.numpy(), mean_probs.numpy(), method_name=method_name, save_path=rel_path,
    )
    print(f"\nsaved: {run_dir / 'test_metrics.json'}")
    print(f"saved: {run_dir / 'test_predictions.npz'}  (shape={tuple(probs_samples.shape)})")
    print(f"saved: {rel_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
