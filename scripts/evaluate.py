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
    expected_calibration_error,
)
from bnn_medmnist.evaluation.plots import plot_reliability_diagram
from bnn_medmnist.evaluation.uncertainty import (
    expected_entropy,
    mutual_information,
    predictive_entropy,
)
from bnn_medmnist.models.small_cnn import SmallCNN


def _load_model(cfg, ckpt_path: Path, device: str, num_classes: int, in_channels: int) -> SmallCNN:
    model = SmallCNN(
        in_channels=in_channels, num_classes=num_classes,
        dropout=float(cfg.model.get("dropout", 0.0)),
    ).to(device)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(probs[S, N, C], y[N], logit_mean[N, C], logit_var[N, C])``.

    The MC softmax samples drive the entropy/spread scores; the analytical
    Gaussian over logits (``pred_type="glm"``) gives the sampling-free
    logit-variance scores.
    """
    all_p, all_y, all_lm, all_lv = [], [], [], []
    for x, y in loader:
        xb = x.to(device)
        s = la.predictive_samples(xb, pred_type="nn", n_samples=n_samples).cpu()
        f_mu, f_var = la._glm_predictive_distribution(xb, diagonal_output=True)
        all_p.append(s)
        all_y.append(y)
        all_lm.append(f_mu.cpu())
        all_lv.append(f_var.cpu())
    return (
        torch.cat(all_p, dim=1),
        torch.cat(all_y),
        torch.cat(all_lm, dim=0),
        torch.cat(all_lv, dim=0),
    )


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
    model = _load_model(cfg, ckpt_path, device, data.metadata.num_classes, data.metadata.in_channels)

    logit_mean = logit_var = None
    if method_name == "deterministic":
        probs_samples, y = _deterministic_samples(model, data.test_loader(), device)
    elif method_name == "last_layer_laplace":
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
        probs_samples, y, logit_mean, logit_var = _laplace_samples(
            la, data.test_loader(), device, n_samples
        )
    else:
        raise NotImplementedError(f"evaluate not implemented for method '{method_name}'")

    mean_probs = probs_samples.mean(dim=0)

    metrics: dict[str, float] = {
        "accuracy": accuracy(y, mean_probs),
        "balanced_accuracy": balanced_accuracy(y, mean_probs),
        "auroc": auroc(y, mean_probs),
        "ece": expected_calibration_error(y, mean_probs),
    }

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

    width = max(len(k) for k in metrics)
    print("\nTest metrics")
    print("-" * (width + 12))
    for k, v in metrics.items():
        print(f"{k:<{width}}  {v:.4f}")

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
