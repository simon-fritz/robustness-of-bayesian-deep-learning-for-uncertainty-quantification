# Approach: First-Layer Laplace

> Status: implemented and evaluated (PneumoniaMNIST / ResNet-18, seed 42).
> Results below; still to sweep more seeds and compare against LLL / ensembles.

## Idea

Last-layer Laplace (LLL) puts the Gaussian posterior over the classifier head
(`fc`, 1,026 params on ResNet-18). First-layer Laplace (FLL) is the mirror
experiment: the posterior sits over the stem conv (`conv1`, 9,408 params)
while everything else stays at its MAP value. Comparing the two probes *where*
in the network the Bayesian treatment matters for uncertainty quality (ID
calibration, long-tail, near/far OOD).

## How it works

Two phases, like LLL (`src/bnn_medmnist/methods/first_layer_laplace.py`):

1. **MAP training** — byte-identical to LLL's phase 1 (15-epoch fine-tune).
   Since the Laplace fit is post-hoc, the MAP model can optionally be *reused*
   rather than retrained: `method.reuse_map_from: <run_name>` picks the
   same-seed checkpoint of a finished run (seed matched via each run's saved
   `config.yaml`; unreadable/other-user runs are skipped), or
   `method.map_checkpoint: <path>` points at an explicit `best.pt`. Reuse makes
   the FLL-vs-LLL comparison exact (identical MAP weights) and skips the
   fine-tune. The default configs retrain from scratch (self-contained).
2. **Subnetwork Laplace fit** (`laplace-torch`, `subset_of_weights=
   "subnetwork"`, full GGN Hessian — 9,408² is small). Implementation detail:
   gradients are disabled for everything except the target modules before the
   fit, because laplace-torch otherwise materializes full-network Jacobians
   `(batch, classes, 11.7M)` and only then column-selects the subnetwork.
   With the trick, Jacobians are subnet-sized. Peak fit memory then scales
   ~ `batch × p²` (the full GGN builds a `(batch, classes, p, p)`-ish einsum
   intermediate; for conv1's `p=9408` that is ~0.7 GB per batch element), so the
   fit uses a dedicated small-batch loader (`laplace.fit_batch_size: 4`) — a
   memory knob, not a speed knob. `fit_batch_size=32` OOMs a 64 GB job.

Target modules are configurable (`method.bayesian_layers`, dotted paths
allowed) — `["conv1"]` on ResNet, `["layer1.0"]` on SmallCNN; later e.g.
`["layer1"]` for a whole block.

Prediction uses the **GLM** (linearized, function-space) predictive, not `"nn"`
weight sampling. conv1's posterior is wide (~ the prior; per-weight std >> the
weight magnitude), so sampling those weights and running them forward scrambles
the stem conv and collapses predictions to the majority class — measured MC acc
0.62 / AUROC 0.45 vs GLM acc 0.92 on PneumoniaMNIST/ResNet18. The GLM predictive
linearizes around the MAP, preserving accuracy while still propagating the
posterior logit variance for uncertainty. Set via `method.laplace.pred_type`
(`glm` for first-layer, `nn` for last-layer whose posterior is well-constrained).

**Eval-time memory.** The GLM predictive runs the same conv1 jacrev as the fit,
which vmaps `batch × classes` backward passes through the whole net, so peak GPU
memory scales ~ `batch × activations` (batch 16 alone needs ~43 GB on
ResNet18@224). Evaluation caps the loader batch to `laplace.eval_batch_size` (4),
overridable per-run with `--eval-batch-size`. This is separate from
`fit_batch_size`; both are memory knobs for the same jacrev.

## Running it

```bash
# standard (trains its own MAP, then the first-layer Laplace)
sbatch slurm/resnet18_fll.sbatch <seed>

# long-tail
sbatch slurm/train_longtail_generic.sbatch configs/experiment/pneumonia_resnet18_longtail_fll.yaml <seed>
```

To skip the MAP fine-tune by reusing an existing same-seed checkpoint, add
`reuse_map_from` / `map_checkpoint` to the experiment config; if the referenced
run's checkpoint is missing or unreadable, training aborts with a clear message.

Local smoke test (SmallCNN, CPU):

```bash
python scripts/train.py --config configs/experiment/training/pneumonia_fll.yaml --smoke
```

## Results (PneumoniaMNIST / ResNet-18, seed 42)

GLM predictive, `eval_batch_size=4`. Full metrics in each run's
`test_metrics.json` / `ood/<scenario>/ood_metrics.json`.

**In-distribution (test):**

| run       | accuracy | balanced acc | AUROC | ECE   |
|-----------|---------:|-------------:|------:|------:|
| standard  |    0.923 |        0.898 | 0.985 | 0.103 |
| long-tail |    0.700 |        0.601 | 0.890 | 0.132 |

The long-tail run subsamples the "normal" class to 2% (24 train images), so the
tail class — not the method — drives the drop in balanced accuracy / calibration.

**OOD detection (AUROC), one score per family** (`logit variance` = `logit_variance_sum`):

| training run | shift | pred. entropy | exp. entropy (aleatoric) | mutual info (epistemic) | logit variance (Laplace) |
|---|---|--:|--:|--:|--:|
| standard  | far-OOD (bloodmnist, semantic)   | 0.451 | 0.446 | 0.464 | 0.552 |
| standard  | near-OOD (organamnist, semantic) | 0.831 | **0.898** | 0.734 | 0.571 |
| long-tail | far-OOD (bloodmnist, semantic)   | 0.717 | 0.345 | 0.779 | 0.849 |
| long-tail | near-OOD (organamnist, semantic) | 0.429 | 0.320 | 0.514 | 0.679 |
| long-tail | tail class (density)             | 0.882 | 0.672 | **0.910** | **0.929** |

On the *standard* model every far-OOD score sits at/below chance (`logit_variance_max`
reaches only 0.65) — the stem's posterior gives essentially no semantic-OOD signal.
The near-OOD row is from the standard model; the long-tail rows are from the
long-tail model.

**Takeaway.** The epistemic signal fires on the **density** shift, not the
**semantic** ones. On long-tail data-scarcity the epistemic scores win (logit
variance 0.93, mutual information 0.91) — what a Bayesian feature extractor
*should* flag — while on semantic shift they are weakest: near-OOD logit variance
0.57 (aleatoric entropy leads at 0.90) and far-OOD collapses to chance for every
score (~0.45). So first-layer epistemic uncertainty tracks *how much data* was
seen, not *how semantically far* an input is — the same Li et al. (2025) pattern
the [long-tail experiment](results_long_tail.md) shows.

**FLL vs. LLL — where the Bayesian treatment matters.** The signal that made
last-layer Laplace a good *semantic* far-OOD detector does **not** transfer to the
first layer: full-data far-OOD `logit_variance` is ≈0.94 (5 seeds) for LLL but only
0.55 for FLL. The useful semantic-OOD signal is a property of the *last-layer*
posterior, not the stem. Two caveats before over-reading this: FLL is
**single-seed (42)** so far, and its calibration is *worse* than LLL's (standard
ECE 0.10 vs. ~0.02) — the wide conv1 posterior inflates the predictive spread. A
same-table head-to-head across 5 seeds (and a first+last-combined variant) is the
natural next step.
