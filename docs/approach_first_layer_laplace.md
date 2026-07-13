# Approach: First-Layer Laplace

> Status: initial write-up — will be extended as results come in.

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

## Open / planned extensions

- Results tables (ID, long-tail, near/far OOD) vs. baseline / LLL / ensembles.
- Other Bayesian extents (first block `layer1`, first+last combined).
- `hessian_structure: diag` fallback if the full fit gets slow at scale.
