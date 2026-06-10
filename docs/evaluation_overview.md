# Evaluation Overview

A plain-language description of how we evaluate a trained model. There are two
steps, run by two scripts:

1. **`scripts/evaluate.py`** — how good is the model on the normal (in-distribution) test set?
2. **`scripts/evaluate_ood.py`** — can the model tell when an input is *out-of-distribution* (OOD), i.e. unlike anything it was trained on?

Both scripts work on an existing trained run; neither retrains anything. You
point them at a run directory (`outputs/<run_name>/<timestamp>/`) and they load
the checkpoint recorded in that run's `checkpoint_path.txt`.

---

## 1. In-distribution evaluation (`evaluate.py`)

We run the model over the held-out test set and measure two things:

**Is it accurate?**
- **Accuracy** — fraction of correct predictions. ↑ better
- **Balanced accuracy** — accuracy averaged per class, so a rare class still
  counts (important for our imbalanced medical data). ↑ better
- **AUROC** — how well the predicted probability separates the classes. ↑ better

**Does it know how confident to be?** (calibration / proper scores)
- **ECE** (Expected Calibration Error) — gap between confidence and actual
  correctness. A well-calibrated model that says "90% sure" is right ~90% of the
  time. ↓ better
- **NLL** (negative log-likelihood) and **Brier score** — reward being both
  correct *and* appropriately confident. ↓ better

**How uncertain is it?** (Bayesian runs only — Last-Layer Laplace, Deep Ensemble)

A deterministic model gives one answer. A Bayesian model gives many (we draw
~100 predictive samples), and the spread of those answers is the uncertainty.
We report the average over the test set of:
- **Predictive entropy** — total uncertainty in the final prediction.
- **Expected entropy** — uncertainty from the data itself (ambiguous images).
- **Mutual information** — uncertainty from the *model* not knowing
  (the part that should spike on unfamiliar inputs).

**Outputs written to the run directory:**
- `test_metrics.json` — all the numbers above.
- `test_predictions.npz` — raw predicted probabilities (and, for Laplace, the
  logit mean/variance) for downstream analysis.
- `figures/reliability_diagram.*` — a calibration plot.

---

## 2. Out-of-distribution evaluation (`evaluate_ood.py`)

The core question of the project: **when shown something it shouldn't recognize,
does the model become uncertain?** A good uncertainty-aware model gives low
uncertainty on in-distribution (ID) data and high uncertainty on OOD data.

### How it works

For every test image we compute a single **uncertainty score**. We use several
score families (entropy of the prediction, mutual information / disagreement
between samples, and — for Laplace — a sampling-free score from the analytical
variance over the logits). Then we ask: do OOD images get higher scores than ID
images? That separation is measured with:

- **AUROC** — how cleanly the score separates ID from OOD. 1.0 = perfect, 0.5 = useless. ↑ better
- **AUPRC** — same idea, precision/recall flavour. ↑ better
- **FPR@95%TPR** — of the OOD images we want to catch (95% of them), how many ID
  images get falsely flagged as OOD. ↓ better

### The OOD scenarios we test (for a PneumoniaMNIST model)

| Scenario | Config | What counts as "OOD" | Meaning |
|----------|--------|----------------------|---------|
| **Far-OOD** | `pneumonia_far_blood.yaml` | BloodMNIST images | A completely different kind of image — should be easy to flag. |
| **Near-OOD** | `pneumonia_near_organ.yaml` | OrganAMNIST images | Still medical, more similar — harder to flag. |
| **Long-tail** | `pneumonia_longtail.yaml` | The under-represented "normal" class | The model was trained with the "normal" class subsampled to 2%, so that class is effectively rare/unfamiliar. Tests whether under-representation shows up as uncertainty. |

Far- and near-OOD are *post-hoc*: they work on any trained pneumonia run.
**Long-tail only works on a run that was trained with the matching class
subsampling** (`class_subsampling: {0: 0.02}`) — the script checks this and
errors out otherwise.

### Outputs written to the run directory
Under `ood/<scenario>/`:
- `ood_metrics.json` — AUROC / AUPRC / FPR@95 per score family, per OOD set.
- `*_predictions.npz` — per-sample scores.
- `figures/` — score histograms (ID vs OOD), ROC curves, and AUROC bar charts.

---

## What we run, end to end

Each training job (`slurm/longtail_*.sbatch`, `slurm/resnet18_*.sbatch`) trains a
model and then immediately runs `evaluate.py` plus `evaluate_ood.py` on the
relevant scenarios. To re-score already-trained runs after improving the
metrics code, `slurm/reeval_all.sbatch` re-runs both eval scripts without
retraining.

The models we compare:
- **Deterministic** baseline — single prediction, a reference point.
- **Last-Layer Laplace (LLL)** — Bayesian last layer, cheap uncertainty.
- **Deep Ensemble** — several independently trained models, strong uncertainty baseline.