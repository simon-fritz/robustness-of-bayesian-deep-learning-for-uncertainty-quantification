# Results: ResNet-18, deterministic vs. last-layer Laplace

Run `20260603_125825`. Same ResNet-18, once deterministic (`baseline`) and once
with last-layer Laplace (`lll`). In-distribution = PneumoniaMNIST, far-OOD =
BloodMNIST. (Near-OOD not yet evaluated.)

## Classification

| Metric | baseline | lll (Laplace) |
|---|---|---|
| accuracy | 0.913 | 0.909 |
| balanced accuracy | 0.885 | 0.880 |
| AUROC | 0.980 | 0.981 |
| **ECE** | **0.061** | **0.022** |

Laplace costs no accuracy but calibrates ~3× better (lower ECE). The point
predictions are essentially unchanged; only the confidence becomes honest.

## OOD detection (AUROC, higher = better)

Scores split into two conceptual orders (Hüllermeier & Waegeman 2021):

- **First-order** — properties of a *single* predictive distribution (its
  confidence / entropy). Defined for any model, including the deterministic one.
- **Second-order** — the *spread of the posterior over distributions*, i.e.
  epistemic uncertainty. Needs a posterior, so a deterministic model has none
  (these collapse to ≈0.500 / "—").

### First-order scores

| Score | captures | baseline | lll |
|---|---|---|---|
| predictive_entropy | total | 0.618 | 0.810 |
| one_minus_max_softmax | total / confidence | 0.618 | 0.810 |
| expected_entropy | aleatoric | 0.618 | 0.722 |

### Second-order scores (epistemic / posterior spread)

| Score | captures | baseline | lll |
|---|---|---|---|
| mutual_information | epistemic (BALD) | 0.500 | 0.865 |
| softmax_variance | spread (MC) | 0.500 | 0.837 |
| expected_pairwise_kl | spread (MC) | — | 0.928 |
| logit_variance | analytical Gaussian (Laplace only) | — | **0.956** |

## Three takeaways

1. **A deterministic model has no epistemic uncertainty.** With one forward pass
   the posterior spread is exactly zero, so `mutual_information` and
   `softmax_variance` collapse to AUROC 0.500 (random, FPR@95 = 1.0). These
   scores only exist once the model is Bayesian.

2. **Last-layer Laplace makes epistemic uncertainty usable** — even though only
   the `fc` layer is Bayesian. `mutual_information` jumps from 0.500 to 0.865,
   and even first-order entropy improves (MC-averaging makes the mean softmax
   less overconfident).

3. **Mutual information is not the best score.** The analytical Gaussian
   logit-variance (Laplace-only, sampling-free) wins at 0.956 (FPR@95 ≈ 0.10),
   ahead of expected pairwise KL (0.928) and well ahead of MI (0.865). This is
   why we compare several score *families* instead of relying on entropy/MI
   alone (Hüllermeier & Waegeman 2021).

## Caveat

Far-OOD (BloodMNIST) is the easy case — it looks nothing like a chest X-ray. The
harder near-OOD evaluation is still missing, and that is where epistemic
uncertainty as an OOD detector is actually tested (cf. Li et al. 2025).
</content>
