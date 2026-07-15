# Results: ResNet-18, deterministic vs. last-layer Laplace

Run `20260603_125825`. Same ResNet-18, once deterministic (`baseline`) and once
with last-layer Laplace (`lll`). In-distribution = PneumoniaMNIST, far-OOD =
BloodMNIST (colour blood-cell microscopy), near-OOD = OrganAMNIST (grayscale
abdominal CT slices — same modality family as a chest X-ray).

> **Note (seeds).** This is a **single-seed** comparison (run `20260603_125825`),
> kept for its per-score detail. For 5-seed means ± std across all
> methods/scenarios see [../results/all_experiments_summary.csv](../results/all_experiments_summary.csv);
> the single-seed figures below are representative but not the averaged values
> (e.g. LLL far-OOD `logit_variance` reads 0.956 here vs. 0.94 ± 0.05 over 5 seeds).

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

| Score | captures | base far | lll far | base near | lll near |
|---|---|---|---|---|---|
| predictive_entropy | total | 0.618 | 0.810 | 0.904 | 0.907 |
| one_minus_max_softmax | total / confidence | 0.618 | 0.810 | 0.904 | 0.907 |
| expected_entropy | aleatoric | 0.618 | 0.722 | 0.904 | 0.879 |

### Second-order scores (epistemic / posterior spread)

| Score | captures | base far | lll far | base near | lll near |
|---|---|---|---|---|---|
| mutual_information | epistemic (BALD) | 0.500 | 0.865 | 0.500 | 0.928 |
| softmax_variance | spread (MC) | 0.500 | 0.837 | 0.500 | 0.924 |
| expected_pairwise_kl | spread (MC) | — | 0.928 | — | **0.931** |
| logit_variance | analytical Gaussian (Laplace only) | — | **0.956** | — | 0.774 |

## Three takeaways

1. **A deterministic model has no epistemic uncertainty.** With one forward pass
   the posterior spread is exactly zero, so `mutual_information` and
   `softmax_variance` collapse to AUROC 0.500 (random, FPR@95 = 1.0). These
   scores only exist once the model is Bayesian.

2. **Last-layer Laplace makes epistemic uncertainty usable** — even though only
   the `fc` layer is Bayesian. `mutual_information` jumps from 0.500 to 0.865,
   and even first-order entropy improves (MC-averaging makes the mean softmax
   less overconfident).

3. **No single score wins everywhere — the best one flips with the OOD type.**
   On far-OOD the analytical Gaussian `logit_variance` is best (0.956), but on
   near-OOD it is the *worst* Laplace score (0.774). The robust scores across
   both are `mutual_information` (0.865 / 0.928) and `expected_pairwise_kl`
   (0.928 / 0.931). This is exactly why we compare several score *families*
   instead of trusting one (Hüllermeier & Waegeman 2021): a score tuned on
   far-OOD can mislead on near-OOD.

## Caveats

- **Far vs. near behave differently.** Far-OOD (BloodMNIST, colour) is visually
  obvious, yet near-OOD (OrganAMNIST, grayscale) is actually *easier* for plain
  first-order confidence (entropy 0.90 vs 0.62) — even the deterministic model
  separates it well. The Bayesian second-order scores mainly help on far-OOD and
  on the harder-to-summarise cases; on near-OOD the gap to first-order is small.
- **Only one dataset per OOD type.** One far- and one near-OOD set is thin
  evidence; conclusions about *which* score to trust need more OOD datasets and
  multiple seeds before they generalise (cf. Li et al. 2025).
</content>
