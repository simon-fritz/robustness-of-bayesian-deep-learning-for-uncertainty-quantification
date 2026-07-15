# Results: Data-Efficiency Sweep (ResNet-18)

How do calibration and the OOD-detection signal change as training data shrinks?
We cap PneumoniaMNIST training to **100 / 1 000 / 10 000** examples (stratified
subsample under the seed; val/test are always the full splits) and train all
three methods — MAP, LLL, Deep Ensemble — on the **same ImageNet-pretrained
ResNet-18** as the full-data and long-tail experiments. 5 seeds (0–4) each;
values are mean ± std.

> **Architecture note.** Every method in this sweep uses ResNet-18. See
> `configs/experiment/training/pneumonia_{lll,map}_n*.yaml`.

Aggregates: [../results/data_efficiency_summary.csv](../results/data_efficiency_summary.csv);
plot: [../results/plots/auroc_vs_train_size.png](../results/plots/auroc_vs_train_size.png).

---

## 1. In-distribution accuracy vs. training size

| train size | MAP | LLL | Deep Ensemble |
|---|--:|--:|--:|
| 100    | 0.713 | 0.702 | 0.604 |
| 1 000  | 0.842 | 0.846 | 0.901 |
| 10 000 | 0.894 | 0.896 | 0.883 |

ID AUROC stays high throughout (0.90–0.98) because PneumoniaMNIST is an easy
binary task, so accuracy is the more discriminating ID axis at small N. All three
recover fast: 1 000 labelled images already put a pretrained ResNet-18 near its
full-data accuracy. At N=100 the ensemble's mean accuracy (0.60) is pulled down
by high seed variance (±0.20).

## 2. OOD detection vs. training size (AUROC)

**far-OOD (BloodMNIST):**

| train size | MAP (pred-entropy) | LLL (MI) | LLL (logit-var) | DE (MI) |
|---|--:|--:|--:|--:|
| 100    | 0.586 | 0.729 | 0.777 | 0.639 |
| 1 000  | 0.668 | 0.814 | 0.833 | 0.849 |
| 10 000 | 0.601 | 0.899 | **0.955** | 0.793 |

**near-OOD (OrganAMNIST):**

| train size | MAP (pred-entropy) | LLL (MI) | LLL (logit-var) | DE (MI) |
|---|--:|--:|--:|--:|
| 100    | 0.557 | 0.749 | 0.782 | 0.334 |
| 1 000  | 0.840 | 0.894 | 0.868 | 0.867 |
| 10 000 | 0.859 | 0.924 | 0.842 | 0.880 |

---

## 3. Findings

1. **For LLL, the Bayesian OOD signal *strengthens* with more data.** LLL's
   epistemic scores rise monotonically with N on far-OOD (MI 0.73 → 0.81 → 0.90;
   `logit_variance` 0.78 → 0.83 → 0.96). More ID data → tighter posterior → the
   residual epistemic uncertainty concentrates more cleanly on genuinely
   unfamiliar inputs. Last-layer Laplace needs a reasonable amount of data before
   its uncertainty is trustworthy.
2. **MAP has no usable far-OOD signal at any size.** MAP's first-order predictive
   entropy on far-OOD stays ≈0.6 regardless of N — a deterministic classifier's
   confidence does not separate BloodMNIST. Its *near*-OOD improves with N (0.56 →
   0.84 → 0.86) because near-OOD is partly separable by first-order confidence.
3. **The ensemble is strong but non-monotonic and unstable at N=100** (near-OOD
   MI 0.33 ± 0.24 — below random on some seeds). With only 100 images and shared
   initial weights, 5 members do not diversify enough for a reliable disagreement
   signal.
4. **No collapse here — that is the point.** Under simple data *scarcity* on a
   fixed distribution, the epistemic signal on genuine far-OOD gets *better* with
   data, not worse. The pathology the project documents (epistemic uncertainty ≠
   distribution membership) shows up under distribution/**density mismatch**
   (the long-tail experiment, where the score fires on the in-distribution rare
   class), not under scarcity alone. The two sweeps are complementary evidence.

## 4. Caveats

- **PneumoniaMNIST's train split is only ~4 700 images**, so `n=10000` is
  effectively the full set — the points are {tiny, small, ~full}, not a
  decade-wide sweep.
- **Binary, easy ID task**: ID AUROC saturates; accuracy is the meaningful ID axis.
- **Ensemble members share initial weights** (`copy.deepcopy`); only data ordering
  differs, so the ensemble's diversity — and its OOD signal — is likely
  understated, especially at N=100 (same open item as
  [results_deep_ensemble.md](results_deep_ensemble.md)).
- **5 seeds**; std is large at N=100 (small-sample regime).
