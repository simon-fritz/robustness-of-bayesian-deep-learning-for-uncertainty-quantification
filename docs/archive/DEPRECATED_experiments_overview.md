# Experiments Overview

OOD detection with Bayesian Neural Networks on MedMNIST, framed around the critique in Li et al. (2025): how well do BNN uncertainty decompositions actually separate in-distribution from out-of-distribution inputs across far-OOD, near-OOD, long-tail, and held-out scenarios?

- Generated: 2026-05-27
- Git commit: `79324418a33a1fe0a36958609a149b20ef137430`

---

## 1. Datasets

Datasets referenced by configs under [configs/data/](../configs/data/) and the OOD scenarios in [configs/experiment/ood/](../configs/experiment/ood/). Split sizes are from `medmnist.INFO`.

| Dataset         | Task          | Classes | Channels | Train  | Val    | Test   | Role in this project                                  |
|-----------------|---------------|---------|----------|--------|--------|--------|-------------------------------------------------------|
| PneumoniaMNIST  | binary-class  | 2       | 1        | 4 708  | 524    | 624    | ID for first experiment; class-imbalanced (normal ≪ pneumonia) |
| BloodMNIST      | multi-class   | 8       | 3        | 11 959 | 1 712  | 3 421  | Far-OOD vs PneumoniaMNIST; planned ID for within-dataset scenarios |
| OrganAMNIST     | multi-class   | 11      | 1        | 34 561 | 6 491  | 17 778 | Near-OOD vs PneumoniaMNIST (same grayscale medical imaging) |
| PathMNIST       | multi-class   | 9       | 3        | 89 996 | 10 004 | 7 180  | Listed in [pneumonia_lll.yaml](../configs/experiment/training/pneumonia_lll.yaml) `ood_datasets`; no OOD eval config or run exists yet |

PneumoniaMNIST class balance: the per-class counts are not cached in the repo; `medmnist.INFO["pneumoniamnist"]` only stores totals. The training config [pneumonia_longtail_lll.yaml](../configs/experiment/training/pneumonia_longtail_lll.yaml) calls class 0 ("normal") the under-represented class, consistent with the known PneumoniaMNIST imbalance.

Channel handling for far-OOD (RGB → grayscale or 1 → 3 replication) is implemented in [src/bnn_medmnist/data/ood_pairs.py](../src/bnn_medmnist/data/ood_pairs.py) and is flagged in-code as a known confound.

---

## 2. Models

Single architecture, [src/bnn_medmnist/models/small_cnn.py](../src/bnn_medmnist/models/small_cnn.py):

| Submodule | Composition                                  | Output      |
|-----------|----------------------------------------------|-------------|
| `layer1`  | Conv(→32) – BN – ReLU – MaxPool              | 14×14       |
| `layer2`  | Conv(→64) – BN – ReLU – MaxPool              | 7×7         |
| `layer3`  | Conv(→128) – BN – ReLU                       | 7×7         |
| `layer4`  | AdaptiveAvgPool(1) – Flatten – Dropout       | feature vec |
| `fc`      | Linear (Bayesian target for last-layer Laplace) | logits   |

Defaults from [configs/model/small_cnn.yaml](../configs/model/small_cnn.yaml): `in_channels=1`, `num_classes=2`, `dropout=0.0`. Channel/class counts are overridden per-experiment by the data config.

---

## 3. Methods

| Method               | Description                                                                 | Bayesian extent | Status                              | Config |
|----------------------|-----------------------------------------------------------------------------|-----------------|-------------------------------------|--------|
| Deterministic        | Standard MAP training, single softmax forward pass                          | none (`[]`)     | implemented & run                   | [deterministic.yaml](../configs/method/deterministic.yaml) |
| Last-Layer Laplace   | MAP training, then `laplace-torch` Laplace fit on `fc` with full Hessian and `marglik` prior tuning; 100 MC samples at predict time | last layer (`["fc"]`) | implemented & run | [last_layer_laplace.yaml](../configs/method/last_layer_laplace.yaml) |

`BayesianMethod.__init__` ([src/bnn_medmnist/methods/base.py](../src/bnn_medmnist/methods/base.py)) documents the extent axis (`["fc"]` / `["layer4","fc"]` / `"all"` / `[]`), but `_resolve_bayesian_params` is still a `# TODO: implement` stub — it is only relevant for methods that need a generic layer-set resolver, neither of the two implemented methods uses it.

Planned / referenced but absent from the repo:
- MC-Dropout
- Deep Ensembles
- Variational Inference / mean-field BNN
- Multi-layer or full-network Laplace (the config field exists but no method config uses it)
- Oracle baseline from Li et al. (2025)

---

## 4. Uncertainty measures

Defined in [src/bnn_medmnist/evaluation/uncertainty.py](../src/bnn_medmnist/evaluation/uncertainty.py) and registered as OOD scores in [src/bnn_medmnist/evaluation/ood.py](../src/bnn_medmnist/evaluation/ood.py) (`SCORE_FNS`).

For each input, the BNN gives us not one prediction but `S` predictions (one per posterior sample). Each score below collapses those `S` predictions into a single "how unsure am I about this input?" number — higher means more unsure / more OOD-like.

| Score                  | Formula                                              | In plain words |
|------------------------|------------------------------------------------------|----------------|
| `predictive_entropy`   | entropy of the *average* predicted distribution      | "After averaging all my guesses, how spread out is the result?" Captures **total** uncertainty: any reason to be unsure shows up here. |
| `expected_entropy`     | average entropy of *each* predicted distribution     | "On a typical sample from my posterior, how spread out is the prediction?" Captures **aleatoric** uncertainty — noise that even a perfect model couldn't remove (e.g. ambiguous image). |
| `mutual_information`   | `predictive_entropy − expected_entropy` (BALD)       | "How much do my posterior samples *disagree* with each other?" Captures **epistemic** uncertainty — the model itself is unsure because it hasn't seen enough data like this. Zero for deterministic models. |
| `one_minus_max_softmax`| `1 − max_class mean_sample p(class)`                 | "1 minus the confidence of my top prediction." Classic baseline score — high when the model is not strongly committed to any class. |

Decomposition follows Depeweg et al. 2018 / Kwon et al. 2020. For OOD detection the theoretical hope is that **epistemic** (mutual information) lights up on unfamiliar inputs while **aleatoric** stays flat — Li et al. (2025) argue this rarely holds cleanly in practice.

---

## 5. Evaluation setup

ID metrics ([src/bnn_medmnist/evaluation/metrics.py](../src/bnn_medmnist/evaluation/metrics.py)):

| Metric | In plain words |
|--------|----------------|
| `accuracy` | Fraction of test images the model labels correctly. |
| `balanced_accuracy` | Same as accuracy, but each class counts equally — so a model can't get a high score by just predicting the majority class. Important on imbalanced datasets like PneumoniaMNIST. |
| `auroc` | "If I pick one positive and one negative image at random, how often does the model give the positive one a higher score?" 1.0 = perfect, 0.5 = random. |
| `expected_calibration_error` (ECE, 15 bins) | "When the model says it's 80% sure, is it actually right 80% of the time?" Measures the gap between confidence and accuracy. Lower is better. |
| `nll` | Negative log-likelihood — penalises both wrong predictions and overconfident wrong predictions. Lower is better. *Implemented but not written by the current eval script.* |
| `brier_score` | Squared error between predicted probabilities and one-hot true labels. Lower is better. *Implemented but not written.* |

OOD detection metrics ([src/bnn_medmnist/evaluation/ood.py](../src/bnn_medmnist/evaluation/ood.py)) — OOD is the positive class, higher uncertainty should mean "more OOD":

| Metric | In plain words |
|--------|----------------|
| `AUROC` | "If I pick one ID image and one OOD image at random, how often is the OOD one ranked as more uncertain?" 1.0 = perfectly separable, 0.5 = no better than chance. |
| `AUPRC` | Area under the precision–recall curve. Like AUROC but more informative when the OOD set is much smaller (or larger) than the ID set. |
| `FPR@95%TPR` | "If I set my OOD threshold so I catch 95% of the real OOD images, what fraction of ID images do I mislabel as OOD?" Lower is better — this is the practitioner-relevant cost of false alarms. |

OOD scenarios ([src/bnn_medmnist/data/ood_pairs.py](../src/bnn_medmnist/data/ood_pairs.py)):

| Scenario          | ID              | OOD source                                | Config(s) |
|-------------------|-----------------|-------------------------------------------|-----------|
| `far_ood`         | PneumoniaMNIST  | BloodMNIST (RGB → grayscale)              | [pneumonia_far_blood.yaml](../configs/experiment/ood/pneumonia_far_blood.yaml) |
| `near_ood`        | PneumoniaMNIST  | OrganAMNIST                               | [pneumonia_near_organ.yaml](../configs/experiment/ood/pneumonia_near_organ.yaml) |
| `long_tail`       | PneumoniaMNIST  | Within-dataset: class 0 ("normal") subsampled to 2% in training, treated as tail at test | [pneumonia_longtail.yaml](../configs/experiment/ood/pneumonia_longtail.yaml) |
| `long_tail`       | BloodMNIST      | Within-dataset: class 7 ("platelet") subsampled to 2% | [bloodmnist_longtail.yaml](../configs/experiment/ood/bloodmnist_longtail.yaml) |
| `held_out_class`  | BloodMNIST      | Class 7 excluded from training, treated as OOD at test | [bloodmnist_heldout.yaml](../configs/experiment/ood/bloodmnist_heldout.yaml) |

The PneumoniaMNIST long-tail config explicitly warns that, because the dataset is binary, the "ID-vs-OOD" comparison there is class 1 vs class 0 and therefore confounds training imbalance with class difficulty — should be cross-checked against the BloodMNIST long-tail.

---

## 6. Results so far

Four run folders found under [outputs/](../outputs/).

### 6.1 ID test metrics

| Run | Experiment | Method | ID dataset | Accuracy | Balanced acc. | AUROC | ECE |
|-----|------------|--------|------------|----------|---------------|-------|-----|
| [`pneumonia_baseline/20260517_172109`](../outputs/pneumonia_baseline/20260517_172109/test_metrics.json) | pneumonia_baseline | Deterministic | PneumoniaMNIST | 0.8830 | 0.8534 | 0.9655 | 0.0412 |
| [`pneumonia_lll/20260517_173207`](../outputs/pneumonia_lll/20260517_173207/test_metrics.json) | pneumonia_lll | Last-Layer Laplace | PneumoniaMNIST | 0.8798 | 0.8517 | 0.9662 | 0.0313 |
| [`pneumonia_longtail_normal2pct_lll/20260520_151709`](../outputs/pneumonia_longtail_normal2pct_lll/20260520_151709/test_metrics.json) | pneumonia_longtail_normal2pct_lll | Last-Layer Laplace | PneumoniaMNIST (class 0 → 2% at train) | 0.8317 | 0.7884 | 0.9278 | 0.0373 |
| [`pneumonia_longtail_normal2pct_det/20260527_114851`](../outputs/pneumonia_longtail_normal2pct_det/20260527_114851/test_metrics.json) | pneumonia_longtail_normal2pct_det | Deterministic | PneumoniaMNIST (class 0 → 2% at train) | 0.8269 | 0.7821 | 0.9204 | 0.0709 |

Last-Layer Laplace matches the deterministic baseline on accuracy/AUROC and lowers ECE on the full PneumoniaMNIST. Both long-tail-trained runs drop on every ID metric as expected (the model has barely seen class 0); the deterministic long-tail run has the worst ECE by a clear margin (0.0709 vs 0.0373 for LLL on the same data split), consistent with Laplace's regularising effect on calibration.

### 6.2 OOD detection

PneumoniaMNIST as ID. Numbers are from each scenario's `ood_metrics.json`.

#### Far-OOD: PneumoniaMNIST → BloodMNIST

| Method | Score | AUROC | AUPRC | FPR@95 |
|--------|-------|-------|-------|--------|
| Deterministic | predictive_entropy   | 0.4716 | 0.8299 | 0.9439 |
| Deterministic | mutual_information   | 0.5000 | 0.8457 | 1.0000 |
| Deterministic | expected_entropy     | 0.4716 | 0.8299 | 0.9439 |
| Deterministic | one_minus_max_softmax| 0.4716 | 0.8299 | 0.9439 |
| LLL          | predictive_entropy   | 0.6206 | 0.8617 | 0.6266 |
| LLL          | mutual_information   | 0.7312 | 0.9107 | 0.5321 |
| LLL          | expected_entropy     | 0.5354 | 0.8316 | 0.7580 |
| LLL          | one_minus_max_softmax| 0.6206 | 0.8617 | 0.6266 |

The deterministic model produces a single sample, so `mutual_information = 0` for every example and AUROC collapses to 0.5; its other scores are barely above chance on this pair. LLL is clearly better but far-OOD performance (BloodMNIST → grayscale) is weaker than usually reported, likely affected by the channel-adaptation confound noted in `ood_pairs.py`.

#### Near-OOD: PneumoniaMNIST → OrganAMNIST

| Method | Score | AUROC | AUPRC | FPR@95 |
|--------|-------|-------|-------|--------|
| Deterministic | predictive_entropy   | 0.5837 | 0.9738 | 0.9631 |
| Deterministic | mutual_information   | 0.5000 | 0.9661 | 1.0000 |
| Deterministic | expected_entropy     | 0.5837 | 0.9738 | 0.9631 |
| Deterministic | one_minus_max_softmax| 0.5837 | 0.9738 | 0.9631 |
| LLL          | predictive_entropy   | 0.7505 | 0.9837 | 0.5705 |
| LLL          | mutual_information   | 0.8912 | 0.9953 | 0.4760 |
| LLL          | expected_entropy     | 0.5975 | 0.9643 | 0.7035 |
| LLL          | one_minus_max_softmax| 0.7505 | 0.9837 | 0.5705 |

For LLL, mutual information (epistemic) is the strongest score on near-OOD — consistent with the theoretical claim that epistemic uncertainty should rise on novel inputs.

#### Long-tail: PneumoniaMNIST (class 0 = normal, 2% kept) → class 0 at test

| Method | Score | AUROC | AUPRC | FPR@95 |
|--------|-------|-------|-------|--------|
| LLL (longtail) | predictive_entropy   | 0.8777 | 0.7446 | 0.3897 |
| LLL (longtail) | mutual_information   | 0.9166 | 0.8212 | 0.2872 |
| LLL (longtail) | expected_entropy     | 0.7225 | 0.5196 | 0.6333 |
| LLL (longtail) | one_minus_max_softmax| 0.8777 | 0.7446 | 0.3897 |

Mutual information again dominates. As warned in the OOD config, this is binary PneumoniaMNIST so the "OOD" set is exactly class 0 — the result conflates "tail class" with "harder class" and should be cross-checked against a BloodMNIST long-tail run (not yet available).

#### Long-tail: PneumoniaMNIST (deterministic, class 0 → 2% at train) → class 0 at test

| Method | Score | AUROC | AUPRC | FPR@95 |
|--------|-------|-------|-------|--------|
| Deterministic (longtail) | predictive_entropy   | 0.7858 | 0.6720 | 0.8231 |
| Deterministic (longtail) | mutual_information   | 0.5000 | 0.3750 | 1.0000 |
| Deterministic (longtail) | expected_entropy     | 0.7858 | 0.6720 | 0.8231 |
| Deterministic (longtail) | one_minus_max_softmax| 0.7858 | 0.6720 | 0.8231 |

As before, deterministic mutual information collapses to chance (single sample → zero epistemic). The remaining three scores all detect the tail class above chance (AUROC ≈ 0.79) but clearly underperform the LLL long-tail run on every score (LLL `predictive_entropy` 0.878, `mutual_information` 0.917) — the gap is the contribution of the last-layer posterior. FPR@95 is also markedly worse (0.82 vs 0.39).

#### BloodMNIST long-tail and held-out

— (not yet run) — OOD configs exist but no matching training run (`bloodmnist_longtail*`, `bloodmnist_heldout*`) has been performed.

#### PneumoniaMNIST → PathMNIST

— (not yet run) — listed in `ood_datasets:` of the LLL training config, but no OOD scenario config or output exists.

---

## 7. Open items / next steps

- **Methods**: add MC-Dropout, Deep Ensembles, and at least one variational / full-Laplace variant to actually exercise the `bayesian_layers` axis beyond `["fc"]`. Implement `BayesianMethod._resolve_bayesian_params` once a method needs it.
- **Oracle baseline** (Li et al. 2025) is the key reference for the critique and is currently **not implemented**.
- **Scenarios still to run**:
  - BloodMNIST training runs and corresponding `long_tail` and `held_out_class` OOD evaluations.
  - PneumoniaMNIST → PathMNIST OOD (no config yet).
- **Multi-seed runs**: every run folder uses `seed: 42`; results are single-seed, so no confidence intervals are available. Add ≥3 seeds before drawing comparative conclusions.
- **Persisted ID metrics**: NLL and Brier are implemented but not written into `test_metrics.json` — consider adding to make calibration comparisons richer.
- **Confound bookkeeping**: document the RGB→grayscale far-OOD adaptation and the binary-class long-tail caveat in any results we share externally.

---

## 8. References

- Li, J. et al. (2025). *Critique of Bayesian uncertainty for OOD detection.* (Project's motivating paper; full citation TBC.)
- Hüllermeier, E. & Waegeman, W. (2021). *Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods.* Machine Learning, 110.
- Jospin, L. V. et al. (2022). *Hands-on Bayesian neural networks — a tutorial for deep learning users.* IEEE Computational Intelligence Magazine.
- Yang, J. et al. (2023). *MedMNIST v2 — a large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 10.

---

## Provenance

Generated by inspecting:

- Configs: all files under [configs/](../configs/) (data, model, method, experiment/training, experiment/ood).
- Source: [src/bnn_medmnist/methods/](../src/bnn_medmnist/methods/) (`base.py`, `deterministic.py`, `last_layer_laplace.py`), [src/bnn_medmnist/models/small_cnn.py](../src/bnn_medmnist/models/small_cnn.py), [src/bnn_medmnist/evaluation/](../src/bnn_medmnist/evaluation/) (`uncertainty.py`, `ood.py`, `metrics.py`), [src/bnn_medmnist/data/](../src/bnn_medmnist/data/) (`ood_pairs.py`, `medmnist_loader.py`).
- Run outputs: every `test_metrics.json`, `ood_metrics.json`, and `run_info.txt` under [outputs/](../outputs/) (4 run folders: `pneumonia_baseline/20260517_172109`, `pneumonia_lll/20260517_173207`, `pneumonia_longtail_normal2pct_lll/20260520_151709`, `pneumonia_longtail_normal2pct_det/20260527_114851`).
- Dataset metadata: `medmnist.INFO` for split sizes and label maps.
- Git commit: `git rev-parse HEAD` at generation time.
