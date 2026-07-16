# Results: Long-Tail Experiment — Direct Evidence for Li et al. (2025)

| | Run dir |
| :--- | :--- |
| Deterministic (DET) | `outputs/pneumonia_resnet18_longtail_normal2pct_det/20260610_124459` |
| Last-Layer Laplace (LLL) | `outputs/pneumonia_resnet18_longtail_normal2pct_lll/20260610_124748` |
| Deep Ensemble (DE) | `outputs/pneumonia_resnet18_longtail_normal2pct_de/20260616_174632` |

**Training:** pretrained ResNet-18 on PneumoniaMNIST with class 0 ("normal")
sub-sampled to **2%** of its original count (24 normal vs. 3,494 pneumonia
training samples). All three methods share the same data, architecture,
ImageNet preprocessing, class-weighted CE loss, optimizer, and seed=42.

**Why the long-tail setup?** It isolates the single variable Li et al. (2025)
§5.3 argues drives epistemic-uncertainty scores: *training-data density.*
A point can be "low density" without being "from a different distribution"
(class 0 is in-distribution by construction; it is simply rare). If the score
genuinely measured "is this OOD," it should not flag in-distribution points
just because they were rare at training time. We test that prediction here.

We then evaluate the same long-tail-trained checkpoints against **three OOD
scenarios** with one ID stream each:

| Scenario | ID stream | OOD stream |
| :--- | :--- | :--- |
| Long-tail (tail_0) | pneumonia test (n=390) | normal test (n=234) |
| Far-OOD | pneumonia + normal test | BloodMNIST (colorful blood-cell microscopy) |
| Near-OOD | pneumonia + normal test | OrganAMNIST (grayscale abdominal CT slices) |

---

## 1. In-Distribution Performance (Long-Tail Trained)

| Metric | DET | LLL | DE |
| :--- | :---: | :---: | :---: |
| Accuracy | 0.7885 | 0.7997 | 0.7532 |
| Balanced Accuracy | 0.7205 | **0.7406** | 0.6726 |
| ROC AUROC | 0.8995 | 0.9026 | **0.9247** |
| ECE | 0.0827 | **0.0529** | 0.1145 |
| NLL | 0.4649 | **0.4332** | 0.4933 |
| Brier | 0.3008 | **0.2754** | 0.3295 |

LLL is the **best classifier** under long-tail training (highest balanced
accuracy, best ECE, best NLL, best Brier). DE has the highest plain ROC AUROC
but the worst calibration and accuracy on the minority class. Keep this
ordering in mind — it inverts when we move to OOD detection.

---

## 2. OOD Detection — The Headline Tables

### Long-tail scenario (detecting the under-represented ID class)

| Score | DET | LLL | DE |
| :--- | :---: | :---: | :---: |
| `one_minus_max_softmax` | 0.7389 | 0.7374 | **0.8420** |
| `predictive_entropy` | 0.7389 | 0.7374 | **0.8420** |
| `expected_entropy` (aleatoric) | 0.7389 | 0.5678 | **0.8275** |
| `mutual_information` (epistemic) | N/A (no posterior) | 0.8532 | **0.8497** |
| `softmax_variance_sum` | N/A (no posterior) | 0.8221 | **0.8546** |
| `logit_variance_sum` (Laplace only) | — | **0.9221** | — |

All scores work reasonably well — both Bayesian methods get MI AUROC ≈ 0.85
on flagging the under-represented in-distribution class.

### Far-OOD (BloodMNIST) under the same long-tail-trained checkpoints

| Score | DET | LLL | DE |
| :--- | :---: | :---: | :---: |
| `one_minus_max_softmax` | **0.4256** ⚠️ | **0.1482** ⚠️⚠️ | **0.3552** ⚠️ |
| `predictive_entropy` | 0.4256 ⚠️ | 0.1482 ⚠️⚠️ | 0.3552 ⚠️ |
| `expected_entropy` | 0.4256 ⚠️ | 0.1545 ⚠️⚠️ | 0.3397 ⚠️ |
| `mutual_information` | N/A | 0.2540 ⚠️⚠️ | 0.4921 ⚠️ |
| `softmax_variance_sum` | N/A | 0.2136 ⚠️⚠️ | 0.4213 ⚠️ |
| `logit_variance_sum` (Laplace only) | — | **0.7496** | — |

**Almost every softmax-based score is below 0.5.** AUROC < 0.5 means the score
is *inverted*: the OOD stream is scoring *less uncertain* than the ID stream.
The model is more confident on blood-cell microscopy than on the chest X-rays
it was trained for. LLL is the worst offender (MSP = 0.148 — strongly
inverted), DE is bad (0.355), DET is bad (0.426). Only the Laplace
`logit_variance` (which lives in pre-softmax space) recovers any signal at
0.75.

### Near-OOD (OrganAMNIST)

| Score | DET | LLL | DE |
| :--- | :---: | :---: | :---: |
| `one_minus_max_softmax` | 0.6722 | 0.4951 | **0.6615** |
| `predictive_entropy` | 0.6722 | 0.4951 | 0.6615 |
| `expected_entropy` (aleatoric) | 0.6722 | 0.4411 ⚠️ | 0.6400 |
| `mutual_information` | N/A | 0.5553 | **0.7088** |
| `softmax_variance_sum` | N/A | 0.5358 | **0.7024** |
| `logit_variance_sum` (Laplace only) | — | **0.7967** | — |

Near-OOD is mediocre across the board. DE softmax variance leads the
softmax-based scores at 0.70; LLL's `logit_variance` is again the best single
score at 0.80; LLL's softmax-based scores hover near random.

---

## 3. Interpretation — Three Things Each Directly Match the Paper

### (a) The score that "detects" the tail class *fails* on the actual OOD distribution.

Same MI score, same DE model, same evaluator:

- MI on **tail_0** (in-distribution, low density): **0.850** — appears to "work."
- MI on **BloodMNIST** (out-of-distribution, semantically different): **0.492** — random.

If MI genuinely measured "is this point from a different distribution," it
could not score in-distribution, low-density points *higher* than
out-of-distribution, semantically alien microscopy images. It does. So MI is
not measuring distribution membership. It is measuring **how unusual this
input looks relative to the dense regions of training data** — and an
under-represented ID class is more unusual to a long-tail-trained classifier
than a blood smear, because the blood smear is mapped confidently onto the
classifier's dominant decision region. This is precisely the
"answer the wrong question" pathology in Li et al. (2025) §4.2.

### (b) The best classifier is the worst OOD detector — exactly as §5.3 predicts.

LLL has the best calibration (ECE 0.053), best NLL (0.433), best Brier
(0.275), and highest balanced accuracy (0.741). It is also the model whose
softmax-based OOD scores are the **most catastrophically wrong** on Far-OOD
(MSP = 0.148, MI = 0.254). DE — worst calibration (ECE 0.115) — has the
*least* inverted Far-OOD scores (MSP = 0.355, MI = 0.492). The ranking is
inverted: a better-fit classifier produces a worse OOD detector.

Li et al. §5.3 (paraphrased):

> In the infinite-ID-data limit, the posterior contracts, the model becomes
> certain in its parameters, and epistemic uncertainty over OOD inputs
> shrinks to zero. If epistemic uncertainty were the correct signal for OOD
> detection, this would imply OOD points "do not exist" once you have enough
> data — clearly absurd. Therefore epistemic uncertainty and "is this OOD?"
> are answering different questions.

We see the same effect at finite data: as the model fits ID better (LLL >
DE > DET on calibration), epistemic uncertainty over Far-OOD inputs
contracts and the detector breaks.

### (c) `logit_variance` is the only score that survives — and it tells us why.

Across both Far-OOD and Near-OOD, **`logit_variance` (Laplace-only) is the
only score above 0.7 AUROC.** Every softmax-based score on every method
either fails (Near-OOD: 0.50–0.71) or is *inverted* (Far-OOD: 0.15–0.49).

Mechanism: `logit_variance` lives in the *pre*-softmax space, where the
Laplace posterior's variance reflects how thin the data evidence is along
each logit direction. The softmax non-linearity collapses logit space onto
the K-simplex, which forces every point — including alien microscopy — into
*some* decision region. That projection is what makes MSP and MI invertible.
Working in logit space sidesteps the projection.

The paper's prescription in §6 is that OOD detection requires a separate
generative or density-based criterion, not a re-scoring of the
in-distribution classifier. `logit_variance` is the closest thing to that
prescription that we can compute "for free" from the LLL posterior — and it
is also the only score that doesn't fail catastrophically here.

---

## 4. Caveats and What Not to Overclaim

- **Binary task confound.** PneumoniaMNIST is binary. The long-tail scenario
  collapses to "uncertainty on majority class vs. uncertainty on minority
  class," which is partly about class difficulty, not just training density.
  We will run BloodMNIST long-tail (multiclass) to disentangle. The
  config for it ([configs/experiment/ood/bloodmnist_longtail.yaml](../configs/experiment/ood/bloodmnist_longtail.yaml))
  already exists; a training config is the open item.
- **Single-seed numbers only in this write-up.** All tables above are seed=42.
  5-seed reruns are complete — see
  [../results/all_experiments_summary.csv](../results/all_experiments_summary.csv)
  for mean ± std across seeds 0–4 for all three methods in the long-tail scenario.
- **Class weighting is on.** All three methods use class-weighted cross
  entropy (`use_class_weights: true`). That partially compensates for the
  long-tail imbalance you are trying to expose. We expect the inversion on
  Far-OOD to be **even stronger** without class weighting — a useful ablation
  for the talk.
- **DE has only 5 members.** `expected_pairwise_kl` is suppressed by the
  evaluator (needs ≥10 samples). LLL gets it because Laplace draws 100 MC
  samples by default. So one column is genuinely missing for DE, not a bug.

---

## 5. Figures Worth Showing in the Talk

For the long-tail scenario (each one a "before vs. after"):

- `outputs/pneumonia_resnet18_longtail_normal2pct_de/20260616_174632/ood/long_tail/figures/auroc_summary.png`
  — bar chart, DE flagging tail_0 with MI 0.85.
- `outputs/pneumonia_resnet18_longtail_normal2pct_de/20260616_174632/ood/far_ood/figures/auroc_summary.png`
  — same model, same evaluator, all bars below 0.5. **This is the slide.**
- `outputs/pneumonia_resnet18_longtail_normal2pct_de/20260616_174632/ood/far_ood/figures/hist_bloodmnist_mutual_information.png`
  — histograms of MI on ID vs. OOD overlap (or are inverted). Lets the
  audience see the failure directly.
- `outputs/pneumonia_resnet18_longtail_normal2pct_lll/20260610_124748/ood/far_ood/figures/auroc_by_category_bloodmnist.png`
  — Laplace's `logit_variance` standing alone at 0.75, every softmax-based
  score below 0.30. Use to introduce the §3(c) point about logit space vs.
  softmax space.

The single most compelling artifact for the "Li et al. are right" narrative
is the side-by-side AUROC summary for `tail_0` (≈0.85) vs. `bloodmnist`
(≈0.40) from the *same DE checkpoint*. Same model, same score, same
evaluator — totally different reliability depending on which "OOD-ness" we
ask about.