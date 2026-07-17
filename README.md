# Robustness of Bayesian Deep Learning for Uncertainty Quantification

OOD detection with Bayesian Neural Networks on MedMNIST (PneumoniaMNIST).

Based on Li et al. (2025). Experiments: Last-Layer Laplace (LLL), First-Layer
Laplace (FLL), MAP (deterministic), and Deep Ensemble on PneumoniaMNIST, evaluated on in-distribution accuracy and
OOD detection (far-OOD: BloodMNIST, near-OOD: OrganAMNIST) across three scenarios:
full balanced data, long-tail class imbalance, and varying training set size.

---

## Install

```bash
pip install -e .
```

Python 3.10+ required. For the test suite, install the dev extras and run pytest:

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_first_layer_laplace.py` runs offline (synthetic tensors);
`tests/test_medmnist_loader_filters.py` downloads BloodMNIST on first run.

---

## Data

MedMNIST datasets are downloaded automatically by the `medmnist` package on first
use. No manual download needed. Default landing path: `./data/`.

Override with an env variable:
```bash
export DATA_ROOT=/some/shared/path
```

---

## Methods

| Method | Architecture | Key idea |
|---|---|---|
| **LLL** (Last-Layer Laplace) | ResNet-18 | MAP train, Gaussian posterior over last-layer weights via `laplace-torch` |
| **FLL** (First-Layer Laplace) | ResNet-18 | MAP train, Gaussian posterior over the stem conv (`conv1`) via `laplace-torch`, GLM predictive — mirror of LLL to probe *where* the Bayesian treatment matters (single-seed, exploratory) |
| **MAP** (deterministic) | ResNet-18 | Standard cross-entropy, no uncertainty |
| **Deep Ensemble** | ResNet-18 × 5 | 5 independently trained models, uncertainty from disagreement |

---

## Experiment scenarios

| Scenario | Description | OOD evaluations |
|---|---|---|
| **full_data** | Balanced full training set | far-OOD (BloodMNIST), near-OOD (OrganAMNIST) |
| **longtail** | 2% of "normal" class kept (class_subsampling) | tail-class-as-OOD + far-OOD + near-OOD |
| **data_eff** | Training capped to 100 / 1000 / 10 000 examples | far-OOD + near-OOD |

---

## Results

Headline across the experiments: epistemic uncertainty tracks *training-data
density*, not *distribution membership* — the central claim of Li et al. (2025).
5-seed aggregates live in
[results/all_experiments_summary.csv](results/all_experiments_summary.csv);
per-topic write-ups:

- [docs/results_resnet18_baseline_vs_laplace.md](docs/results_resnet18_baseline_vs_laplace.md) — full-data MAP vs. LLL: Laplace calibrates ~3× better at no accuracy cost; the best OOD score flips between far- and near-OOD.
- [docs/results_long_tail.md](docs/results_long_tail.md) — the headline experiment: an epistemic "OOD detector" fires on the *in-distribution* rare class (MI ≈ 0.85) yet inverts on true far-OOD.
- [docs/results_deep_ensemble.md](docs/results_deep_ensemble.md) — Deep Ensemble ID/OOD + a stability anomaly (better classifier → worse far-OOD detector).
- [docs/results_data_efficiency.md](docs/results_data_efficiency.md) — how calibration and the OOD signal change with 100 / 1 000 / 10 000 training images (all ResNet-18).
- [docs/approach_first_layer_laplace.md](docs/approach_first_layer_laplace.md) — First-Layer Laplace (`conv1`) vs. last layer: where in the network the Bayesian treatment matters *(single-seed, exploratory)*.

Background: [docs/architecture_choice.md](docs/architecture_choice.md), [docs/evaluation_overview.md](docs/evaluation_overview.md).

---

## Running experiments

> The `slurm/*.sbatch` headers (`--partition=universe,asteroids`, `--qos`,
> `ml python/anaconda3`) target the TUM CIT cluster — adjust them for your
> scheduler. No cluster? Run any config without SLURM via
> [Running locally](#running-locally-no-slurm).

### Full-data + long-tail (5 seeds each → 30 jobs)

```bash
bash slurm/submit_all_experiments.sh
```

Submits 30 jobs: 3 methods × 2 scenarios (full_data + longtail) × 5 seeds.
After all complete:
```bash
python scripts/aggregate_all.py --seeds 0 1 2 3 4
```

Produces `results/all_experiments_summary.csv` and `results/all_experiments_raw.csv`.

### Data-efficiency sweep (5 seeds each → 45 jobs)

```bash
bash slurm/submit_data_efficiency_sweep.sh
```

Submits 45 jobs: 3 methods × 3 train sizes × 5 seeds.
After all complete:
```bash
python scripts/aggregate_data_efficiency.py --seeds 0 1 2 3 4
```

Produces `results/data_efficiency_summary.csv`, `results/data_efficiency_raw.csv`,
and plots under `results/plots/`.

### Run everything at once

```bash
bash slurm/submit_all_experiments.sh
bash slurm/submit_data_efficiency_sweep.sh
# monitor
squeue -u $USER
# after all 75 jobs complete
python scripts/aggregate_all.py --seeds 0 1 2 3 4
python scripts/aggregate_data_efficiency.py --seeds 0 1 2 3 4
```

### First-Layer Laplace (FLL) — single-seed exploratory

FLL is not in the 30-job sweep (single-seed, exploratory). Run separately:

```bash
# full-data FLL (seed=42)
sbatch slurm/resnet18_fll.sbatch
# long-tail FLL (seed=42)
sbatch slurm/train_longtail_generic.sbatch configs/experiment/pneumonia_resnet18_longtail_fll.yaml
```

Results documented in [docs/approach_first_layer_laplace.md](docs/approach_first_layer_laplace.md).

### Submit a single job manually

```bash
# Generic: takes <config> <seed>
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/pneumonia_resnet18_lll.yaml 0
sbatch slurm/train_longtail_generic.sbatch    configs/experiment/pneumonia_resnet18_longtail_lll.yaml 0
```

---

## Configs

**Full-data balanced:**

| Config | Method |
|---|---|
| `configs/experiment/pneumonia_resnet18_lll.yaml` | LLL |
| `configs/experiment/pneumonia_resnet18_fll.yaml` | FLL |
| `configs/experiment/pneumonia_resnet18_baseline.yaml` | MAP |
| `configs/experiment/training/pneumonia_resnet18_deep_ensemble.yaml` | Ensemble |

**Long-tail (2% normal class):**

| Config | Method |
|---|---|
| `configs/experiment/pneumonia_resnet18_longtail_lll.yaml` | LLL |
| `configs/experiment/pneumonia_resnet18_longtail_fll.yaml` | FLL |
| `configs/experiment/pneumonia_resnet18_longtail_baseline.yaml` | MAP |
| `configs/experiment/pneumonia_resnet18_longtail_deep_ensemble.yaml` | Ensemble |

**Data-efficiency sweep:**

| Config | Method | Train size |
|---|---|---|
| `configs/experiment/training/pneumonia_lll_n100/1000/10000.yaml` | LLL | 100 / 1000 / 10000 |
| `configs/experiment/training/pneumonia_map_n100/1000/10000.yaml` | MAP | 100 / 1000 / 10000 |
| `configs/experiment/training/pneumonia_ensemble_n100/1000/10000.yaml` | Ensemble | 100 / 1000 / 10000 |

Training-size subsampling is stratified (preserves class proportions). Val and test
sets are always the full MedMNIST splits.

---

## Paper reproduction (Li et al. 2025)

The `feat/reproduction` branch contains a full reproduction of the original paper's
experiments (CIFAR-10, `reproduction/` folder). It is kept separate from main to
avoid mixing the paper's workflow with this project's MedMNIST experiments.

```bash
git checkout feat/reproduction
# see reproduction/README or the branch README for run instructions
```

---

## Reproducing results exactly

All randomness is controlled by the `seed` field in each config.
`set_seed()` seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA) and enables
cuDNN deterministic mode. Data subsampling uses a derived RNG from the same seed.
The actual seed used is saved to `outputs/<run>/<timestamp>/config.yaml`.

Both submission scripts use seeds 0–4. Running the same config + seed on the same
hardware gives identical results.

---

## Outputs

```
outputs/<run_name>/<timestamp>/
  config.yaml           — full config used (including actual seed)
  test_metrics.json     — in-distribution accuracy, balanced accuracy, AUROC, ECE, NLL, Brier
  checkpoint_path.txt   — path to best checkpoint
  sigma_summary.json    — LLL / FLL only: mean/max diag(Σ), ‖Σ‖_F
  ood/
    far_ood/ood_metrics.json    — AUROC per score (BloodMNIST)
    near_ood/ood_metrics.json   — AUROC per score (OrganAMNIST)
    long_tail/ood_metrics.json  — AUROC per score (longtail only)

results/
  all_experiments_raw.csv        — one row per run (all scenarios)
  all_experiments_summary.csv    — mean ± std per (scenario, method)
  data_efficiency_raw.csv        — one row per run (sweep only)
  data_efficiency_summary.csv    — mean ± std per (method, train_size)
  plots/
    auroc_vs_train_size.png
    sigma_vs_train_size.png
```

Checkpoints: `checkpoints/<run_name>/<timestamp>/best.pt`

---

## Running locally (no SLURM)

```bash
python scripts/train.py --config configs/experiment/pneumonia_resnet18_lll.yaml --seed 0

RUN_DIR=outputs/pneumonia_resnet18_lll/<timestamp>
python scripts/evaluate.py --run-dir "$RUN_DIR"
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
```

---

## Repository layout

```
configs/
  experiment/                    One YAML per experiment
    training/                    Data-efficiency sweep configs
    ood/                         OOD evaluation scenarios
  data/ model/ method/           Reusable config blocks
src/bnn_medmnist/
  data/                          MedMNIST loader (subsampling, filtering)
  models/                        SmallCNN, ResNet-18 builder
  methods/                       LLL, MAP (deterministic), Deep Ensemble
  training/                      Trainer (early stopping on val AUROC)
  evaluation/                    Metrics, uncertainty scores, OOD scoring
  utils/                         Config loading, seeding, logging
scripts/
  train.py                       Training entry point (--config, --seed)
  evaluate.py                    In-distribution evaluation
  evaluate_ood.py                OOD evaluation
  aggregate_all.py               Aggregate ALL experiments (mean ± std)
  aggregate_data_efficiency.py   Aggregate data-efficiency sweep only
slurm/
  train_lll_data_efficiency.sbatch   Generic job: full-data + sweep (config + seed args)
  train_longtail_generic.sbatch      Generic job: longtail (config + seed args)
  submit_all_experiments.sh          Submit full-data + longtail × 5 seeds (30 jobs)
  submit_data_efficiency_sweep.sh    Submit data-efficiency sweep × 5 seeds (45 jobs)
  resnet18_lll.sbatch                Legacy single-run scripts (seed=42)
  resnet18_deep_ensemble.sbatch
  longtail_lll.sbatch
  longtail_det.sbatch
  longtail_deep_ensemble.sbatch
results/                         Aggregated CSVs and plots (committed after runs)
data/                            Raw datasets (gitignored, auto-downloaded)
checkpoints/                     Model weights (gitignored)
logs/                            SLURM + TensorBoard logs (gitignored)
outputs/                         Per-run artifacts (config, metrics, OOD JSON)
```
