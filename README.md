# Robustness of Bayesian Deep Learning for Uncertainty Quantification

OOD detection with Bayesian Neural Networks on MedMNIST (PneumoniaMNIST).

Based on Li et al. (2025). Experiments: Last-Layer Laplace (LLL), MAP (deterministic),
and Deep Ensemble on PneumoniaMNIST, evaluated on in-distribution accuracy and
OOD detection (far-OOD: BloodMNIST, near-OOD: OrganAMNIST).

---

## Install

```bash
pip install -e .
```

Python 3.10+ required.

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

| Method | Config prefix | Architecture | Key idea |
|---|---|---|---|
| **LLL** (Last-Layer Laplace) | `pneumonia_lll*` | SmallCNN / ResNet-18 | MAP train, Gaussian posterior over last-layer weights via `laplace-torch` |
| **MAP** (deterministic) | `pneumonia_map*` | SmallCNN | Standard cross-entropy training, no uncertainty |
| **Deep Ensemble** | `pneumonia_ensemble*` | ResNet-18 × 5 | 5 independently trained models, uncertainty from prediction disagreement |

---

## Running experiments

### Full-data runs (baseline, on SLURM)

```bash
sbatch slurm/resnet18_lll.sbatch          # LLL with ResNet-18
sbatch slurm/train_baseline.sbatch        # MAP with SmallCNN
sbatch slurm/resnet18_deep_ensemble.sbatch # Deep Ensemble (5 × ResNet-18)
```

Each job trains and then runs in-distribution + far-OOD + near-OOD evaluation.
Results land in `outputs/<experiment_name>/<timestamp>/`.

### Data-efficiency sweep (Section 5.3)

Trains each method on 100 / 1000 / 10 000 examples, repeated across 5 seeds.

**Submit all 45 jobs at once:**
```bash
bash slurm/submit_data_efficiency_sweep.sh
```

Or submit individual runs manually:
```bash
# sbatch slurm/train_lll_data_efficiency.sbatch <config> <seed>
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n100.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n1000.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n10000.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_map_n100.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_map_n1000.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_map_n10000.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_ensemble_n100.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_ensemble_n1000.yaml 0
sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_ensemble_n10000.yaml 0
```

**Monitor jobs:**
```bash
squeue -u $USER
```

**Aggregate results after all jobs complete:**
```bash
python scripts/aggregate_data_efficiency.py
```

Produces:
- `results/data_efficiency_raw.csv` — one row per run (method, n, seed)
- `results/data_efficiency_summary.csv` — mean ± std per (method, n) across seeds
- `results/plots/auroc_vs_train_size.png`
- `results/plots/sigma_vs_train_size.png`

---

## Reproducing results exactly

All randomness is controlled by the `seed` field in each config (default `42`).
`set_seed()` seeds Python's `random`, NumPy, and PyTorch (CPU + CUDA) plus enables
cuDNN deterministic mode. Training data subsampling uses a derived RNG from the same
seed. Running the same config with the same seed on the same hardware gives identical
results.

The sweep submission script (`submit_data_efficiency_sweep.sh`) uses seeds 0–4.
Configs default to seed 42 (single-seed / full-data runs).

---

## Configuration

- All configs live under `configs/`.
- One experiment = one YAML under `configs/experiment/training/` (training) or
  `configs/experiment/ood/` (OOD eval).
- Key data-efficiency configs:

| Config | Method | Train size |
|---|---|---|
| `pneumonia_lll_n100.yaml` | LLL | 100 |
| `pneumonia_lll_n1000.yaml` | LLL | 1000 |
| `pneumonia_lll_n10000.yaml` | LLL | 10000 |
| `pneumonia_map_n100/1000/10000.yaml` | MAP | 100 / 1000 / 10000 |
| `pneumonia_ensemble_n100/1000/10000.yaml` | Ensemble | 100 / 1000 / 10000 |

Training-size subsampling is stratified (preserves class proportions via
largest-remainder rounding). Val and test sets are always the full MedMNIST splits.

---

## Outputs

```
outputs/<run_name>/<timestamp>/
  config.yaml           — full config used (including actual seed)
  metrics.json          — in-distribution accuracy, AUROC, ECE
  checkpoint_path.txt   — path to best checkpoint
  sigma_summary.json    — LLL only: mean/max diag(Σ), ‖Σ‖_F
  ood/
    far_ood/ood_metrics.json   — AUROC per uncertainty score (BloodMNIST)
    near_ood/ood_metrics.json  — AUROC per uncertainty score (OrganAMNIST)
```

Checkpoints: `checkpoints/<run_name>/<timestamp>/best.pt`

---

## Running locally (no SLURM)

```bash
python scripts/train.py --config configs/experiment/training/pneumonia_lll_n1000.yaml

# Then evaluate:
python scripts/evaluate.py --run-dir outputs/pneumonia_lll_n1000/<timestamp>
python scripts/evaluate_ood.py --run-dir outputs/pneumonia_lll_n1000/<timestamp> \
    --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
python scripts/evaluate_ood.py --run-dir outputs/pneumonia_lll_n1000/<timestamp> \
    --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
```

---

## Repository layout

```
configs/                   Hydra-composable YAMLs
  experiment/training/     One YAML per training experiment
  experiment/ood/          OOD evaluation scenarios
  data/ model/ method/     Reusable config blocks
src/bnn_medmnist/
  data/                    Dataset wrappers (MedMNIST loader, subsampling)
  models/                  SmallCNN, ResNet-18 builder
  methods/                 LLL, MAP (deterministic), Deep Ensemble
  training/                Trainer (early stopping on val AUROC)
  evaluation/              Metrics, uncertainty scores, OOD scoring
  utils/                   Config loading, seeding, logging
scripts/
  train.py                 Training entry point
  evaluate.py              In-distribution evaluation
  evaluate_ood.py          OOD evaluation
  aggregate_data_efficiency.py   Sweep aggregation (mean ± std across seeds)
slurm/
  resnet18_lll.sbatch              Full-data LLL (ResNet-18)
  resnet18_deep_ensemble.sbatch    Full-data Deep Ensemble
  train_baseline.sbatch            Full-data MAP
  train_lll_data_efficiency.sbatch Data-efficiency single job (takes config + seed)
  submit_data_efficiency_sweep.sh  Submit all 45 sweep jobs
results/                   Aggregated CSVs and plots (generated, not tracked)
data/                      Raw datasets (gitignored, auto-downloaded)
checkpoints/               Model weights (gitignored)
logs/                      SLURM + TensorBoard logs (gitignored)
outputs/                   Run artifacts (small text files tracked)
```
