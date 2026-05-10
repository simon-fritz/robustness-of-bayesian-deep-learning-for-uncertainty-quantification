# bnn-medmnist

OOD detection with Bayesian Neural Networks on MedMNIST.

Based on Li et al. (2025). First experiment: last-layer Laplace approximation on
PneumoniaMNIST, extending to other datasets (BloodMNIST, PathMNIST), methods
(MC-Dropout, VI, Deep Ensembles), and varying Bayesian extent (last layer /
last N layers / full network).

## Install

```bash
pip install -e .
```

Python 3.10+ required.

## Configuration

- Configs are Hydra-composable YAMLs under `configs/`.
- One experiment = one composed config under `configs/experiment/`.
- Data root is read from `$DATA_ROOT` (default: `~/.medmnist`).

## Run locally

```bash
./scripts/run_local.sh configs/experiment/pneumonia_lll.yaml
```

## Run on SLURM

```bash
sbatch slurm/train_lll.sbatch
```

Logs land in `logs/slurm/`.

## Directory map

```
configs/         Hydra configs (data / model / method / experiment)
src/bnn_medmnist/
  data/          Dataset wrappers, OOD pair definitions
  models/        Network architectures
  methods/       Bayesian inference methods (Laplace, MC-Dropout, VI, ...)
  training/      Trainer
  evaluation/    Metrics, uncertainty, OOD scoring
  utils/         Config, seeding, logging
scripts/         Entry points (train.py, evaluate.py, run_local.sh)
slurm/           sbatch templates
notebooks/       Exploration only — never used for training
tests/
```
