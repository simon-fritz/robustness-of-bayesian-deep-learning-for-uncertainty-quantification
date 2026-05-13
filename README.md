# Robustness of Bayesian Deep Learning for Uncertainty Quantification

OOD detection with Bayesian Neural Networks on MedMNIST.

Based on Li et al. (2025). First experiment: last-layer Laplace approximation on
PneumoniaMNIST, extending to other datasets (BloodMNIST, PathMNIST), methods
(MC-Dropout, VI, Deep Ensembles), and varying Bayesian extent (last layer /
last N layers / full network).

The Python package lives in [bnn-medmnist/](bnn-medmnist/).

## Install

```bash
cd bnn-medmnist
pip install -e .
```

Python 3.10+ required.

## Configuration

- Configs are Hydra-composable YAMLs under `bnn-medmnist/configs/`.
- One experiment = one composed config under `bnn-medmnist/configs/experiment/`.
- Data root is read from `$DATA_ROOT` (default: `~/.medmnist`).

## Run locally

```bash
cd bnn-medmnist
./scripts/run_local.sh configs/experiment/pneumonia_lll.yaml
```

## Run on SLURM

```bash
cd bnn-medmnist
sbatch slurm/train_lll.sbatch
```

Logs land in `logs/slurm/`.

## Repository layout

```
bnn-medmnist/        Python package (pyproject.toml lives here)
  configs/           Hydra configs (data / model / method / experiment)
  src/bnn_medmnist/
    data/            Dataset wrappers, OOD pair definitions
    models/          Network architectures
    methods/         Bayesian inference methods (Laplace, MC-Dropout, VI, ...)
    training/        Trainer
    evaluation/      Metrics, uncertainty, OOD scoring
    utils/           Config, seeding, logging
  scripts/           Entry points (train.py, evaluate.py, run_local.sh)
  slurm/             sbatch templates
  notebooks/         Exploration only — never used for training
  tests/
  data/              Raw datasets (gitignored)
  checkpoints/       Model weights (gitignored)
  logs/              Training / tensorboard logs (gitignored)
  outputs/           Run artifacts — only small text files tracked
```

All artifacts (data, checkpoints, logs, outputs) live under `bnn-medmnist/`
because scripts use relative paths and are run from there.
