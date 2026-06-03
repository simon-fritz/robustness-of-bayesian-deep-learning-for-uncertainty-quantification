# Robustness of Bayesian Deep Learning for Uncertainty Quantification

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

## Data

MedMNIST datasets are downloaded automatically by the [`medmnist`](https://pypi.org/project/medmnist/)
package on first use (the `.npz` files are fetched from Zenodo). You do not
need to download anything manually.

By default they land in `./data/`. Override the location with `DATA_ROOT`:

```bash
export DATA_ROOT=/some/shared/path
```

Relative paths are resolved against the repository root, absolute paths and `~`
are honored.

## Models

Two architectures coexist and are switchable via `cfg.model` (`configs/model/`):

- **`small_cnn`** (`configs/model/small_cnn.yaml`) — compact from-scratch CNN for
  native 28×28 MedMNIST inputs. Fast; used for pipeline validation and as a
  controlled comparison point.
- **`pretrained_resnet18`** (`configs/model/resnet18.yaml`) — ImageNet-pretrained
  torchvision ResNet-18, fine-tuned. Inputs are resized to 224×224 and expanded
  to 3 channels via the data loader's `image_transform` block. Its classifier
  head is named `fc`, so last-layer Laplace works unchanged on both models.

Both work with every method (deterministic, last-layer Laplace). See
[docs/architecture_choice.md](docs/architecture_choice.md) for why ResNet-18 was
chosen as the second architecture and what we deliberately did not pick.

Example ResNet-18 experiments:

```bash
python scripts/train.py --config configs/experiment/pneumonia_resnet18_baseline.yaml
python scripts/train.py --config configs/experiment/pneumonia_resnet18_lll.yaml
```

## Configuration

- Configs are Hydra-composable YAMLs under `configs/`.
- One experiment = one composed config under `configs/experiment/`, split into
  `configs/experiment/training/` (produce checkpoints) and
  `configs/experiment/ood/` (post-hoc OOD evaluation scenarios).

## Run locally

```bash
./scripts/run_local.sh configs/experiment/training/pneumonia_lll.yaml
```

## Run on SLURM

```bash
sbatch slurm/train_lll.sbatch
```

Logs land in `logs/slurm/`.

## Repository layout

```
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

All artifact paths are anchored to the repository root inside the scripts, so
they end up there regardless of which directory you launch from.
