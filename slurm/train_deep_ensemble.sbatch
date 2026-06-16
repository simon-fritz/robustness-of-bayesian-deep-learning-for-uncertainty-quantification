#!/bin/bash
#SBATCH --job-name=train_deep_ensemble
#SBATCH --output=logs/slurm/train_deep_ensemble-%A.out
#SBATCH --error=logs/slurm/train_deep_ensemble-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -e

ml python/anaconda3
cd "$SLURM_SUBMIT_DIR"

export TENSORBOARD_DIR=off

python scripts/train.py --config configs/experiment/training/pneumonia_deep_ensemble.yaml
