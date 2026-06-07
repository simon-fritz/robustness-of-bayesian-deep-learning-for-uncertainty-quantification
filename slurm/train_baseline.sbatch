#!/bin/bash
#SBATCH --job-name=train_baseline                          # job name shown in squeue
#SBATCH --output=logs/slurm/train_baseline-%A.out          # stdout, %A = job id
#SBATCH --error=logs/slurm/train_baseline-%A.err           # stderr
#SBATCH --partition=universe,asteroids                     # cluster partitions
#SBATCH --qos=master-queuesave                             # default QoS
#SBATCH --time=0-04:00:00                                  # walltime D-HH:MM:SS
#SBATCH --gres=gpu:1                                       # one GPU
#SBATCH --cpus-per-task=8                                  # CPU cores for the job
#SBATCH --mem=16G                                          # host RAM

ml python/anaconda3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --user -e .

python scripts/train.py --config configs/experiment/training/pneumonia_baseline.yaml
