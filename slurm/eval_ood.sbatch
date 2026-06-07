#!/bin/bash
#SBATCH --job-name=eval_ood                                # job name shown in squeue
#SBATCH --output=logs/slurm/eval_ood-%A.out                # stdout, %A = job id
#SBATCH --error=logs/slurm/eval_ood-%A.err                 # stderr
#SBATCH --partition=universe,asteroids                     # cluster partitions
#SBATCH --qos=master-queuesave                             # default QoS
#SBATCH --time=0-04:00:00                                  # walltime D-HH:MM:SS
#SBATCH --gres=gpu:1                                       # one GPU
#SBATCH --cpus-per-task=8                                  # CPU cores for the job
#SBATCH --mem=16G                                          # host RAM

ml python/anaconda3
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --user -e .

RUN_DIR=$(ls -td outputs/pneumonia_lll/*/ | head -1)
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
