#!/bin/bash
#SBATCH --job-name=rn18_baseline
#SBATCH --output=logs/slurm/rn18_baseline-%A.out
#SBATCH --error=logs/slurm/rn18_baseline-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --user -e .

CFG=configs/experiment/pneumonia_resnet18_baseline.yaml
python scripts/train.py --config "$CFG"

RUN_DIR=$(ls -td outputs/pneumonia_resnet18_baseline/*/ | head -1)
echo "RUN_DIR=$RUN_DIR"

python scripts/evaluate.py --run-dir "$RUN_DIR"
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
