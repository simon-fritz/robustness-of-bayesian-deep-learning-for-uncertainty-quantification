#!/bin/bash
#SBATCH --job-name=data_eff_eval
#SBATCH --output=logs/slurm/data_eff_eval-%A.out
#SBATCH --error=logs/slurm/data_eff_eval-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

ml python/anaconda3
set -e
cd $SLURM_SUBMIT_DIR

python -m pip install --user -e .

# Usage:
#   sbatch slurm/data_efficiency_eval.sbatch <run_dir>
#
# Runs both far-OOD and near-OOD evaluation on an already-trained run.
# Use this to re-run eval without retraining.

RUN_DIR=$1
if [ -z "$RUN_DIR" ]; then
    echo "Error: must provide run_dir as first argument."
    echo "Usage: sbatch slurm/data_efficiency_eval.sbatch outputs/<run_name>/<timestamp>"
    exit 1
fi

echo "=== Evaluating run: $RUN_DIR ==="

echo "=== Far-OOD ==="
python scripts/evaluate_ood.py \
    --run-dir "$RUN_DIR" \
    --ood-config configs/experiment/ood/pneumonia_far_blood.yaml

echo "=== Near-OOD ==="
python scripts/evaluate_ood.py \
    --run-dir "$RUN_DIR" \
    --ood-config configs/experiment/ood/pneumonia_near_organ.yaml

echo "=== Done ==="
