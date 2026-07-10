#!/bin/bash
#SBATCH --job-name=longtail_sweep
#SBATCH --output=logs/slurm/longtail_sweep-%A.out
#SBATCH --error=logs/slurm/longtail_sweep-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Usage:
#   sbatch slurm/train_longtail_generic.sbatch <config> <seed>
# Example:
#   sbatch slurm/train_longtail_generic.sbatch configs/experiment/pneumonia_resnet18_longtail_lll.yaml 0

set -e
cd "$SLURM_SUBMIT_DIR"
export TENSORBOARD_DIR=off

flock ~/.pip_install.lock python -m pip install --user -e .

CONFIG=${1:-"configs/experiment/pneumonia_resnet18_longtail_lll.yaml"}
SEED=${2:-""}
SEED_ARG=""
if [ -n "$SEED" ]; then
    SEED_ARG="--seed $SEED"
fi
echo "=== Training config: $CONFIG seed=${SEED:-from_config} ==="

python scripts/train.py --config "$CONFIG" $SEED_ARG | tee /tmp/train_out_$SLURM_JOB_ID.txt

RUN_DIR=$(grep "^run_dir:" /tmp/train_out_$SLURM_JOB_ID.txt | tail -1 | awk '{print $2}')
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: could not parse run_dir from training output"
    exit 1
fi
echo "=== run_dir: $RUN_DIR ==="

echo "=== Running ID evaluation ==="
python scripts/evaluate.py --run-dir "$RUN_DIR"

echo "=== Running long-tail OOD evaluation ==="
python scripts/evaluate_ood.py \
    --run-dir "$RUN_DIR" \
    --ood-config configs/experiment/ood/pneumonia_longtail.yaml

echo "=== Running far-OOD evaluation ==="
python scripts/evaluate_ood.py \
    --run-dir "$RUN_DIR" \
    --ood-config configs/experiment/ood/pneumonia_far_blood.yaml

echo "=== Running near-OOD evaluation ==="
python scripts/evaluate_ood.py \
    --run-dir "$RUN_DIR" \
    --ood-config configs/experiment/ood/pneumonia_near_organ.yaml

echo "=== All done for $CONFIG — run_dir=$RUN_DIR ==="
rm -f /tmp/train_out_$SLURM_JOB_ID.txt
