#!/bin/bash
#SBATCH --job-name=data_eff_sweep
#SBATCH --output=logs/slurm/data_eff_sweep-%A.out
#SBATCH --error=logs/slurm/data_eff_sweep-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

ml python/anaconda3
set -e
cd $SLURM_SUBMIT_DIR

python -m pip install --user -e .

# Usage:
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n100.yaml
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n1000.yaml
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n10000.yaml
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_map_n100.yaml
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_map_n1000.yaml
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_map_n10000.yaml

CONFIG=${1:-"configs/experiment/training/pneumonia_lll_n1000.yaml"}
echo "=== Training config: $CONFIG ==="

# --- Phase 1: train ---
python scripts/train.py --config "$CONFIG" | tee /tmp/train_out_$SLURM_JOB_ID.txt

RUN_DIR=$(grep "^run_dir:" /tmp/train_out_$SLURM_JOB_ID.txt | tail -1 | awk '{print $2}')
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: could not parse run_dir from training output"
    exit 1
fi
echo "=== run_dir: $RUN_DIR ==="

# --- Phase 2: in-distribution evaluation ---
echo "=== Running ID evaluation ==="
python scripts/evaluate.py --run-dir "$RUN_DIR"

# --- Phase 3: OOD evaluations ---
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
