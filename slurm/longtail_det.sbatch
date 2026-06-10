#!/bin/bash
#SBATCH --job-name=longtail_rn18_det
#SBATCH --output=logs/slurm/longtail_rn18_det-%A.out
#SBATCH --error=logs/slurm/longtail_rn18_det-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -e

ml python/anaconda3
cd "$SLURM_SUBMIT_DIR"

# Avoid TensorBoard event writes on the shared network mount (Errno 5 / EIO).
export TENSORBOARD_DIR=off

CFG=configs/experiment/pneumonia_resnet18_longtail_baseline.yaml
python scripts/train.py --config "$CFG"

RUN_DIR=$(ls -td outputs/pneumonia_resnet18_longtail_normal2pct_det/*/ | head -1)
echo "RUN_DIR=$RUN_DIR"

python scripts/evaluate.py --run-dir "$RUN_DIR"
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_longtail.yaml
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
