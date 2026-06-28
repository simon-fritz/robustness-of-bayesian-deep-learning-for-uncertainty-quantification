#!/bin/bash
#SBATCH --job-name=resnet18_deep_ensemble
#SBATCH --output=logs/slurm/resnet18_deep_ensemble-%A.out
#SBATCH --error=logs/slurm/resnet18_deep_ensemble-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=atlas
set -e

ml python/anaconda3
cd "$SLURM_SUBMIT_DIR"

# Avoid TensorBoard event writes on the shared network mount (Errno 5 / EIO).
export TENSORBOARD_DIR=off

CFG=configs/experiment/training/pneumonia_deep_ensemble.yaml
python scripts/train.py --config "$CFG"

RUN_DIR=$(ls -td outputs/pneumonia_deep_ensemble/*/ | head -1)
echo "RUN_DIR=$RUN_DIR"

python scripts/evaluate.py --run-dir "$RUN_DIR"
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
