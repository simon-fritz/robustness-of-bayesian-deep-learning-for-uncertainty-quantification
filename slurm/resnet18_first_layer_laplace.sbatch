#!/bin/bash
#SBATCH --job-name=rn18_first_layer_laplace
#SBATCH --output=logs/slurm/rn18_first_layer_laplace-%A.out
#SBATCH --error=logs/slurm/rn18_first_layer_laplace-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -e

ml python/anaconda3
cd "$SLURM_SUBMIT_DIR"

# Avoid TensorBoard event writes on the shared network mount (Errno 5 / EIO).
export TENSORBOARD_DIR=off

# Laplace fit uses torch.func.jacrev with vmap — reduce CUDA memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=configs/experiment/pneumonia_resnet18_first_layer_laplace.yaml
python scripts/train.py --config "$CFG"

RUN_DIR=$(ls -td outputs/pneumonia_resnet18_first_layer_laplace/*/ | head -1)
echo "RUN_DIR=$RUN_DIR"

python scripts/evaluate.py --run-dir "$RUN_DIR"
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
