#!/bin/bash
#SBATCH --job-name=rn18_near
#SBATCH --output=logs/slurm/rn18_near-%A.out
#SBATCH --error=logs/slurm/rn18_near-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -e
cd $SLURM_SUBMIT_DIR

python -m pip install --user -e .

for RUN in pneumonia_resnet18_lll pneumonia_resnet18_baseline; do
  RUN_DIR=outputs/$RUN/20260603_125825
  echo "== near-OOD for $RUN_DIR =="
  python scripts/evaluate_ood.py --run-dir "$RUN_DIR" \
    --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
done
