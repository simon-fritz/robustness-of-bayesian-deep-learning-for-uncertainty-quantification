#!/bin/bash
#SBATCH --job-name=reeval_all
#SBATCH --output=logs/slurm/reeval_all-%A.out
#SBATCH --error=logs/slurm/reeval_all-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -e

ml python/anaconda3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install --user -e .

# Already-trained runs to re-score with the updated evaluate.py.
RUNS=(
  outputs/pneumonia_deep_ensemble/20260607_133223
  outputs/pneumonia_resnet18_baseline/20260603_125825
  outputs/pneumonia_resnet18_lll/20260603_125825
)

for RUN_DIR in "${RUNS[@]}"; do
  echo "== re-evaluating $RUN_DIR =="
  python scripts/evaluate.py --run-dir "$RUN_DIR"
  python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
  python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
done
