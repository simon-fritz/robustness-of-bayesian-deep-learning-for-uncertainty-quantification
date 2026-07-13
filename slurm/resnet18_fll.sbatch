#!/bin/bash
#SBATCH --job-name=rn18_fll
#SBATCH --output=logs/slurm/rn18_fll-%A.out
#SBATCH --error=logs/slurm/rn18_fll-%A.err
#SBATCH --partition=universe,asteroids
#SBATCH --qos=master-queuesave
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# pipefail so a failed `train.py | tee` aborts the job instead of falling
# through to evaluation on a half-written run dir.
set -eo pipefail
cd $SLURM_SUBMIT_DIR

flock ~/.pip_install.lock python -m pip install --user -e .

# Avoid TensorBoard event writes on the shared network mount (Errno 5 / EIO).
export TENSORBOARD_DIR=off

# Trains its own MAP model then fits the first-layer Laplace. Optional seed arg
# (defaults to the config's seed=42):
#   sbatch slurm/resnet18_fll.sbatch <seed>
# To reuse an existing same-seed MAP checkpoint instead, add reuse_map_from /
# map_checkpoint to the experiment config (see comments there).
SEED=${1:-""}
SEED_ARG=""
if [ -n "$SEED" ]; then
    SEED_ARG="--seed $SEED"
fi

CFG=configs/experiment/pneumonia_resnet18_fll.yaml
python scripts/train.py --config "$CFG" $SEED_ARG | tee /tmp/train_out_$SLURM_JOB_ID.txt

RUN_DIR=$(grep "^run_dir:" /tmp/train_out_$SLURM_JOB_ID.txt | tail -1 | awk '{print $2}')
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: could not parse run_dir from training output"
    exit 1
fi
echo "RUN_DIR=$RUN_DIR"

python scripts/evaluate.py --run-dir "$RUN_DIR"
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_far_blood.yaml
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config configs/experiment/ood/pneumonia_near_organ.yaml
rm -f /tmp/train_out_$SLURM_JOB_ID.txt
