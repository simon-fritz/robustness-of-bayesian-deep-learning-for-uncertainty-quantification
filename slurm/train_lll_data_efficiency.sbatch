#!/bin/bash
#SBATCH --job-name=train_lll_data_eff                      # job name shown in squeue
#SBATCH --output=logs/slurm/train_lll_data_eff-%A.out      # stdout, %A = job id
#SBATCH --error=logs/slurm/train_lll_data_eff-%A.err       # stderr
#SBATCH --partition=universe,asteroids                     # cluster partitions
#SBATCH --qos=master-queuesave                             # default QoS
#SBATCH --time=0-04:00:00                                  # walltime D-HH:MM:SS
#SBATCH --gres=gpu:1                                       # one GPU
#SBATCH --cpus-per-task=8                                  # CPU cores for the job
#SBATCH --mem=16G                                          # host RAM

ml python/anaconda3
set -e

cd $SLURM_SUBMIT_DIR

python -m pip install --user -e .

# Data-efficiency sweep: pick the train-set-size variant via the config path.
# Defaults to the 1000-example run if no config is given.
CONFIG=${1:-"configs/experiment/training/pneumonia_lll_n1000.yaml"}

echo "Training config: $CONFIG"

# Usage:
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n100.yaml
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n1000.yaml
#   sbatch slurm/train_lll_data_efficiency.sbatch configs/experiment/training/pneumonia_lll_n10000.yaml
python scripts/train.py --config "$CONFIG"
