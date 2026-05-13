#!/bin/bash
#SBATCH --job-name=train_lll                               # job name shown in squeue
#SBATCH --output=logs/slurm/train_lll-%A.out               # stdout, %A = job id
#SBATCH --error=logs/slurm/train_lll-%A.err                # stderr
#SBATCH --partition=universe,asteroids                     # cluster partitions
#SBATCH --qos=master-queuesave                             # default QoS
#SBATCH --time=0-04:00:00                                  # walltime D-HH:MM:SS
#SBATCH --gres=gpu:1                                       # one GPU
#SBATCH --cpus-per-task=8                                  # CPU cores for the job
#SBATCH --mem=16G                                          # host RAM

ml python/anaconda3
conda deactivate
conda activate TODO-env-name

python scripts/train.py --config configs/experiment/pneumonia_lll.yaml
