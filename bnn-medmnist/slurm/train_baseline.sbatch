#!/bin/bash
#SBATCH --job-name=train_baseline                          # job name shown in squeue
#SBATCH --output=logs/slurm/train_baseline-%A.out          # stdout, %A = job id
#SBATCH --error=logs/slurm/train_baseline-%A.err           # stderr
#SBATCH --partition=universe,asteroids                     # cluster partitions
#SBATCH --qos=master-queuesave                             # default QoS
#SBATCH --time=0-04:00:00                                  # walltime D-HH:MM:SS
#SBATCH --gres=gpu:1                                       # one GPU
#SBATCH --cpus-per-task=8                                  # CPU cores for the job
#SBATCH --mem=16G                                          # host RAM

ml python/anaconda3
conda deactivate
conda activate TODO-env-name

# TODO: add configs/experiment/pneumonia_baseline.yaml (deterministic) and switch to it.
python scripts/train.py --config configs/experiment/pneumonia_baseline.yaml
