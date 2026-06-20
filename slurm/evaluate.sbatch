#!/bin/bash
#SBATCH --job-name=evaluate                                # job name shown in squeue
#SBATCH --output=logs/slurm/evaluate-%A.out                # stdout, %A = job id
#SBATCH --error=logs/slurm/evaluate-%A.err                 # stderr
#SBATCH --partition=universe,asteroids                     # cluster partitions
#SBATCH --qos=master-queuesave                             # default QoS
#SBATCH --time=0-00:30:00                                  # walltime
#SBATCH --gres=gpu:1                                       # one GPU
#SBATCH --cpus-per-task=4                                  # CPU cores for the job
#SBATCH --mem=16G                                          # host RAM

ml python/anaconda3
set -e

cd "$SLURM_SUBMIT_DIR"

RUN_DIR=$1

if [ -z "$RUN_DIR" ]; then
    echo "Error: You must provide a run directory."
    echo "Usage: sbatch slurm/evaluate.sbatch <path_to_run_dir>"
    exit 1
fi

echo "Evaluating Model: $RUN_DIR"

python scripts/evaluate.py --run-dir "$RUN_DIR"
