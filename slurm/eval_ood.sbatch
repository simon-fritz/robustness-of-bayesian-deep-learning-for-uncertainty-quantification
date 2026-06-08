#!/bin/bash
#SBATCH --job-name=eval_ood                                # job name shown in squeue
#SBATCH --output=logs/slurm/eval_ood-%A.out                # stdout, %A = job id
#SBATCH --error=logs/slurm/eval_ood-%A.err                 # stderr
#SBATCH --partition=universe,asteroids                     # cluster partitions
#SBATCH --qos=master-queuesave                             # default QoS
#SBATCH --time=0-01:00:00                                  # walltime (lowered to 1hr for fast eval)
#SBATCH --gres=gpu:1                                       # one GPU
#SBATCH --cpus-per-task=8                                  # CPU cores for the job
#SBATCH --mem=16G                                          # host RAM

ml python/anaconda3
set -e

cd $SLURM_SUBMIT_DIR

python -m pip install --user -e .

# Grab the first and second arguments passed to sbatch
RUN_DIR=$1
# Default to far_blood if no second argument is provided
OOD_CONFIG=${2:-"configs/experiment/ood/pneumonia_far_blood.yaml"} 

# Safety check to ensure you actually provided a run_dir
if [ -z "$RUN_DIR" ]; then
    echo "Error: You must provide a run directory."
    echo "Usage: sbatch slurm/eval_ood.sbatch <path_to_run_dir> [path_to_ood_config]"
    exit 1
fi

echo "Evaluating Model: $RUN_DIR"
echo "OOD Dataset: $OOD_CONFIG"

# Run the python script with the dynamic variables
python scripts/evaluate_ood.py --run-dir "$RUN_DIR" --ood-config "$OOD_CONFIG"