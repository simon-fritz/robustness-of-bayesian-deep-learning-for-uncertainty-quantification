#!/bin/bash
# Submit the full data-efficiency sweep: 9 configs × 5 seeds = 45 SLURM jobs.
#
# Usage:
#   bash slurm/submit_data_efficiency_sweep.sh
#
# Each job trains one (method, train_size, seed) combination, then runs
# in-distribution + far-OOD + near-OOD evaluation.  Results land in
# outputs/<run_name>/<timestamp>/.
#
# After all jobs complete, aggregate with:
#   python scripts/aggregate_data_efficiency.py

SEEDS=(0 1 2 3 4)

CONFIGS=(
    configs/experiment/training/pneumonia_lll_n100.yaml
    configs/experiment/training/pneumonia_lll_n1000.yaml
    configs/experiment/training/pneumonia_lll_n10000.yaml
    configs/experiment/training/pneumonia_map_n100.yaml
    configs/experiment/training/pneumonia_map_n1000.yaml
    configs/experiment/training/pneumonia_map_n10000.yaml
    configs/experiment/training/pneumonia_ensemble_n100.yaml
    configs/experiment/training/pneumonia_ensemble_n1000.yaml
    configs/experiment/training/pneumonia_ensemble_n10000.yaml
)

cd "$(dirname "$0")/.." || exit 1

submitted=0
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        jid=$(sbatch --parsable slurm/train_lll_data_efficiency.sbatch "$cfg" "$seed")
        echo "submitted job $jid  cfg=$cfg  seed=$seed"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "=== submitted $submitted jobs ==="
echo "Monitor:  squeue -u \$USER"
echo "Aggregate after completion:"
echo "  python scripts/aggregate_data_efficiency.py"
