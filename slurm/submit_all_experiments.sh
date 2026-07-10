#!/bin/bash
# Submit full-data and long-tail experiments across 5 seeds (30 jobs).
#
# Usage:
#   bash slurm/submit_all_experiments.sh
#
# After all jobs complete, aggregate with:
#   python scripts/aggregate_all.py --seeds 0 1 2 3 4
#
# Data-efficiency sweep is separate:
#   bash slurm/submit_data_efficiency_sweep.sh

SEEDS=(0 1 2 3 4)

# Full-data balanced experiments (train + far-OOD + near-OOD)
FULL_DATA_CONFIGS=(
    configs/experiment/pneumonia_resnet18_lll.yaml
    configs/experiment/pneumonia_resnet18_baseline.yaml
    configs/experiment/training/pneumonia_resnet18_deep_ensemble.yaml
)

# Long-tail experiments (train + long_tail-OOD + far-OOD + near-OOD)
LONGTAIL_CONFIGS=(
    configs/experiment/pneumonia_resnet18_longtail_lll.yaml
    configs/experiment/pneumonia_resnet18_longtail_baseline.yaml
    configs/experiment/pneumonia_resnet18_longtail_deep_ensemble.yaml
)

cd "$(dirname "$0")/.." || exit 1

submitted=0

echo "=== Submitting full-data experiments ==="
for cfg in "${FULL_DATA_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        jid=$(sbatch --parsable slurm/train_lll_data_efficiency.sbatch "$cfg" "$seed")
        echo "submitted job $jid  cfg=$cfg  seed=$seed"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "=== Submitting long-tail experiments ==="
for cfg in "${LONGTAIL_CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        jid=$(sbatch --parsable slurm/train_longtail_generic.sbatch "$cfg" "$seed")
        echo "submitted job $jid  cfg=$cfg  seed=$seed"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "=== submitted $submitted jobs ==="
echo "Monitor:  squeue -u \$USER"
echo "Aggregate after completion:"
echo "  python scripts/aggregate_all.py --seeds 0 1 2 3 4"
