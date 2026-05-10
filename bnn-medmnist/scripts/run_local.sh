#!/usr/bin/env bash
# Run training locally (no SLURM). Mirrors the command used in slurm/*.sbatch
# so behavior matches across local and cluster runs.
#
# Usage:
#   ./scripts/run_local.sh configs/experiment/pneumonia_lll.yaml
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path/to/experiment.yaml>" >&2
    exit 1
fi

CONFIG="$1"

# Default DATA_ROOT if the user has not set one.
export DATA_ROOT="${DATA_ROOT:-$HOME/.medmnist}"

python scripts/train.py --config "$CONFIG"
