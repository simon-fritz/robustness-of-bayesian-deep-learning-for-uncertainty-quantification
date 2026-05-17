#!/usr/bin/env bash
# Run training locally (no SLURM). Mirrors the command used in slurm/*.sbatch
# so behavior matches across local and cluster runs.
#
# Usage:
#   ./scripts/run_local.sh configs/experiment/training/pneumonia_lll.yaml
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <path/to/experiment.yaml>" >&2
    exit 1
fi

CONFIG="$1"

# Always run from the package root so relative paths (configs, ./data, ...) resolve here.
cd "$(dirname "$0")/.."

# If the user has not set DATA_ROOT, fall through to the YAML default (./data,
# anchored to the package root by the loader). Setting it here would override
# the YAML and split downloads across two locations.

python scripts/train.py --config "$CONFIG"
