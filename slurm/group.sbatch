#!/bin/bash
# One group-level step (00, 01, 05, 06, 07) as a single job.
#
#   sbatch slurm/group.sbatch 05_features.py
#   sbatch --time=2-00:00:00 slurm/group.sbatch 07_decoding.py --stages BD
#
# 01_forward_model needs a GPU node for MCX; 06 and 07 are CPU-bound (permutations) and take
# hours at the default permutation counts.
#SBATCH --job-name=dot-group
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
STEP=${1:?usage: sbatch slurm/group.sbatch <step script> [args]}
shift
CONFIG=${CONFIG:-config/config.yaml}
export PYTHONUNBUFFERED=1
python scripts/"$STEP" --config "$CONFIG" "$@"
