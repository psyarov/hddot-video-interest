#!/bin/bash
# Per-session step as a SLURM job array: one task per row of the sessions CSV.
#
#   N=$(python -m dotinterest.config config/config.yaml)     # index of the last session
#   sbatch --array=0-$N slurm/session_array.sbatch 02_preprocess.py
#   sbatch --array=0-$N --dependency=aftercorr:<jobid> slurm/session_array.sbatch 03_reconstruct.py
#   sbatch --array=0-$N --dependency=aftercorr:<jobid> slurm/session_array.sbatch 04_glm.py
#
# Resources that worked for 567-channel sessions of ~75 min:
#   02_preprocess   CPU, 64 GB, ~1-2 h        03_reconstruct  GPU recommended, 64 GB, ~1 h
#   04_glm          CPU, 4 cores, 16 GB, minutes
#SBATCH --job-name=dot-session
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
STEP=${1:?usage: sbatch --array=0-N slurm/session_array.sbatch <step script>}
CONFIG=${CONFIG:-config/config.yaml}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1

# Either activate the environment ...
python scripts/"$STEP" --config "$CONFIG" --index "$SLURM_ARRAY_TASK_ID"
# ... or run inside a container, e.g.
# apptainer run --nv --bind /path/to/data /path/to/cedalion.sif \
#     python scripts/"$STEP" --config "$CONFIG" --index "$SLURM_ARRAY_TASK_ID"
