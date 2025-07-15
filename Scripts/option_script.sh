#!/usr/bin/env bash
#SBATCH --job-name=exp10
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=2G        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-0

set -euo pipefail

# Move into repo
cd ~/scratch/MDP-Experiment-Framework

# Load modules & env
# module python/3.10
source ~/ENV/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# Compute array‐task index
IDX=$SLURM_ARRAY_TASK_ID   # 1…300
# ---------------Configs--------- 
CONFIG="config_options"
OPTION_TYPE="MaskedOptionLearner"
NAME_TAG="$IDX"
SEED=$IDX

# ------------------------------



python learn_options.py \
  --config            "$CONFIG" \
  --option_type       "$OPTION_TYPE"\
  --seed              "$SEED" \
  --name_tag          "$NAME_TAG" \
