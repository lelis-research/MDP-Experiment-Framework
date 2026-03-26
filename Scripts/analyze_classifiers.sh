#!/usr/bin/env bash
#SBATCH --job-name=analyze_classifiers
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30    # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=rrg-lelis_cpu
#SBATCH --array=0-0

##SBATCH --gres=gpu:1          # <-- uncomment if you want GPU

set -euo pipefail

# ------------------ Paths & modules ------------------
cd ~/scratch/MDP-Experiment-Framework

module load apptainer

CONTAINER=~/scratch/rlbase-amd64.sif

APPTAINER_CMD="apptainer exec"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  APPTAINER_CMD="apptainer exec --nv"
fi

# ------------------ Env vars (visible inside container) ------------------
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ------------------ Configs ------------------
BASE_DIR="Runs/Classifier/MiniGrid-UnlockPickupLimitedColor-v0_/FullyObs_OneHotImageDirCarry/VQOptionCritic"

INCLUDE="onehot-d19"
EXCLUDE=""

EXP_TAG="Feat[delta_last]_KL[0.00]_ReprDim[19]"
NAME_TAG="uniform_d19_KL00_dim19"

# ------------------ Run inside container ------------------
INCLUDE_ARGS=""
for s in $INCLUDE; do
  INCLUDE_ARGS="$INCLUDE_ARGS --include $s"
done

EXCLUDE_ARGS=""
for s in $EXCLUDE; do
  EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude $s"
done

$APPTAINER_CMD "$CONTAINER" \
  python analyze_classifiers.py \
    --base_dir  "$BASE_DIR" \
    --exp_tag   "$EXP_TAG" \
    --name_tag  "$NAME_TAG" \
    $INCLUDE_ARGS \
    $EXCLUDE_ARGS
