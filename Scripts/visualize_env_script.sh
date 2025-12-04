#!/usr/bin/env bash
#SBATCH --job-name=visualizeEnv
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=2G        # memory per node
#SBATCH --time=0-00:02      # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=1-1

set -euo pipefail

# ------------------ Paths & modules ------------------
cd ~/scratch/MDP-Experiment-Framework

module load apptainer

CONTAINER=~/scratch/rlbase-amd64.sif

# If CUDA_VISIBLE_DEVICES is set, we assume we’re on a GPU node and use --nv
APPTAINER_CMD="apptainer exec"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  APPTAINER_CMD="apptainer exec --nv"
fi

# ------------------ Env vars (visible inside container) ------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# Compute array‐task index
IDX=$SLURM_ARRAY_TASK_ID   # 1…300

# ---------------Configs--------- 
seed=$((IDX * 1))
ENV="BigCurriculumEnv-v0"
ENV_WRAPPING='["FullyObs","FixedSeed"]' #'["ViewSize","FlattenOnehotObj","FixedSeed", "FixedRandomDistractor"]' ,"FixedRandomDistractor"
WRAPPING_PARAMS='[{},{"seed":'$seed'}]' #'[{"agent_view_size":9},{},{"seed":5000}, {"num_distractors": 40, "seed": 100}]' ,{"num_distractors": 5, "seed": 100}
ENV_PARAMS='{}'
NAME_TAG="seed_$seed" #"$seed"
# ------------------------------

$APPTAINER_CMD "$CONTAINER" \
  python visualize_env.py \
    --env               "$ENV" \
    --name_tag          "$NAME_TAG" \
    --env_params        "$ENV_PARAMS" \
    --env_wrapping      "$ENV_WRAPPING" \
    --wrapping_params   "$WRAPPING_PARAMS"