#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=rrg-lelis_cpu
#SBATCH --array=2-50

##SBATCH --gres=gpu:1          # <-- uncomment if you want GPU

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
export FLEXIBLAS=imkl
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ---------------Configs---------
IDX=$SLURM_ARRAY_TASK_ID
EXP_DIR_REL="Runs/Train/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/OptionPPO/conv_"$IDX"_seed["$IDX"]"
NAME_TAG="" 
SEED=$IDX

# These flags only matter if you override the env 
# NOTE: if ENV is not given then the rest of the params will be ignored and just loaded from the train
ENV="" #"TwoRoomKeyDoorTwoGoalEnv-v0"
ENV_WRAPPING='[]'
WRAPPING_PARAMS='[]'
ENV_PARAMS='{}'
EPISODE_MAX_STEPS=500  


NUM_RUNS=5                # how many test runs (test.py will make one GIF per run)
RENDER_MODE=""             # test.py forces rgb_array_list internally; this arg is parsed but not used
STORE_TRANSITIONS=false   
# ------------------------------


if [ -n "$RENDER_MODE" ]; then
  RENDER_FLAG="--render_mode $RENDER_MODE"
else
  RENDER_FLAG=""
fi

if [ "$STORE_TRANSITIONS" = true ]; then
  STORE_FLAG="--store_transitions"
else
  STORE_FLAG=""
fi

if [ -n "$ENV" ]; then
    ENV="--env $ENV"
else
  ENV=""
fi

# ------------------ Run inside container ------------------
$APPTAINER_CMD "$CONTAINER" \
  python test.py \
    --exp_dir "$EXP_DIR_REL" \
    $ENV \
    --env_wrapping      "$ENV_WRAPPING" \
    --wrapping_params   "$WRAPPING_PARAMS" \
    --env_params        "$ENV_PARAMS" \
    --seed              "$SEED" \
    --num_runs          "$NUM_RUNS" \
    --episode_max_steps "$EPISODE_MAX_STEPS" \
    $RENDER_FLAG \
    $STORE_FLAG \
    --name_tag          "$NAME_TAG"

