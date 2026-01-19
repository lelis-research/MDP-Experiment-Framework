#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0-03:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-50

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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# ------------------ SLURM array index ------------------
IDX=$SLURM_ARRAY_TASK_ID
SEED=$IDX
NAME_TAG="conv_$IDX"

# ---------------Configs---------
CONFIG="config_agents_base"
AGENT="PPO"
ENV="MiniGrid-SimpleCrossingS9N1-v0"

ENV_WRAPPING='["OneHotImageDir"]'
WRAPPING_PARAMS='[{}]'
ENV_PARAMS='{}'

NUM_WORKERS=1 # if you want to run in parallel equal to NUM_RUNS
NUM_EPISODES=0
NUM_RUNS=1
TOTAL_STEPS=500_000
NUM_ENVS=1
EPISODE_MAX_STEPS=100

RENDER_MODE=""
STORE_TRANSITIONS=false
CHECKPOINT_FREQ=0

INFO='{
}'
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

if [ -n "$CHECKPOINT_FREQ" ]; then
  CP_FLAG="--checkpoint_freq $CHECKPOINT_FREQ"
else
  CP_FLAG=""
fi

# ------------------ Run inside container ------------------
$APPTAINER_CMD "$CONTAINER" \
  python train.py \
    --config            "$CONFIG" \
    --agent             "$AGENT" \
    --env               "$ENV" \
    --seed              "$SEED" \
    --num_runs          "$NUM_RUNS" \
    --num_episodes      "$NUM_EPISODES" \
    --total_steps       "$TOTAL_STEPS" \
    --episode_max_steps "$EPISODE_MAX_STEPS" \
    --num_envs          "$NUM_ENVS" \
    $RENDER_FLAG \
    $STORE_FLAG \
    $CP_FLAG \
    --name_tag          "$NAME_TAG" \
    --num_workers       "$NUM_WORKERS" \
    --info              "$INFO" \
    --env_params        "$ENV_PARAMS" \
    --env_wrapping      "$ENV_WRAPPING" \
    --wrapping_params   "$WRAPPING_PARAMS"