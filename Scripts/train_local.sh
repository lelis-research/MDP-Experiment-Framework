#!/usr/bin/env bash
set -euo pipefail

#— Move into your repo
cd /Users/kiarash/Library/Mobile\ Documents/com~apple~CloudDocs/PhdCode/Code\ Base

#— Activate your virtualenv
source ~/miniconda3/bin/activate
conda activate rl_v1

#— Pin BLAS/OpenMP threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

#— Compute “array” index (defaults to 0 if none given)
IDX=${1:-0}

#— Experiment configuration (adjust SCRIPT_NAME if needed)
CONFIG="config3"
AGENT="A2C"
ENV_NAME="MiniGrid-SimpleCrossingS9N1-v0"
NAME_TAG="$IDX"
SEED=1
NUM_WORKERS=6

NUM_EPISODES=0
NUM_RUNS=30
TOTAL_STEPS=1000000
NUM_ENVS=1
EPISODE_MAX_STEPS=300

#— Optional flags; leave empty to disable
RENDER_MODE=""           # human, rgb_array_list, or ""
STORE_TRANSITIONS=false  # true / false
CHECKPOINT_FREQ=0       # e.g. 1000

#— Build CLI flags
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

#— Finally, launch your experiment
python train.py \
  --config            "$CONFIG" \
  --agent             "$AGENT" \
  --env               "$ENV_NAME" \
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
  --num_workers       "$NUM_WORKERS"