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
CONFIG="config1"
AGENT="A2C"
ENV_NAME="MiniGrid-SimpleCrossingS9N1-v0"

NAME_TAG="$IDX"
SEED=1

NUM_TRIALS=200 # only used if NOT exhaustive

NUM_EPISODES=0
NUM_RUNS=5
NUM_WORKERS_EACH_TRIALS=5
TOTAL_STEPS=100000
NUM_ENVS=1
EPISODE_MAX_STEPS=300

METRIC_RATIO=0.5
EXHAUSTIVE=false

#— Build CLI flags
if [ "$EXHAUSTIVE" = true ]; then
  EXH_FLAG="--exhaustive"
else
  EXH_FLAG=""
fi



#— Finally, launch your experiment
python tune_hp.py \
  --config            "$CONFIG" \
  --agent             "$AGENT" \
  --env               "$ENV_NAME" \
  --name_tag    "$NAME_TAG" \
  --seed              "$SEED" \
  --num_trials  "$NUM_TRIALS" \
  --num_runs          "$NUM_RUNS" \
  --num_workers_each_trial "$NUM_WORKERS_EACH_TRIALS" \
  --num_episodes      "$NUM_EPISODES" \
  --total_steps       "$TOTAL_STEPS" \
  --episode_max_steps "$EPISODE_MAX_STEPS" \
  --num_envs          "$NUM_ENVS" \
  --metric_ratio "$METRIC_RATIO" \
  $EXH_FLAG