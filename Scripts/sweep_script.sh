#!/usr/bin/env bash
#SBATCH --job-name=sweep_config10
#SBATCH --cpus-per-task=5
#SBATCH --mem=2G          # memory per node
#SBATCH --time=0-03:00    # time (DD-HH:MM)
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --account=def-lelis
#SBATCH --array=0-44      # 3*3*5 - 1 = 44 combos

set -euo pipefail

cd ~/scratch/MDP-Experiment-Framework

# Load modules & env
# module python/3.10
source ~/ENV/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export FLEXIBLAS=imkl
export PYTHONUNBUFFERED=1

# Array index → sweep index
IDX=$SLURM_ARRAY_TASK_ID

# --------------- Hyperparam sweep settings ---------------
CONFIG="config10"
AGENT="A2C"
ENV="MiniGrid-SimpleCrossingS9N1-v0"
SEED=1
NUM_RUNS=5
NUM_EPISODES=0
TOTAL_STEPS=300000
EPISODE_MAX_STEPS=300
NUM_ENVS=1
NUM_WORKERS=5
NAME_TAG=GridSearch
# ---------------------------------------------------------

python sweep.py \
  --idx                "$IDX" \
  --config             "$CONFIG" \
  --agent              "$AGENT" \
  --env                "$ENV" \
  --name_tag          "$NAME_TAG" \
  --seed               "$SEED" \
  --num_runs           "$NUM_RUNS" \
  --num_episodes       "$NUM_EPISODES" \
  --total_steps        "$TOTAL_STEPS" \
  --episode_max_steps  "$EPISODE_MAX_STEPS" \
  --num_envs           "$NUM_ENVS" \
  --num_workers        "$NUM_WORKERS"