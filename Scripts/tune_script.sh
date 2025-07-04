#!/usr/bin/env bash
#SBATCH --job-name=exp
#SBATCH --cpus-per-task=3   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-02:30      # time (DD-HH:MM)
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --account=rrg-lelis
#SBATCH --mail-user=aghakasi@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=1-750

set -euo pipefail

# Move into repo
cd ~/scratch/MDP-Experiment-Framework

# Load modules & env
# module python/3.10
source ENV/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# Compute array‐task index
IDX=$SLURM_ARRAY_TASK_ID   # 1…300
# ---------------Configs--------- 
CONFIG="config1"
AGENT="A2C"
ENV="MiniGrid-SimpleCrossingS9N1-v0"

NAME_TAG="grid_search"
SEED=1

NUM_TRIALS=0 # only used if NOT exhaustive

NUM_EPISODES=0
NUM_RUNS=3
NUM_WORKERS_EACH_TRIALS=3
TOTAL_STEPS=500000
NUM_ENVS=1
EPISODE_MAX_STEPS=300

METRIC_RATIO=0.5
EXHAUSTIVE=true
# ------------------------------

if [ "$EXHAUSTIVE" = true ]; then
  EXH_FLAG="--exhaustive"
else
  EXH_FLAG=""
fi

python tune_hp.py \
  --config      "$CONFIG" \
  --agent       "$AGENT" \
  --env         "$ENV" \
  --name_tag    "$NAME_TAG" \
  --seed        "$SEED" \
  --num_trials  "$NUM_TRIALS" \
  --num_episodes "$NUM_EPISODES" \
  --num_runs    "$NUM_RUNS" \
  --num_workers_each_trial "$NUM_WORKERS_EACH_TRIALS" \
  --total_steps "$TOTAL_STEPS" \
  --num_envs    "$NUM_ENVS" \
  --episode_max_steps "$EPISODE_MAX_STEPS" \
  --metric_ratio "$METRIC_RATIO" \
  $EXH_FLAG