#!/usr/bin/env bash
#SBATCH --job-name=exp
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-10:00      # time (DD-HH:MM)
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --account=rrg-lelis
#SBATCH --mail-user=aghakasi@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

set -euo pipefail

# Move into repo
cd ~/scratch/MDP-Experiment-Framework

# Load modules & env
# module python/3.10
source ~/MDP-Experiment-Framework/ENV/bin/activate

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
AGENT="Random"
ENV="MiniGrid-Empty-5x5-v0"
NAME_TAG="test"
SEED=1
NUM_WORKERS=1

NUM_EPISODES=10
NUM_RUNS=2
TOTAL_STEPS=0
NUM_ENVS=1
EPISODE_MAX_STEPS=200

RENDER_MODE=""           # options: human, rgb_array_list, or leave empty for none
STORE_TRANSITIONS=false  # true / false
CHECKPOINT_FREQ=         # integer (e.g. 1000), or leave empty for no checkpoints

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

python $SCRIPT_NAME \
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
  --num_workers       "$NUM_WORKERS"