#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=2G        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-50

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
CONFIG="config_agents_base"
AGENT="OptionRandom"
ENV="MiniGrid-FourRooms-v0"
ENV_WRAPPING='["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
WRAPPING_PARAMS='[{"agent_view_size":9},{},{"seed":5000},{"num_distractors": 30, "seed": 100}]'
ENV_PARAMS='{}'
NAME_TAG="Mask-l1_WithDistractor_$IDX"
SEED=$IDX
NUM_WORKERS=1


NUM_EPISODES=0
NUM_RUNS=1
TOTAL_STEPS=500000
NUM_ENVS=1
EPISODE_MAX_STEPS=300

RENDER_MODE=""           # options: human, rgb_array_list, or leave empty for none
STORE_TRANSITIONS=false  # true / false
CHECKPOINT_FREQ=0         # integer (e.g. 1000), or leave empty for no checkpoints, 0 for only last
INFO='{
  "option_path": "Runs/Options/MaskedOptionLearner/Distractor_MaxLen-20_Mask-l1_'"$SLURM_ARRAY_TASK_ID"'/selected_options_10.t",
  "step_size": 0.0001,
  "epsilon": 0.1,
  "batch_size": 128,
  "target_update_freq": 200
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
  --wrapping_params   "$WRAPPING_PARAMS"\