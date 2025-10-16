#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=4G          # memory per node
#SBATCH --time=0-01:30      # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-50

## SBATCH --gres=gpu:1

set -euo pipefail

# Move into repo
cd ~/scratch/MDP-Experiment-Framework

# Load modules & env
# module python/3.10
module load mujoco
export MUJOCO_GL=egl
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
AGENT="ContinualOptionQLearning"
ENV="TwoRoomKeyDoorTwoGoalEnv-v0"
#'["NormalizeObs","ClipObs","NormalizeReward", "ClipReward"]' #'["CombineObs"]' #'["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
ENV_WRAPPING='["FullyObs", "FixedSeed"]' #'["RGBImgPartialObs", "FixedSeed"]'
#'[{}, {}, {}, {}]' #'[{"agent_view_size":9},{},{"seed":5000},{"num_distractors": 40, "seed": 100}]'
WRAPPING_PARAMS='[{},{"seed":1}]' #'[{"tile_size":7}, {"seed":5000}]'
ENV_PARAMS='{}' #'{"reward_win":1.0, "reward_lose": 0.0, "penalty_step": 0.0}' #'{"continuing_task":False}'
NAME_TAG="250K_reset_$IDX" #"Test_$IDX"
SEED=$IDX
NUM_WORKERS=1


NUM_EPISODES=0
NUM_RUNS=1
TOTAL_STEPS=250_000
NUM_ENVS=1
EPISODE_MAX_STEPS=300

RENDER_MODE=""           # options: human, rgb_array_list, or leave empty for none
STORE_TRANSITIONS=false  # true / false
CHECKPOINT_FREQ=0         # integer (e.g. 1000), or leave empty for no checkpoints, 0 for only last
INFO='{
  "discount_option_flag": true,
  "epilon_decay_steps": 50000,
  "epsilon_end": 0.01,
  "epsilon_start": 1.0,
  "gamma": 0.99,
  "n_steps": 20,
  "option_init_mode": "reset",
  "option_len": 20,
  "step_size": 0.01,
  "update_action_within_option_flag": false
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
  --wrapping_params   "$WRAPPING_PARAMS"

echo "---- SLURM JOB STATS ----"
seff $SLURM_JOBID || sacct -j $SLURM_JOBID --format=JobID,ReqMem,MaxRSS,Elapsed,State