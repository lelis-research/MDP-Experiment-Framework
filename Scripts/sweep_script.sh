#!/usr/bin/env bash
#SBATCH --job-name=sweep
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G          # memory per node
#SBATCH --time=0-03:00    # time (DD-HH:MM)
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-26      # check sweep.py to calculate arrays

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
CONFIG="config_agents_base"
AGENT="OptionA2C"
ENV="MiniGrid-FourRooms-v0"
ENV_WRAPPING='["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
WRAPPING_PARAMS='[{"agent_view_size":9},{},{"seed":5000},{"num_distractors": 40, "seed": 100}]'
ENV_PARAMS='{}'
SEED=1

NUM_RUNS=3
NUM_EPISODES=0
TOTAL_STEPS=300000
EPISODE_MAX_STEPS=300
NUM_ENVS=1

NUM_WORKERS=3
NAME_TAG="Mask-input-l1-Reg-0.01"
INFO='{
  "option_path":"Runs/Options/MaskedOptionLearner/MaxLen-20_Mask-input-l1_Regularized-0.01_0/selected_options_10.t",
  "actor_step_size": 0.001,
  "critic_step_size": 0.01,
  "rollout_steps": 20
}'  
  # "option_path":"Runs/Options/MaskedOptionLearner/MaxLen-20_Mask-l1_0/selected_options_10.t",

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
  --num_workers        "$NUM_WORKERS" \
  --info               "$INFO" \
  --env_params        "$ENV_PARAMS" \
  --env_wrapping      "$ENV_WRAPPING" \
  --wrapping_params   "$WRAPPING_PARAMS"\