#!/usr/bin/env bash
#SBATCH --job-name=sweep
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G          # memory per node
#SBATCH --time=0-03:00    # time (DD-HH:MM)
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-71      # check HP_SEARCH_SPACE to calculate the space size

set -euo pipefail

cd ~/scratch/MDP-Experiment-Framework

# Load modules & env
# module python/3.10
module load mujoco
export MUJOCO_GL=egl
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
AGENT="A2C"
ENV="MiniGrid-SimpleCrossingS9N1-v0"
#'["NormalizeObs","ClipObs","NormalizeReward", "ClipReward"]' #'["CombineObs"]' #'["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
ENV_WRAPPING='["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
#'[{}, {}, {}, {}]' #'[{"agent_view_size":9},{},{"seed":5000},{"num_distractors": 40, "seed": 100}]'
WRAPPING_PARAMS='[{"agent_view_size":9},{},{"seed":1},{"num_distractors": 25, "seed": 100}]'
ENV_PARAMS='{}' #'{"continuing_task":False}'
SEED=1

NUM_RUNS=3
NUM_EPISODES=0
TOTAL_STEPS=200_000
EPISODE_MAX_STEPS=300
NUM_ENVS=1

NUM_WORKERS=3
NAME_TAG=""
INFO='{
  "gamma": 0.99,
  "lamda": 0.95,
  "anneal_step_size_flag": false,
  "actor_network": "fc_network_relu",
  "critic_network": "fc_network_relu",
  "entropy_coef": 0.0
}'  

HP_SEARCH_SPACE='{
  "actor_step_size": [1e-4, 3e-4, 3e-5], 
  "critic_step_size": [1e-4, 3e-4, 3e-5],
  "rollout_steps": [16, 32, 128, 1024],
  "norm_adv_flag": [true, false]
}'
# "mini_batch_size":  [32, 64]

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
  --wrapping_params   "$WRAPPING_PARAMS" \
  --hp_search_space   "$HP_SEARCH_SPACE"  