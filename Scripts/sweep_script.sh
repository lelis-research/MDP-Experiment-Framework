#!/usr/bin/env bash
#SBATCH --job-name=sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G          # memory per node
#SBATCH --time=0-00:30    # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-11     # check HP_SEARCH_SPACE to calculate the space size

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
SEED=1
NAME_TAG=""

# ---------------Configs---------
CONFIG="config_agents_base"
AGENT="QLearning"
ENV="CliffWalking-v1"

ENV_WRAPPING='[]'
WRAPPING_PARAMS='[]'
ENV_PARAMS='{}'

NUM_WORKERS=2 # if you want to run in parallel equal to NUM_RUNS
NUM_EPISODES=0
NUM_RUNS=2
TOTAL_STEPS=50_000
NUM_ENVS=1
EPISODE_MAX_STEPS=100

INFO='{
  "gamma": 0.99,
  "epsilon_start": 1.0,
  "epsilon_end": 0.05
}'

HP_SEARCH_SPACE='{
  "step_size": [0.01, 0.001, 0.0001],
  "epsilon_decay_steps": [20000, 50000],
  "n_steps": [1, 3]
}'


# ------------------ Run inside container ------------------
$APPTAINER_CMD "$CONTAINER" \
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
