#!/usr/bin/env bash
#SBATCH --job-name=sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G          # memory per node
#SBATCH --time=1-00:00    # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-540     # check HP_SEARCH_SPACE to calculate the space size

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
IDX=$((SLURM_ARRAY_TASK_ID + 0)) # offset to avoid conflicts with other sweeps
SEED=1
NAME_TAG="conv-dim16"

# ---------------Configs---------
CONFIG="config_agents_base"
AGENT="VQOptionCritic"
ENV="MiniGrid-MazeRooms-v0"

ENV_WRAPPING='["OneHotImageDirCarry"]'
WRAPPING_PARAMS='[{}]'
ENV_PARAMS='{}'

NUM_WORKERS=2 # if you want to run in parallel equal to NUM_RUNS
NUM_EPISODES=0
NUM_RUNS=2
TOTAL_STEPS=2_000_000
NUM_ENVS=1
EPISODE_MAX_STEPS=500

INFO='{
  "gamma": 0.99,
  "hl_lamda": 0.95,

  "hl_actor_network": "MiniGrid/PPO/conv_imgdircarry_actor",
  "hl_actor_eps": 1e-8,
  "hl_clip_range_actor_init": 0.2,
  "hl_anneal_clip_range_actor": false,

  "hl_critic_network": "MiniGrid/PPO/conv_imgdircarry_critic",
  "hl_critic_eps": 1e-8,
  "hl_clip_range_critic_init": null,
  "hl_anneal_clip_range_critic": false,

  "hl_critic_coef": 0.5,
  "hl_max_grad_norm": 0.5,

  "hl_target_kl": null,
  "hl_min_logstd": null,
  "hl_max_logstd": null,

  "hl_enable_stepsize_anneal": false,
  "hl_total_steps": 200000,

  "hl_enable_advantage_normalization": true,
  "hl_enable_transform_action": true,

  "codebook_embedding_dim": 16,
  "codebook_embedding_low": -1.0,
  "codebook_embedding_high": 1.0,
  "codebook_eps": 1e-5,
  "codebook_max_grad_norm": 1.0,

  "hl_rollout_steps": 1024,
  "hl_mini_batch_size": 128,
  "hl_num_epochs": 10
}'

HP_SEARCH_SPACE='{
  "hl_actor_step_size": [1e-4, 3e-4, 1e-3],
  "hl_critic_step_size": [1e-4, 3e-4, 1e-3],

  "hl_entropy_coef": [0.0, 0.0003, 0.001, 0.003, 0.01],

  "commit_coef": [0.05, 0.1, 0.2, 0.4],
  "codebook_step_size": [1e-4, 3e-4, 1e-3]
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
