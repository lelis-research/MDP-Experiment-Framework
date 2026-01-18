#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-50

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
SEED=$IDX
NAME_TAG="cb-32d_$IDX"

# ---------------Configs---------
CONFIG="config_agents_base"
AGENT="VQOptionCritic"
ENV="MiniGrid-MazeRooms-v0"

ENV_WRAPPING='[]'
WRAPPING_PARAMS='[]'
ENV_PARAMS='{}'

NUM_WORKERS=1 # if you want to run in parallel equal to NUM_RUNS
NUM_EPISODES=0
NUM_RUNS=1
TOTAL_STEPS=5_000_000
NUM_ENVS=1
EPISODE_MAX_STEPS=500

RENDER_MODE=""
STORE_TRANSITIONS=false
CHECKPOINT_FREQ=0

INFO='{
  "codebook_embedding_high": 1.0,
  "codebook_embedding_low": -1.0,
  "codebook_eps": 1e-05,
  "codebook_max_grad_norm": 1.0,
  "codebook_num_embeddings": 32,
  "codebook_step_size": 0.0003,
  "commit_coef": 0.3,
  "gamma": 0.99,
  "hl_actor_eps": 1e-08,
  "hl_actor_network": "MiniGrid/PPO/mlp_actor",
  "hl_actor_step_size": 0.0003,
  "hl_anneal_clip_range_actor": false,
  "hl_anneal_clip_range_critic": false,
  "hl_clip_range_actor_init": 0.2,
  "hl_clip_range_critic_init": 0.2,
  "hl_critic_coef": 0.5,
  "hl_critic_eps": 1e-08,
  "hl_critic_network": "MiniGrid/PPO/mlp_critic",
  "hl_critic_step_size": 0.0003,
  "hl_enable_advantage_normalization": true,
  "hl_enable_stepsize_anneal": false,
  "hl_enable_transform_action": true,
  "hl_entropy_coef": 0.01,
  "hl_lamda": 0.95,
  "hl_max_grad_norm": 0.5,
  "hl_max_logstd": null,
  "hl_min_logstd": null,
  "hl_mini_batch_size": 64,
  "hl_num_epochs": 10,
  "hl_rollout_steps": 2048,
  "hl_target_kl": null,
  "hl_total_steps": 200000
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

# ------------------ Run inside container ------------------
$APPTAINER_CMD "$CONTAINER" \
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