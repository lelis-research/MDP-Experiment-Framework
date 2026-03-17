#!/usr/bin/env bash
#SBATCH --job-name=sweep-ppo
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G          # memory per node
#SBATCH --time=0-2:00    # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=rrg-lelis_cpu
#SBATCH --array=0-35     # check HP_SEARCH_SPACE to calculate the space size

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
export FLEXIBLAS=imkl
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ------------------ SLURM array index ------------------
IDX=$((SLURM_ARRAY_TASK_ID + 0)) # offset to avoid conflicts with other sweeps
SEED=1
# enc[conv]_cb[dim42-l2]_opt[offline-detailed-noNone]_emb[uniform]_dist[cat]
NAME_TAG="Options_WithColor" 

# ---------------Configs---------
CONFIG="config_agents_base"
AGENT="OptionPPO"
ENV="MiniGrid-Unlock-v0"

ENV_WRAPPING='["FullyObs", "OneHotImageDirCarry"]'
WRAPPING_PARAMS='[{}, {}]'
ENV_PARAMS='{}'

NUM_WORKERS=2 # if you want to run in parallel equal to NUM_RUNS
NUM_EPISODES=0
NUM_RUNS=2
TOTAL_STEPS=300_000
NUM_ENVS=1
EPISODE_MAX_STEPS=200

INFO='{
  "gamma": 0.99,
  "lamda": 0.95,

  "actor_network": "MiniGrid/PPO/conv_imgdircarry_actor",
  "critic_network": "MiniGrid/PPO/conv_imgdircarry_critic",

  "actor_eps": 1e-8,
  "critic_eps": 1e-8,

  "clip_range_actor_init": 0.2,
  "anneal_clip_range_actor": false,

  "clip_range_critic_init": null,
  "anneal_clip_range_critic": false,

  "critic_coef": 0.5,
  "max_grad_norm": 0.5,

  "min_logstd": null,
  "max_logstd": null,

  "enable_stepsize_anneal": false,
  "total_steps": 500000,

  "update_type": "per_env",
  "enable_advantage_normalization": true,
  "enable_transform_action": true,

  "target_kl": null,

  "rollout_steps": 1024,
  "mini_batch_size": 128,
  "num_epochs": 10, 

  "init_option_lst": "unlock_pickup_lst_withcolor"
}'
#categorical, continuous, all, actions, l2, cosine
#432
HP_SEARCH_SPACE='{
  "actor_step_size": [1e-4, 3e-4, 1e-3],
  "critic_step_size": [1e-4, 3e-4, 1e-3],
  "entropy_coef": [0.0, 0.001, 0.01, 0.05]
}'

TIED_PARAMS='{}'

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
    --hp_search_space   "$HP_SEARCH_SPACE" \
    --tied_params       "$TIED_PARAMS" 
