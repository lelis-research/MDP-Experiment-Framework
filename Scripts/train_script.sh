#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=0-09:00 # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=rrg-lelis_cpu
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
export FLEXIBLAS=imkl
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ------------------ SLURM array index ------------------
IDX=$SLURM_ARRAY_TASK_ID
SEED=$IDX
# NAME_TAG="Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]_$IDX"
NAME_TAG="Options_Add[recreate]_Count[20]_Reset[True]_$IDX"

# ---------------Configs---------
CONFIG="config_agents_base"
AGENT="OptionPPO"
ENV="MiniGrid-MazeRooms-v0"

ENV_WRAPPING='["OneHotImageDirCarry"]'
WRAPPING_PARAMS='[{}]'
ENV_PARAMS='{}'

NUM_WORKERS=1 # if you want to run in parallel equal to NUM_RUNS
NUM_EPISODES=0
NUM_RUNS=1
TOTAL_STEPS=3_000_000
NUM_ENVS=1
EPISODE_MAX_STEPS=1000

RENDER_MODE=""
STORE_TRANSITIONS=false
CHECKPOINT_FREQ=0

INFO='{
  "actor_eps": 1e-08,
  "actor_network": "MiniGrid/PPO/conv_imgdircarry_actor",
  "actor_step_size": 0.001,
  "all_options": "all",
  "anneal_clip_range_actor": false,
  "anneal_clip_range_critic": false,
  "clip_range_actor_init": 0.2,
  "clip_range_critic_init": null,
  "critic_coef": 0.5,
  "critic_eps": 1e-08,
  "critic_network": "MiniGrid/PPO/conv_imgdircarry_critic",
  "critic_step_size": 0.0003,
  "enable_advantage_normalization": true,
  "enable_stepsize_anneal": false,
  "enable_transform_action": true,
  "entropy_coef": 0.01,
  "gamma": 0.99,
  "init_options_lst": "actions",
  "lamda": 0.95,
  "max_grad_norm": 0.5,
  "max_logstd": null,
  "min_logstd": null,
  "mini_batch_size": 128,
  "num_epochs": 10,
  "option_add_policy": "recreate",
  "option_count_to_add": 20,
  "option_learner_reset_at_add": true,
  "rollout_steps": 1024,
  "target_kl": null,
  "total_steps": 500000,
  "update_type": "per_env"
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


    ## vulcan: aip-lelis
    ## fir: def-lelis_cpu, def-lelis_gpu, rrg-lelis_cpu, rrg-lelis_gpu
    ## rorqual: def-lelis_cpu, def-lelis_c=gpu
    ## nibi: def-lelis_cpu, def-lelis_gpu
    ## trillium: def-lelis