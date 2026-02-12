#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=0-02:00 # time (DD-HH:MM)
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
#"Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]_$IDX"
NAME_TAG="Baseline_emb[dim_16-init_u0.05]_sf[256]_obs[256_16]_dp[0.2]_inp[obs_emb]-nce[0.01-1.0]_$IDX"

# ---------------Configs---------
CONFIG="config_agents_base"
AGENT="OptionRandomSFCodebook"
ENV="MiniGrid-TestOptionRoom-v0"

ENV_WRAPPING='["FullyObs", "OneHotImageDir", "FixedSeed"]'
WRAPPING_PARAMS='[{}, {}, {"seed": 10}]'
ENV_PARAMS='{}'

NUM_WORKERS=1 # if you want to run in parallel equal to NUM_RUNS
NUM_EPISODES=0
NUM_RUNS=1
TOTAL_STEPS=500_000
NUM_ENVS=1
EPISODE_MAX_STEPS=200

RENDER_MODE=""
STORE_TRANSITIONS=false
CHECKPOINT_FREQ=0

INFO='{
  "codebook_embedding_dim": 16,
  "codebook_init_type": "uniform",
  "codebook_init_emb_range": 0.05,

  "sf_hidden_dims": [256],
  "obs_proj_dims": [256, 16],
  "obs_dropout": 0.2,

  "pred_input": "obs-emb",
  "nce_coef": 0.01,
  "nce_tau": 1.0,




  "gamma": 0.99,
  "sf_rollout_steps": 256,

  "codebook_embedding_low": -1.0,
  "codebook_embedding_high": 1.0,

  "codebook_step_size": 3e-4,
  "codebook_eps": 1e-8,
  "codebook_max_grad_norm": 1.0

  
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