#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G          # memory per node
#SBATCH --time=0-00:30      # time (DD-HH:MM)
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
AGENT="QLearning"
ENV="MiniGrid-DoorKey-5x5-v0"
#'["NormalizeObs","ClipObs","NormalizeReward", "ClipReward"]' #'["CombineObs"]' #'["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
ENV_WRAPPING='["ImgObs"]' #'["RGBImgPartialObs", "FixedSeed"]'
#'[{}, {}, {}, {}]' #'[{"agent_view_size":9},{},{"seed":5000},{"num_distractors": 40, "seed": 100}]'
WRAPPING_PARAMS='[{}]' #'[{"tile_size":7}, {"seed":5000}]'
ENV_PARAMS='{}' #'{"reward_win":1.0, "reward_lose": 0.0, "penalty_step": 0.0}' #'{"continuing_task":False}'
NAME_TAG="$IDX" #"Test_$IDX"
SEED=$IDX
NUM_WORKERS=1


NUM_EPISODES=0
NUM_RUNS=1
TOTAL_STEPS=200_000
NUM_ENVS=1
EPISODE_MAX_STEPS=300

RENDER_MODE=""           # options: human, rgb_array_list, or leave empty for none
STORE_TRANSITIONS=false  # true / false
CHECKPOINT_FREQ=0         # integer (e.g. 1000), or leave empty for no checkpoints, 0 for only last
INFO='{
  "gamma": 0.99,
  "step_size": 0.1,
  "epsilon": 0.1
}'  
  # "actor_eps": 1e-05,
  # "option_path": "Runs/Options/MaskedOptionLearner/MaxLen-20_Mask-input-l1_Regularized-0.01_'"$SLURM_ARRAY_TASK_ID"'/selected_options_10.t",
  # "target_kl": 0.01,
  # "total_steps": 1000000
  # "max_grad_norm": 0.5,
  # "mini_batch_size": 64,
  # "norm_adv_flag": true,
  # "anneal_clip_range_actor": false,
  # "anneal_clip_range_critic": false,
  # "anneal_step_size_flag": false,
  # "clip_range_critic_init": null,
  # "critic_coef": 0.5,
  # "critic_eps": 1e-05,
  # "num_epochs": 5,
  # "clip_range_actor_init": 0.2,
  # "entropy_coef": 0.0,
  # "actor_network": "fc_network_1",
  # "actor_step_size": 0.001,
  # "critic_network": "fc_network_1",
  # "critic_step_size": 0.001,
  # "gamma": 0.99,
  # "lamda": 0.95,
  # "rollout_steps": 10,
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