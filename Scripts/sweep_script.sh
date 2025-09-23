#!/usr/bin/env bash
#SBATCH --job-name=sweep-ppo
#SBATCH --cpus-per-task=3
#SBATCH --mem=1G          # memory per node
#SBATCH --time=0-02:00    # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-971      # check HP_SEARCH_SPACE to calculate the space size

########SBATCH --gres=gpu:1

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
AGENT="PPO"
ENV="MiniHack-Corridor-R2-v0"
#'["NormalizeObs","ClipObs","NormalizeReward", "ClipReward"]' #'["CombineObs"]' #'["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
ENV_WRAPPING='[]' #'["RGBImgPartialObs", "FixedSeed"]' #, "DropMission", "FrameStack", "MergeStackIntoChannels"]'
#'[{}, {}, {}, {}]' #'[{"agent_view_size":9},{},{"seed":5000},{"num_distractors": 40, "seed": 100}]'
WRAPPING_PARAMS='[]' #'[{"tile_size":7}, {"seed":5000}]' #, {}, {"stack_size":4}, {}]'
ENV_PARAMS='{"seed":12, "view_size":9}' #'{"continuing_task":False}'
SEED=1

NUM_RUNS=3
NUM_WORKERS=3 #If you want all the runs to be parallel NUM_WORKERS and NUM_RUNS should be equal
NUM_EPISODES=0
TOTAL_STEPS=1_000_000
EPISODE_MAX_STEPS=300
NUM_ENVS=1


NAME_TAG=""
INFO='{
  "gamma": 0.99,
  "lamda": 0.95,
  "mini_batch_size": 64,
  "rollout_steps": 256,
  "num_epochs": 5,

  "clip_range_critic_init": null,
  "anneal_clip_range_critic": false,

  "actor_network": "minihack_actor",
  "critic_network": "minihack_critic",
  "actor_eps": 1e-5,
  "critic_eps": 1e-5,
  "anneal_step_size_flag": true,
  "total_steps": 1000000, 
  
  "norm_adv_flag": true,
  "critic_coef": 0.5,
  "option_path": "Runs/Options/MaskedOptionLearner/PPO_MaxLen-20_RGB_Mask-l8_Regularized-0.01_0/selected_options_5.t" 

}'  
# "option_path": "Runs/Options/MaskedOptionLearner/MaxLen-20_Mask-input_Regularized-0.01_NumDistractors-25_0/selected_options_10.t"

HP_SEARCH_SPACE='{
  "clip_range_actor_init": [0.2, 0.1],
  "anneal_clip_range_actor": [true, false],
  "target_kl":[0.01, 0.02, 0.03],
  "actor_step_size": [1e-4, 3e-4, 5e-4], 
  "critic_step_size": [1e-4, 3e-4, 5e-4],
  "entropy_coef": [0.0, 0.01, 0.02],
  "max_grad_norm": [0.3, 0.5, 0.7]
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


echo "---- SLURM JOB STATS ----"
seff $SLURM_JOBID || sacct -j $SLURM_JOBID --format=JobID,ReqMem,MaxRSS,Elapsed,State