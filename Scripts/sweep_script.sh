#!/usr/bin/env bash
#SBATCH --job-name=sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G          # memory per node
#SBATCH --time=0-06:00    # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-242      # check HP_SEARCH_SPACE to calculate the space size

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
AGENT="OptionDQN" 
ENV="BigCurriculumEnv-v0"
#'["NormalizeObs","ClipObs","NormalizeReward", "ClipReward"]' #'["CombineObs"]' #'["ViewSize","FlattenOnehotObj","FixedSeed","FixedRandomDistractor"]'
ENV_WRAPPING='[]' #'["RGBImgPartialObs", "FixedSeed"]' #, "DropMission", "FrameStack", "MergeStackIntoChannels"]'
#'[{}, {}, {}, {}]' #'[{"agent_view_size":9},{},{"seed":5000},{"num_distractors": 40, "seed": 100}]'
WRAPPING_PARAMS='[]' #'[{"tile_size":7}, {"seed":5000}]' #, {}, {"stack_size":4}, {}]'
ENV_PARAMS='{}' #'{"continuing_task":False}'
SEED=1

NUM_RUNS=2
NUM_WORKERS=2 #If you want all the runs to be parallel NUM_WORKERS and NUM_RUNS should be equal
NUM_EPISODES=0
TOTAL_STEPS=400_000
EPISODE_MAX_STEPS=2500
NUM_ENVS=1


NAME_TAG=""
INFO='{
  "gamma": 0.99,
  "discount_option_flag": true,
  "option_len": 20,
  "epsilon_start": 1.0,
  "target_update_freq": 20,
  "replay_buffer_cap": 100000
}'  
# "option_path": "Runs/Options/MaskedOptionLearner/MaxLen-20_Mask-input_Regularized-0.01_NumDistractors-25_0/selected_options_10.t"

HP_SEARCH_SPACE='{
  "step_size": [0.01, 0.001, 0.0001],
  "epsilon_end":[0.1, 0.01, 0.001],
  "epsilon_decay_steps": [10000, 50000, 100000],
  "batch_size": [32, 128, 512],
  "n_steps": [1, 5, 10]
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