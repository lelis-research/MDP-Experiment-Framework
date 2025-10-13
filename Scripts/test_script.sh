#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:01
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-50

set -euo pipefail

# ----------------- repo + env -----------------
cd ~/scratch/MDP-Experiment-Framework

# If your tests need MuJoCo (MiniGrid doesn't), keep; otherwise you can drop these two lines.
module load mujoco
export MUJOCO_GL=egl

source ~/ENV/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# ----------------- sweep vars -----------------
IDX=$SLURM_ARRAY_TASK_ID   # 1…300
EXP_DIR_REL="TwoRoomKeyDoorTwoGoalEnv-v0_/FullyObs_FixedSeed(seed-1)/OptionQLearning/"$IDX"_seed["$IDX"]"


# These flags only matter if you override the env 
# NOTE: if ENV is not given then the rest of the params will be ignored and just loaded from the train
ENV="" #"TwoRoomKeyDoorTwoGoalEnv-v0"
ENV_WRAPPING='[]'
WRAPPING_PARAMS='[]'
ENV_PARAMS='{}'
EPISODE_MAX_STEPS=0  
# ---


SEED=$IDX
NUM_RUNS=1                 # how many test runs (test.py will make one GIF per run)
RENDER_MODE=""             # test.py forces rgb_array_list internally; this arg is parsed but not used
STORE_TRANSITIONS=false   
NAME_TAG="" 

# ----------------- optional flags -----------------
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

if [ -n "$ENV" ]; then
    ENV="--env $ENV"
else
  ENV=""
fi

# ----------------- run -----------------
python test.py \
  --exp_dir "$EXP_DIR_REL" \
  $ENV \
  --env_wrapping      "$ENV_WRAPPING" \
  --wrapping_params   "$WRAPPING_PARAMS" \
  --env_params        "$ENV_PARAMS" \
  --seed              "$SEED" \
  --num_runs          "$NUM_RUNS" \
  --episode_max_steps "$EPISODE_MAX_STEPS" \
  $RENDER_FLAG \
  $STORE_FLAG \
  --name_tag          "$NAME_TAG"

echo "---- SLURM JOB STATS ----"
seff "$SLURM_JOBID" || sacct -j "$SLURM_JOBID" --format=JobID,ReqMem,MaxRSS,Elapsed,State