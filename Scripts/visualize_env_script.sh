#!/usr/bin/env bash
#SBATCH --job-name=visualizeEnv
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=2G        # memory per node
#SBATCH --time=0-00:02      # time (DD-HH:MM)
#SBATCH --output=logs/vis_%A_%a.out
#SBATCH --error=logs/vis_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=1-1

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
seed=$((IDX * 1000))
ENV="MiniGrid-Empty-8x8-v0"
ENV_WRAPPING='["RGBImgPartialObs","FixedSeed"]' #'["ViewSize","FlattenOnehotObj","FixedSeed", "FixedRandomDistractor"]' ,"FixedRandomDistractor"
WRAPPING_PARAMS='[{"tile_size":7},{"seed":5000}]' #'[{"agent_view_size":9},{},{"seed":5000}, {"num_distractors": 40, "seed": 100}]' ,{"num_distractors": 5, "seed": 100}
ENV_PARAMS='{}'
NAME_TAG="Seed-5000" #"$seed"
# ------------------------------

python visualize_env.py \
  --env               "$ENV" \
  --name_tag          "$NAME_TAG" \
  --env_params        "$ENV_PARAMS" \
  --env_wrapping      "$ENV_WRAPPING" \
  --wrapping_params   "$WRAPPING_PARAMS"

echo "---- SLURM JOB STATS ----"
seff $SLURM_JOBID || sacct -j $SLURM_JOBID --format=JobID,ReqMem,MaxRSS,Elapsed,State