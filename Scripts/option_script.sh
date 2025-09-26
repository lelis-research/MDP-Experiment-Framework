#!/usr/bin/env bash
#SBATCH --job-name=Option-Symbolic
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-00:05      # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-0


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
IDX=$SLURM_ARRAY_TASK_ID   

# --------Random Variables-------
NUM_DISTRACTORS=15

# ---------------Configs--------- 
CONFIG="config_options_base"
OPTION_TYPE="ManualSymbolicOptionLearner"
NAME_TAG="FindKey_$IDX" #"Distractor_MaxLen-20_Mask-l1_$IDX"
SEED=$IDX
EXP_PATH_LIST=(
)

INFO='{
    "max_option_len": 20,

    "max_num_options": 5,
    "n_neighbours": 100,
    "n_restarts": 300,
    "n_iteration": 200,

    "n_epochs": 500,
    "actor_lr": 5e-4,

    "reg_coef": 0.0,
    "masked_layers":["1","3","5"]
}' 

RUN_IND_LIST=() #(1 1 1 1 1)
NUM_WORKERS=16
# ----------------------------------

# Invoke Python script with all arguments
python learn_options.py \
  --config            "$CONFIG" \
  --option_type       "$OPTION_TYPE" \
  --seed              "$SEED" \
  --name_tag          "$NAME_TAG" \
  --exp_path_lst      "${EXP_PATH_LIST[@]}" \
  --run_ind_lst       "${RUN_IND_LIST[@]}" \
  --num_workers       "$NUM_WORKERS" \
  --info              "$INFO"


echo "---- SLURM JOB STATS ----"
seff $SLURM_JOBID || sacct -j $SLURM_JOBID --format=JobID,ReqMem,MaxRSS,Elapsed,State

# Temporary line
python clean_storage.py Runs/Options/ --condition selected_options_5.t --target all_options.t --apply