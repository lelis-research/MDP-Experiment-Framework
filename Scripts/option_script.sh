#!/usr/bin/env bash
#SBATCH --job-name=Option
#SBATCH --cpus-per-task=16   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=256G        # memory per node
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --account=aip-lelis
#SBATCH --array=0-50


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
OPTION_TYPE="MaskedOptionLearner"
NAME_TAG="MaxLen-20_RGB_Mask-l1-l3-l5_$IDX" #"Distractor_MaxLen-20_Mask-l1_$IDX"
SEED=$IDX
EXP_PATH_LIST=(
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-1000)/A2C/${IDX}_seed[${IDX}]"
    "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-2000)/A2C/${IDX}_seed[${IDX}]"
    "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-3000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-4000)/A2C/${IDX}_seed[${IDX}]"
    "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-5000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-6000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-7000)/A2C/${IDX}_seed[${IDX}]"
    "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-8000)/A2C/${IDX}_seed[${IDX}]"
    "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-9000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/RGBImgPartialObs(tile_size-7)_FixedSeed(seed-10000)/A2C/${IDX}_seed[${IDX}]"

    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-1000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-3000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-4000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-6000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-7000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-8000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-9000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-10000)_FixedRandomDistractor(num_distractors-${NUM_DISTRACTORS}_seed-100)/A2C/${IDX}_seed[${IDX}]"


    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-1000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-3000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-4000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-6000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-7000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-8000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-9000)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-10000)/A2C/${IDX}_seed[${IDX}]"

    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-1000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-2000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-3000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-4000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-5000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-6000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-7000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-8000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-9000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
    # "Runs/Train/MiniGrid-SimpleCrossingS9N1-v0_/ViewSize(agent_view_size-9)_FlattenOnehotObj_FixedSeed(seed-10000)_FixedRandomDistractor(num_distractors-10_seed-100)/A2C/${IDX}_seed[${IDX}]"
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
    "masked_layers":["1", "3", "5"]
}' 

RUN_IND_LIST=(1 1 1 1 1) #1 1 1 1 1)
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