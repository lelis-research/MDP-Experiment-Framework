#!/usr/bin/env bash
#SBATCH --job-name=analyze_sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G          # memory per node
#SBATCH --time=0-01:30    # time (DD-HH:MM)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=rrg-lelis_cpu
#SBATCH --array=0-0     # 

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


# ---------------Configs---------
EXP_DIR="Runs/Sweep/MiniGrid-MazeRooms-v0_/OneHotImageDirCarry/VQOptionCritic/conv_dim-8_seed[1]"
RATIO=0.9
AUC_TYPE="steps"




# ------------------ Run inside container ------------------
$APPTAINER_CMD "$CONTAINER" \
  python analyze_sweep.py \
    --exp_dir                "$EXP_DIR" \
    --ratio                  "$RATIO" \
    --auc_type               "$AUC_TYPE"
    
