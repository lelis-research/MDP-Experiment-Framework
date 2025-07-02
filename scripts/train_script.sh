#!/usr/bin/env bash
#SBATCH --job-name=exp
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=1G        # memory per node
#SBATCH --time=0-10:00      # time (DD-HH:MM)
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err
#SBATCH --account=rrg-lelis
#SBATCH --mail-user=aghakasi@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

set -euo pipefail

# Move into repo
# cd ~/scratch/neurips-2025-paper-neural-decomposition

# Load modules & env
# module load StdEnv/2020 gcc flexiblas python/3.10 mujoco/2.3.6
# source /home/aghakasi/ENV/bin/activate
source ~/MDP-Experiment-Framework/venvs/rl_v1/bin/activate

# Pin BLAS/OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

# Compute array‐task index
IDX=$SLURM_ARRAY_TASK_ID   # 1…300

python train.py --num_episodes 10