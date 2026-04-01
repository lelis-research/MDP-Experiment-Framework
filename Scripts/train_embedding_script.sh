#!/usr/bin/env bash
#SBATCH --job-name=train_embedding
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-01:00 # time (DD-HH:MM)
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

APPTAINER_CMD="apptainer exec"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  APPTAINER_CMD="apptainer exec --nv"
fi

# ------------------ Env vars (visible inside container) ------------------
export PYTHONUNBUFFERED=1
export FLEXIBLAS=imkl

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TORCH_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ------------------ One exp_dir per array index ------------------

IDX=$SLURM_ARRAY_TASK_ID
EXP_DIR="Runs/Train/MiniGrid-UnlockPickupLimitedColor-v0_/FullyObs_OneHotImageDirCarry/VQOptionCritic/Options_LimitedColor_emb[onehot-d19]_${IDX}_seed[$IDX]" # uniform-d8, uniform-d4, onehot-d19

# ------------------ Model type ------------------
MODEL_TYPE="feature"   # encoder | classifier | feature

# ------------------ Classifier to load (only used when MODEL_TYPE=classifier) ------------------
CLASSIFIER_TAG="Feat[delta_last]_KL[0.00]_ReprDim[4]"
CLASSIFIER_CKPT="final_classifier.t"

# ------------------ Embedding config ------------------
NAME_TAG=""            # optional suffix appended to mode in the output path (e.g. "v2")
FEATURE="delta_last_enc"
MODE="repr_learned"        # repr | repr_pca | repr_learned | repr_pca_learned
PCA_DIM=4              # only used when MODE contains "pca"

# Learned-mode params (only used when MODE=repr_learned or repr_pca_learned)
EMB_DIM=4
KL_WEIGHT=0.05
KL_METRIC="l2"     # cosine | l2
KL_TEMPERATURE=1.0
NUM_EPOCHS=20
LR=1e-3

DEVICE="cpu"
SEED=42
BATCH_SIZE=512

# ------------------ Run inside container ------------------
$APPTAINER_CMD "$CONTAINER" \
  python train_option_embedding.py \
    --exp_dir         "$EXP_DIR" \
    --model_type      "$MODEL_TYPE" \
    --classifier_tag  "$CLASSIFIER_TAG" \
    --classifier_ckpt "$CLASSIFIER_CKPT" \
    --name_tag        "$NAME_TAG" \
    --feature         "$FEATURE" \
    --mode            "$MODE" \
    --pca_dim         "$PCA_DIM" \
    --emb_dim         "$EMB_DIM" \
    --kl_weight       "$KL_WEIGHT" \
    --kl_metric       "$KL_METRIC" \
    --kl_temperature  "$KL_TEMPERATURE" \
    --num_epochs      "$NUM_EPOCHS" \
    --lr              "$LR" \
    --device          "$DEVICE" \
    --seed            "$SEED" \
    --batch_size      "$BATCH_SIZE"