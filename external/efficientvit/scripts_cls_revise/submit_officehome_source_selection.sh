#!/bin/bash
#SBATCH --job-name=cls_oh_select
#SBATCH --output=logs_cls_revise/officehome_source_selection/%A_%a.out
#SBATCH --error=logs_cls_revise/officehome_source_selection/%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2

set -euo pipefail

cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs_cls_revise/officehome_source_selection

TARGETS=(Art Clipart Product RealWorld)
TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}
SPLIT_SEED=${SPLIT_SEED:-0}
TARGET_SHOTS=${TARGET_SHOTS:-3}
VAL_SHOTS=${VAL_SHOTS:-2}
METHODS=${METHODS:-all}
BUDGETS=${BUDGETS:-"1 3 5"}
GRADIENT_MODE=${GRADIENT_MODE:-full}

SPLIT_DIR=data_cls_revise/splits/officehome/${TARGET}/seed$(printf "%02d" "${SPLIT_SEED}")
WARMUP_CKPT=experiments_cls_revise/officehome/${TARGET}/warmup_full/split$(printf "%02d" "${SPLIT_SEED}")/train_seed00/best.pt
CACHE_DIR=data_cls_revise/source_subsets/officehome/${TARGET}/seed$(printf "%02d" "${SPLIT_SEED}")/cache

if [[ ! -f "${WARMUP_CKPT}" ]]; then
  echo "Missing warmup checkpoint: ${WARMUP_CKPT}" >&2
  exit 2
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "target=${TARGET}"
echo "warmup=${WARMUP_CKPT}"
echo "budgets=${BUDGETS}"
echo "methods=${METHODS}"

for B in ${BUDGETS}; do
  OUT_DIR=data_cls_revise/source_subsets/officehome/${TARGET}/seed$(printf "%02d" "${SPLIT_SEED}")/B${B}
  echo "Running source selection: target=${TARGET} B=${B}"
  python -u scripts_cls_revise/select_source_officehome.py \
    --source-list "${SPLIT_DIR}/source_train.txt" \
    --target-train-list "${SPLIT_DIR}/target_train_${TARGET_SHOTS}shot.txt" \
    --target-val-list "${SPLIT_DIR}/target_val_${VAL_SHOTS}shot.txt" \
    --output-dir "${OUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --methods ${METHODS} \
    --budget-per-class "${B}" \
    --warmup-ckpt "${WARMUP_CKPT}" \
    --gradient-mode "${GRADIENT_MODE}" \
    --seed "${SPLIT_SEED}" \
    --use-cache
done
