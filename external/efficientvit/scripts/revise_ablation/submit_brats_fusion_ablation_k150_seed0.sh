#!/bin/bash
set -euo pipefail

ROOT="external/efficientvit"
cd "${ROOT}"
source ~/envs/brain310/bin/activate

ACCOUNT="${ACCOUNT:-YOUR_SLURM_ACCOUNT}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/revise_ablation_brats_k150_seed0_$(date +%Y%m%d)}"
mkdir -p "${LOG_DIR}"

echo "Submitting BraTS K150 fusion ablation jobs to ${LOG_DIR}"

jid_search=$(sbatch --parsable \
  --job-name="seg_ab_search" \
  --output="${LOG_DIR}/seg_ab_search_%A_%a.out" \
  --error="${LOG_DIR}/seg_ab_search_%A_%a.err" \
  --partition=gpu \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --constraint="a40|a100|v100" \
  --cpus-per-task=8 \
  --mem=96G \
  --time="36:00:00" \
  --array="0-7" \
  --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
TARGETS=(C4 C5 TCGA_LGG TCGA_GBM C4 C5 TCGA_LGG TCGA_GBM)
VARIANTS=(TargetCriteria TargetCriteria TargetCriteria TargetCriteria SourceCriteria SourceCriteria SourceCriteria SourceCriteria)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
echo "target=${TARGET} variant=${VARIANT}"
python -u scripts/revise_ablation/brats_fusion_ablation_k150.py \
  --target "${TARGET}" \
  --variant "${VARIANT}" \
  --train-seed 0 \
  --search-seed 0 \
  --seeds-eval 0 \
  --seeds-refine 0 \
  --run-final-train
')
echo "BraTS Target/SourceCriteria search+train job: ${jid_search}"

jid_uniform=$(sbatch --parsable \
  --job-name="seg_ab_uniform" \
  --output="${LOG_DIR}/seg_ab_uniform_%A_%a.out" \
  --error="${LOG_DIR}/seg_ab_uniform_%A_%a.err" \
  --partition=gpu \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --constraint="a40|a100|v100" \
  --cpus-per-task=8 \
  --mem=96G \
  --time="06:00:00" \
  --array="0-3" \
  --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
TARGETS=(C4 C5 TCGA_LGG TCGA_GBM)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "target=${TARGET} variant=Uniform"
python -u scripts/revise_ablation/brats_fusion_ablation_k150.py \
  --target "${TARGET}" \
  --variant Uniform \
  --train-seed 0 \
  --run-final-train
')
echo "BraTS Uniform train job: ${jid_uniform}"
