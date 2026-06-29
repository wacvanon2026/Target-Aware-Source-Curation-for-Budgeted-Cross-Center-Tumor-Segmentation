#!/bin/bash
set -euo pipefail

ROOT="external/efficientvit"
cd "${ROOT}"
source ~/envs/brain310/bin/activate

ACCOUNT="${ACCOUNT:-YOUR_SLURM_ACCOUNT}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_cls_revise/officehome_fusion_ablation_b15_seed0_$(date +%Y%m%d)}"
mkdir -p "${LOG_DIR}"

COMMON_EXPORTS='POPSIZE=20,MU=10,N_GEN=12,EVAL_EPOCHS=5,REFINE_EPOCHS=15,REFINE_TOPK=8,EVAL_SEEDS=0,REFINE_SEEDS=0,FINAL_TRAIN_SEEDS=0,RUN_FINAL_TRAIN=1,MAX_EVALS=0'

echo "Submitting OfficeHome B15 fusion ablation jobs to ${LOG_DIR}"

submit_search () {
  local variant="$1"
  local script="$2"
  local job="oh_ab_${variant}"
  sbatch --parsable \
    --job-name="${job}" \
    --output="${LOG_DIR}/${job}_%A_%a.out" \
    --error="${LOG_DIR}/${job}_%A_%a.err" \
    --partition=gpu \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --constraint="a40|a100|v100" \
    --cpus-per-task=8 \
    --mem=64G \
    --time="20:00:00" \
    --array="0-3" \
    --export=ALL,${COMMON_EXPORTS},SCRIPT="${script}" \
    --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
TARGETS=(Art Clipart Product RealWorld)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "target=${TARGET} script=${SCRIPT}"
python -u "${SCRIPT}" \
  --target "${TARGET}" \
  --budget-per-class 15 \
  --split-seed 0 \
  --search-seed 0 \
  --train-seed 0 \
  --eval-seeds "${EVAL_SEEDS}" \
  --refine-seeds "${REFINE_SEEDS}" \
  --final-train-seeds "${FINAL_TRAIN_SEEDS}" \
  --search-output-root experiments_cls_revise/officehome_fusion_ablation_search \
  --final-subset-root data_cls_revise/source_subsets/officehome_fusion_ablation \
  --final-config-root configs_cls_revise/officehome_fusion_ablation \
  --final-output-root experiments_cls_revise/officehome_fusion_ablation \
  --popsize "${POPSIZE}" \
  --mu "${MU}" \
  --n-gen "${N_GEN}" \
  --eval-epochs "${EVAL_EPOCHS}" \
  --refine-epochs "${REFINE_EPOCHS}" \
  --refine-topk "${REFINE_TOPK}" \
  --max-evals "${MAX_EVALS}" \
  --run-final-train
'
}

submit_uniform () {
  local job="oh_ab_uniform"
  sbatch --parsable \
    --job-name="${job}" \
    --output="${LOG_DIR}/${job}_%A_%a.out" \
    --error="${LOG_DIR}/${job}_%A_%a.err" \
    --partition=gpu \
    --account="${ACCOUNT}" \
    --gres=gpu:1 \
    --constraint="a40|a100|v100" \
    --cpus-per-task=8 \
    --mem=64G \
    --time="03:00:00" \
    --array="0-3" \
    --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
TARGETS=(Art Clipart Product RealWorld)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "target=${TARGET} uniform"
python -u scripts_cls_revise/ablations/search_officehome_uniform_b15.py \
  --target "${TARGET}" \
  --budget-per-class 15 \
  --split-seed 0 \
  --search-seed 0 \
  --train-seed 0 \
  --search-output-root experiments_cls_revise/officehome_fusion_ablation_search \
  --final-subset-root data_cls_revise/source_subsets/officehome_fusion_ablation \
  --final-config-root configs_cls_revise/officehome_fusion_ablation \
  --final-output-root experiments_cls_revise/officehome_fusion_ablation \
  --dry-run
CFG="configs_cls_revise/officehome_fusion_ablation/${TARGET}/TAVO_Uniform/B15/split00/train_seed00.yaml"
python -u scripts_cls/train_cls.py --config "${CFG}"
'
}

jid1=$(submit_search targetcrit scripts_cls_revise/ablations/search_officehome_targetcrit_b15.py)
echo "OfficeHome TargetCriteria job: ${jid1}"
jid2=$(submit_search sourcecrit scripts_cls_revise/ablations/search_officehome_sourcecrit_b15.py)
echo "OfficeHome SourceCriteria job: ${jid2}"
jid3=$(submit_uniform)
echo "OfficeHome Uniform job: ${jid3}"
