#!/bin/bash
set -euo pipefail

ROOT="external/efficientvit"
cd "${ROOT}"
source ~/envs/brain310/bin/activate

ACCOUNT="${ACCOUNT:-YOUR_SLURM_ACCOUNT}"
JOB_NAME="${JOB_NAME:-oh_tavo_b25_f3}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_cls_revise/officehome_tavo_b25_formal3seed_$(date +%Y%m%d)}"
TIME_LIMIT="${TIME_LIMIT:-20:00:00}"

POPSIZE="${POPSIZE:-20}"
MU="${MU:-10}"
N_GEN="${N_GEN:-12}"
EVAL_EPOCHS="${EVAL_EPOCHS:-5}"
REFINE_EPOCHS="${REFINE_EPOCHS:-15}"
REFINE_TOPK="${REFINE_TOPK:-8}"
TAVO_EVAL_SEEDS="${TAVO_EVAL_SEEDS:-0,1,2}"
TAVO_REFINE_SEEDS="${TAVO_REFINE_SEEDS:-0,1,2}"
TAVO_FINAL_TRAIN_SEEDS="${TAVO_FINAL_TRAIN_SEEDS:-0,1,2}"
RUN_FINAL_TRAIN="${RUN_FINAL_TRAIN:-1}"
MAX_EVALS="${MAX_EVALS:-0}"

mkdir -p "${LOG_DIR}"

sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --output="${LOG_DIR}/%A_%a.out" \
  --error="${LOG_DIR}/%A_%a.err" \
  --partition=gpu \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --constraint="a40|a100|v100" \
  --cpus-per-task=8 \
  --mem=64G \
  --time="${TIME_LIMIT}" \
  --array="0-3" \
  --export=ALL,POPSIZE="${POPSIZE}",MU="${MU}",N_GEN="${N_GEN}",EVAL_EPOCHS="${EVAL_EPOCHS}",REFINE_EPOCHS="${REFINE_EPOCHS}",REFINE_TOPK="${REFINE_TOPK}",TAVO_EVAL_SEEDS="${TAVO_EVAL_SEEDS}",TAVO_REFINE_SEEDS="${TAVO_REFINE_SEEDS}",TAVO_FINAL_TRAIN_SEEDS="${TAVO_FINAL_TRAIN_SEEDS}",RUN_FINAL_TRAIN="${RUN_FINAL_TRAIN}",MAX_EVALS="${MAX_EVALS}" \
  --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
TARGETS=(Art Clipart Product RealWorld)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "target=${TARGET}"
echo "formal3seed TAVO: popsize=${POPSIZE} mu=${MU} n_gen=${N_GEN} eval_epochs=${EVAL_EPOCHS} refine_epochs=${REFINE_EPOCHS}"
echo "eval_seeds=${TAVO_EVAL_SEEDS} refine_seeds=${TAVO_REFINE_SEEDS} final_train_seeds=${TAVO_FINAL_TRAIN_SEEDS}"
ARGS=(
  --target "${TARGET}"
  --budget-per-class 25
  --split-seed 0
  --search-seed 0
  --train-seed 0
  --eval-seeds "${TAVO_EVAL_SEEDS}"
  --refine-seeds "${TAVO_REFINE_SEEDS}"
  --final-train-seeds "${TAVO_FINAL_TRAIN_SEEDS}"
  --search-output-root experiments_cls_revise/officehome_tavo_search_formal3seed
  --final-subset-root data_cls_revise/source_subsets/officehome_tavo_formal3seed
  --final-config-root configs_cls_revise/officehome_tavo_formal3seed
  --final-output-root experiments_cls_revise/officehome_tavo_formal3seed
  --popsize "${POPSIZE}"
  --mu "${MU}"
  --n-gen "${N_GEN}"
  --eval-epochs "${EVAL_EPOCHS}"
  --refine-epochs "${REFINE_EPOCHS}"
  --refine-topk "${REFINE_TOPK}"
  --max-evals "${MAX_EVALS}"
)
if [[ "${RUN_FINAL_TRAIN}" == "1" ]]; then
  ARGS+=(--run-final-train)
fi
python -u scripts_cls_revise/search_officehome_tavo_8d.py "${ARGS[@]}"
'
