#!/bin/bash
set -euo pipefail

ROOT="external/efficientvit"
cd "${ROOT}"
source ~/envs/brain310/bin/activate

ACCOUNT="${ACCOUNT:-YOUR_SLURM_ACCOUNT}"
JOB_NAME="${JOB_NAME:-oh_dir_b15_s0}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_cls_revise/officehome_dirichlet_b15_seed0_$(date +%Y%m%d)}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
POPSIZE="${POPSIZE:-20}"
N_GEN="${N_GEN:-12}"
N_RANDOM="${N_RANDOM:-0}"
EVAL_EPOCHS="${EVAL_EPOCHS:-5}"
REFINE_EPOCHS="${REFINE_EPOCHS:-15}"
REFINE_TOPK="${REFINE_TOPK:-8}"
EVAL_SEEDS="${EVAL_SEEDS:-0}"
REFINE_SEEDS="${REFINE_SEEDS:-0}"
FINAL_TRAIN_SEEDS="${FINAL_TRAIN_SEEDS:-0}"
RUN_FINAL_TRAIN="${RUN_FINAL_TRAIN:-1}"
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
  --export=ALL,POPSIZE="${POPSIZE}",N_GEN="${N_GEN}",N_RANDOM="${N_RANDOM}",EVAL_EPOCHS="${EVAL_EPOCHS}",REFINE_EPOCHS="${REFINE_EPOCHS}",REFINE_TOPK="${REFINE_TOPK}",EVAL_SEEDS="${EVAL_SEEDS}",REFINE_SEEDS="${REFINE_SEEDS}",FINAL_TRAIN_SEEDS="${FINAL_TRAIN_SEEDS}",RUN_FINAL_TRAIN="${RUN_FINAL_TRAIN}" \
  --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
TARGETS=(Art Clipart Product RealWorld)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "target=${TARGET} method=TAVO_Dirichlet_8D B15"
echo "Dirichlet budget: corners+uniform + ${POPSIZE}*${N_GEN}; n_random=${N_RANDOM}; eval_epochs=${EVAL_EPOCHS}; refine_epochs=${REFINE_EPOCHS}"
ARGS=(
  --target "${TARGET}"
  --budget-per-class 15
  --split-seed 0
  --search-seed 0
  --train-seed 0
  --eval-seeds "${EVAL_SEEDS}"
  --refine-seeds "${REFINE_SEEDS}"
  --final-train-seeds "${FINAL_TRAIN_SEEDS}"
  --search-output-root experiments_cls_revise/officehome_tavo_search_dirichlet_b15
  --final-subset-root data_cls_revise/source_subsets/officehome_tavo_dirichlet_b15
  --final-config-root configs_cls_revise/officehome_tavo_dirichlet_b15
  --final-output-root experiments_cls_revise/officehome_tavo_dirichlet_b15
  --popsize "${POPSIZE}"
  --n-gen "${N_GEN}"
  --n-random "${N_RANDOM}"
  --eval-epochs "${EVAL_EPOCHS}"
  --refine-epochs "${REFINE_EPOCHS}"
  --refine-topk "${REFINE_TOPK}"
)
if [[ "${RUN_FINAL_TRAIN}" == "1" ]]; then
  ARGS+=(--run-final-train)
fi
python -u scripts_cls_revise/ablations/search_officehome_dirichlet_8d.py "${ARGS[@]}"
'
