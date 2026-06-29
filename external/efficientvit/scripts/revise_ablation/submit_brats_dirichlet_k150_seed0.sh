#!/bin/bash
set -euo pipefail

ROOT="external/efficientvit"
cd "${ROOT}"
source ~/envs/brain310/bin/activate

ACCOUNT="${ACCOUNT:-YOUR_SLURM_ACCOUNT}"
JOB_NAME="${JOB_NAME:-seg_dir_k150_s0}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/revise_ablation_brats_dirichlet_k150_seed0_$(date +%Y%m%d)}"
TIME_LIMIT="${TIME_LIMIT:-36:00:00}"
POPSIZE="${POPSIZE:-20}"
N_GEN="${N_GEN:-12}"
N_RANDOM="${N_RANDOM:-0}"
ITERS_EVAL="${ITERS_EVAL:-500}"
ITERS_REFINE="${ITERS_REFINE:-1500}"
REFINE_TOPK="${REFINE_TOPK:-8}"
SEEDS_EVAL="${SEEDS_EVAL:-0}"
SEEDS_REFINE="${SEEDS_REFINE:-0}"
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
  --mem=96G \
  --time="${TIME_LIMIT}" \
  --array="0-3" \
  --export=ALL,POPSIZE="${POPSIZE}",N_GEN="${N_GEN}",N_RANDOM="${N_RANDOM}",ITERS_EVAL="${ITERS_EVAL}",ITERS_REFINE="${ITERS_REFINE}",REFINE_TOPK="${REFINE_TOPK}",SEEDS_EVAL="${SEEDS_EVAL}",SEEDS_REFINE="${SEEDS_REFINE}",RUN_FINAL_TRAIN="${RUN_FINAL_TRAIN}" \
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
TARGETS=(C4 C5 TCGA_LGG TCGA_GBM)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "target=${TARGET} method=TAVO_Dirichlet_8D K150"
echo "Dirichlet budget: corners+uniform + ${POPSIZE}*${N_GEN}; n_random=${N_RANDOM}; iters_eval=${ITERS_EVAL}; iters_refine=${ITERS_REFINE}"
ARGS=(
  --target "${TARGET}"
  --train-seed 0
  --search-seed 0
  --seeds-eval "${SEEDS_EVAL}"
  --seeds-refine "${SEEDS_REFINE}"
  --popsize "${POPSIZE}"
  --n-gen "${N_GEN}"
  --n-random "${N_RANDOM}"
  --iters-eval "${ITERS_EVAL}"
  --iters-refine "${ITERS_REFINE}"
  --refine-topk "${REFINE_TOPK}"
)
if [[ "${RUN_FINAL_TRAIN}" == "1" ]]; then
  ARGS+=(--run-final-train)
fi
python -u scripts/revise_ablation/brats_dirichlet_ablation_k150.py "${ARGS[@]}"
'
