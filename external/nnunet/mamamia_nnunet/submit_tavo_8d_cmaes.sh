#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$(basename "$(dirname "${SCRIPT_DIR}")")" == "nnunet" && "$(basename "$(dirname "$(dirname "${SCRIPT_DIR}")")")" == "external" ]]; then
    REPO_ROOT="$(cd -P "${SCRIPT_DIR}/../../.." && pwd -P)"
else
    REPO_ROOT="$(cd -P "${SCRIPT_DIR}/../.." && pwd -P)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"

SUBMIT=0
TARGETS=(NACT ISPY1 DUKE ISPY2)
BUDGETS=(50 150 250)
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

GENERATIONS="${GENERATIONS:-12}"
POPSIZE="${POPSIZE:-20}"
MU="${MU:-10}"
N_STEPS="${N_STEPS:-1000}"
REFINE_STEPS="${REFINE_STEPS:-2000}"
REFINE_TOPK="${REFINE_TOPK:-8}"
SEEDS_EVAL="${SEEDS_EVAL:-0}"
SEEDS_REFINE="${SEEDS_REFINE:-0,1,2}"
BEST_K="${BEST_K:-4}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-1e-3}"
CROP_SIZE="${CROP_SIZE:-256}"
SEED="${SEED:-42}"
SCORE_ONLY=0

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-YOUR_SLURM_ACCOUNT}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100|a40|l40s|v100}"
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
SBATCH_MEM="${SBATCH_MEM:-48G}"
SBATCH_CPUS="${SBATCH_CPUS:-8}"
CONDA_ENV="${CONDA_ENV:-mamamia_nnunet}"

usage() {
    cat <<'EOF'
Usage: submit_tavo_8d_cmaes.sh [--submit] [--score-only] [--targets T1 ...] [--budgets B1 ...]

Submits paper-style 8D TAVO CMA-ES jobs for MAMAMIA. This is the BraTS-style
algorithmic path:
  methods: rds less orient craig gradmatch kmeans kcenter diversity
  score normalization: tie-aware rank normalization
  simplex: clip + renormalize
  fitness: target-val median proxy Dice after candidate training on selected
           source plus target-train cases

Defaults intentionally match the BraTS 8D CMA scripts:
  generations/popsize/mu: 12 / 20 / 10
  n-steps/refine-steps/refine-topk: 1000 / 2000 / 8

Use environment overrides for cheaper smoke runs:
  GENERATIONS POPSIZE MU N_STEPS REFINE_STEPS REFINE_TOPK SEEDS_EVAL SEEDS_REFINE
  RUN_TAG SBATCH_* CONDA_ENV PROJECT_ROOT
EOF
}

normalize_target() {
    case "$(echo "$1" | tr '[:lower:]' '[:upper:]' | tr -d '-')" in
        NACT) echo NACT ;;
        ISPY1) echo ISPY1 ;;
        DUKE) echo DUKE ;;
        ISPY2) echo ISPY2 ;;
        *) return 1 ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)
            SUBMIT=1
            shift
            ;;
        --score-only)
            SCORE_ONLY=1
            shift
            ;;
        --targets|--target)
            shift
            TARGETS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                TARGETS+=("$(normalize_target "$1")")
                shift
            done
            ;;
        --budgets|--budget)
            shift
            BUDGETS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                BUDGETS+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "${REPO_ROOT}/logs/mamamia/tavo_8d_cmaes"
MANIFEST="${REPO_ROOT}/logs/mamamia/tavo_8d_cmaes/manifest_${RUN_TAG}.tsv"

echo "SUBMIT=${SUBMIT} SCORE_ONLY=${SCORE_ONLY} RUN_TAG=${RUN_TAG}"
echo "Targets: ${TARGETS[*]}"
echo "Budgets: ${BUDGETS[*]}"
echo "8D CMA: generations=${GENERATIONS} popsize=${POPSIZE} mu=${MU} n_steps=${N_STEPS} refine_steps=${REFINE_STEPS}"
echo "Slurm: partition=${SBATCH_PARTITION} constraint=${SBATCH_CONSTRAINT} time=${SBATCH_TIME}"
echo -e "job_id\ttarget\tbudget\toutput_dir\tlog_out\tlog_err" > "${MANIFEST}"

for target in "${TARGETS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        case "${budget}" in
            ''|*[!0-9]*)
                echo "Invalid budget: ${budget}" >&2
                exit 1
                ;;
        esac
        out_dir="${PROJECT_ROOT}/outputs/meta/tavo_8d_cmaes_${RUN_TAG}_${target}_b${budget}"
        log_prefix="${REPO_ROOT}/logs/mamamia/tavo_8d_cmaes/${RUN_TAG}_${target}_b${budget}"
        log_out="${log_prefix}_%j.out"
        log_err="${log_prefix}_%j.err"
        job_name="mamamia_${target}_tavo8d_b${budget}"

        score_arg=""
        if [[ "${SCORE_ONLY}" == "1" ]]; then
            score_arg="--score-only"
        fi

        cmd=$(cat <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
module purge
module load gcc/13.3.0 cuda/12.6.3 cudnn/8.9.7.29-12-cuda conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/run_tavo_8d_cmaes.py" \
  --target "${target}" \
  --budget "${budget}" \
  --project-root "${PROJECT_ROOT}" \
  --output-dir "${out_dir}" \
  --generations "${GENERATIONS}" \
  --popsize "${POPSIZE}" \
  --mu "${MU}" \
  --n-steps "${N_STEPS}" \
  --refine-steps "${REFINE_STEPS}" \
  --refine-topk "${REFINE_TOPK}" \
  --seeds-eval "${SEEDS_EVAL}" \
  --seeds-refine "${SEEDS_REFINE}" \
  --best-k "${BEST_K}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --crop-size "${CROP_SIZE}" \
  --seed "${SEED}" \
  ${score_arg}
EOF
)
        if [[ "${SUBMIT}" == "1" ]]; then
            job_id="$(sbatch --parsable \
                --account="${SBATCH_ACCOUNT}" \
                --partition="${SBATCH_PARTITION}" \
                --nodes=1 \
                --ntasks=1 \
                --cpus-per-task="${SBATCH_CPUS}" \
                --gpus-per-task=1 \
                --constraint="${SBATCH_CONSTRAINT}" \
                --mem="${SBATCH_MEM}" \
                --time="${SBATCH_TIME}" \
                --job-name="${job_name}" \
                --output="${log_out}" \
                --error="${log_err}" \
                --wrap="${cmd}")"
            echo -e "${job_id}\t${target}\t${budget}\t${out_dir}\t${log_out//%j/${job_id}}\t${log_err//%j/${job_id}}" >> "${MANIFEST}"
            echo "Submitted ${target} budget ${budget}: ${job_id}"
        else
            echo
            echo "Would submit ${target} budget ${budget}"
            echo "  output: ${out_dir}"
            echo "  log: ${log_out}"
            printf '%s\n' "${cmd}" | sed 's/^/    /'
        fi
    done
done

echo
echo "Manifest: ${MANIFEST}"
