#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -P "${SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-.}"

SUBMIT=0
TARGETS=(NACT ISPY1 DUKE ISPY2)
BUDGETS=(50 150 250)
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

GENERATIONS="${GENERATIONS:-20}"
POPSIZE="${POPSIZE:-8}"
N_STEPS="${N_STEPS:-500}"
FITNESS="${FITNESS:-median_val}"
BEST_K="${BEST_K:-4}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-1e-3}"
CROP_SIZE="${CROP_SIZE:-256}"
SEED="${SEED:-42}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-karimire_1837}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100|a40|l40s|v100}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
CONDA_ENV="${CONDA_ENV:-data_selection_3_10}"

usage() {
    cat <<'EOF'
Usage: submit_real_tavo_cmaes.sh [--submit] [--targets T1 T2 ...] [--budgets B1 B2 ...]

Runs the actual TAVO CMA-ES nnUNet-proxy search, not final training on an
already-materialized TAVO list. Outputs are written to fresh target-local dirs:
  <target-root>/outputs/meta/cmaes_nnunet500_real_<RUN_TAG>_b<BUDGET>

Defaults:
  targets: NACT ISPY1 DUKE ISPY2
  budgets: 50 150 250
  generations/popsize/n-steps: 20 / 8 / 500

Environment overrides:
  RUN_TAG GENERATIONS POPSIZE N_STEPS FITNESS BEST_K BATCH_SIZE LR CROP_SIZE SEED
  SBATCH_ACCOUNT SBATCH_PARTITION SBATCH_CONSTRAINT SBATCH_TIME SBATCH_MEM SBATCH_CPUS
  CONDA_ENV PROJECT_ROOT
EOF
}

normalize_target() {
    case "$(echo "$1" | tr '[:lower:]' '[:upper:]')" in
        NACT) echo NACT ;;
        ISPY1|I-SPY1) echo ISPY1 ;;
        DUKE) echo DUKE ;;
        ISPY2|I-SPY2) echo ISPY2 ;;
        *) return 1 ;;
    esac
}

target_root() {
    case "$1" in
        NACT) echo "${PROJECT_ROOT}/mamamia_clean" ;;
        ISPY1) echo "${PROJECT_ROOT}/mamamia_ispy1" ;;
        DUKE) echo "${PROJECT_ROOT}/mamamia_duke" ;;
        ISPY2) echo "${PROJECT_ROOT}/mamamia_ispy2" ;;
        *) return 1 ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)
            SUBMIT=1
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

mkdir -p "${REPO_ROOT}/logs/mamamia" "${REPO_ROOT}/logs/mamamia/real_tavo_cmaes"
MANIFEST="${REPO_ROOT}/logs/mamamia/real_tavo_cmaes/manifest_${RUN_TAG}.tsv"

echo "SUBMIT=${SUBMIT} RUN_TAG=${RUN_TAG}"
echo "Targets: ${TARGETS[*]}"
echo "Budgets: ${BUDGETS[*]}"
echo "CMA-ES: generations=${GENERATIONS} popsize=${POPSIZE} n_steps=${N_STEPS} fitness=${FITNESS} best_k=${BEST_K}"
echo "Slurm: partition=${SBATCH_PARTITION} constraint=${SBATCH_CONSTRAINT} time=${SBATCH_TIME}"
echo -e "job_id\ttarget\tbudget\toutput_dir\tlog_out\tlog_err" > "${MANIFEST}"

for target in "${TARGETS[@]}"; do
    root="$(target_root "${target}")"
    runner="${root}/scripts/python/run_meta_cmaes_nnunet_proxy.py"
    if [[ ! -f "${runner}" ]]; then
        echo "Missing runner for ${target}: ${runner}" >&2
        exit 1
    fi
    for budget in "${BUDGETS[@]}"; do
        case "${budget}" in
            ''|*[!0-9]*)
                echo "Invalid budget: ${budget}" >&2
                exit 1
                ;;
        esac
        out_dir="outputs/meta/cmaes_nnunet500_real_${RUN_TAG}_b${budget}"
        abs_out_dir="${root}/${out_dir}"
        log_prefix="${REPO_ROOT}/logs/mamamia/real_tavo_cmaes/${RUN_TAG}_${target}_b${budget}"
        log_out="${log_prefix}_%j.out"
        log_err="${log_prefix}_%j.err"
        job_name="mamamia_${target}_real_tavo_b${budget}"
        cmd=$(cat <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
module purge
module load gcc/13.3.0 cuda/12.6.3 cudnn/8.9.7.29-12-cuda conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${root}"
mkdir -p "${out_dir}"
python "${runner}" --budget "${budget}" --generations "${GENERATIONS}" --popsize "${POPSIZE}" --n-steps "${N_STEPS}" --fitness "${FITNESS}" --best-k "${BEST_K}" --seed "${SEED}" --batch-size "${BATCH_SIZE}" --lr "${LR}" --crop-size "${CROP_SIZE}" --output-dir "${out_dir}"
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
            echo -e "${job_id}\t${target}\t${budget}\t${abs_out_dir}\t${log_out//%j/${job_id}}\t${log_err//%j/${job_id}}" >> "${MANIFEST}"
            echo "Submitted ${target} budget ${budget}: ${job_id}"
        else
            echo
            echo "Would submit ${target} budget ${budget}"
            echo "  output: ${abs_out_dir}"
            echo "  log: ${log_out}"
            echo "  command:"
            printf '%s\n' "${cmd}" | sed 's/^/    /'
        fi
    done
done

echo
echo "Manifest: ${MANIFEST}"
