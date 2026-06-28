#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MAMAMIA_REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd -P "${MAMAMIA_REPO_ROOT}" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts/mamamia_nnunet" ]]; then
    REPO_ROOT="$(cd -P "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
    SCRIPT_DIR_FALLBACK="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    if [[ "$(basename "$(dirname "${SCRIPT_DIR_FALLBACK}")")" == "nnunet" && "$(basename "$(dirname "$(dirname "${SCRIPT_DIR_FALLBACK}")")")" == "external" ]]; then
        REPO_ROOT="$(cd -P "${SCRIPT_DIR_FALLBACK}/../../.." && pwd -P)"
    else
        REPO_ROOT="$(cd -P "${SCRIPT_DIR_FALLBACK}/../.." && pwd -P)"
    fi
fi
if [[ -d "${REPO_ROOT}/external/nnunet/mamamia_nnunet" ]]; then
    SCRIPT_DIR="${REPO_ROOT}/external/nnunet/mamamia_nnunet"
else
    SCRIPT_DIR="${REPO_ROOT}/scripts/mamamia_nnunet"
fi

MANIFEST="${1:-${REAL_TAVO_MANIFEST:-}}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
CONDA_ENV="${CONDA_ENV:-mamamia_nnunet}"

GENERATIONS="${GENERATIONS:-20}"
POPSIZE="${POPSIZE:-8}"
N_STEPS="${N_STEPS:-500}"
FITNESS="${FITNESS:-median_val}"
BEST_K="${BEST_K:-4}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-1e-3}"
CROP_SIZE="${CROP_SIZE:-256}"
SEED="${SEED:-42}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-YOUR_SLURM_ACCOUNT}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100|a40|l40s|v100}"
SBATCH_TIME="${SBATCH_TIME:-2-00:00:00}"
SBATCH_MEM="${SBATCH_MEM:-32G}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
RUN_TAG="${RUN_TAG:-$(basename "${MANIFEST:-manifest}" .tsv)_rescue_$(date +%Y%m%d_%H%M%S)}"
FINALIZE_ON_COMPLETE="${FINALIZE_ON_COMPLETE:-0}"

if [[ -z "${MANIFEST}" ]]; then
    echo "Usage: rescue_real_tavo_cmaes.sh MANIFEST.tsv" >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing real TAVO CMA-ES manifest: ${MANIFEST}" >&2
    exit 1
fi

target_root() {
    case "$1" in
        NACT) echo "${PROJECT_ROOT}/mamamia_clean" ;;
        ISPY1) echo "${PROJECT_ROOT}/mamamia_ispy1" ;;
        DUKE) echo "${PROJECT_ROOT}/mamamia_duke" ;;
        ISPY2) echo "${PROJECT_ROOT}/mamamia_ispy2" ;;
        *) return 1 ;;
    esac
}

mapfile -t ROWS < <(awk -F'\t' 'NR > 1 && $1 ~ /^[0-9]+$/ {print}' "${MANIFEST}")
if [[ "${#ROWS[@]}" -eq 0 ]]; then
    echo "No job rows in ${MANIFEST}" >&2
    exit 1
fi

JOB_IDS=()
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r job_id _target _budget _out_dir _log_out _log_err <<< "${row}"
    JOB_IDS+=("${job_id}")
done
ids_csv="$(IFS=,; echo "${JOB_IDS[*]}")"

state_for() {
    local job_id="$1"
    printf '%s\n' "${STATES}" | awk -v id="${job_id}" '$1 == id {print $2; found=1} END {if (!found) print "UNKNOWN"}'
}

is_live_state() {
    case "$1" in
        COMPLETED|RUNNING|PENDING|CONFIGURING|COMPLETING) return 0 ;;
        *) return 1 ;;
    esac
}

submit_retry() {
    local target="$1"
    local budget="$2"
    local abs_out_dir="$3"

    local root runner out_dir log_prefix log_out log_err job_name cmd job_id
    root="$(target_root "${target}")"
    runner="${root}/scripts/python/run_meta_cmaes_nnunet_proxy.py"
    if [[ ! -f "${runner}" ]]; then
        echo "Missing runner for ${target}: ${runner}" >&2
        exit 1
    fi

    case "${abs_out_dir}" in
        "${root}"/*) out_dir="${abs_out_dir#${root}/}" ;;
        *) out_dir="${abs_out_dir}" ;;
    esac

    log_prefix="${REPO_ROOT}/logs/mamamia/real_tavo_cmaes/${RUN_TAG}_${target}_b${budget}"
    log_out="${log_prefix}_%j.out"
    log_err="${log_prefix}_%j.err"
    job_name="mamamia_${target}_real_tavo_b${budget}_rescue"

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

    echo -e "${job_id}\t${target}\t${budget}\t${abs_out_dir}\t${log_out//%j/${job_id}}\t${log_err//%j/${job_id}}"
    echo "Submitted rescue retry for ${target} budget ${budget}: ${job_id}" >&2
}

echo "[$(date)] rescue monitor watching ${#JOB_IDS[@]} real TAVO CMA-ES jobs from ${MANIFEST}"
while true; do
    STATES="$(
        sacct -X -j "${ids_csv}" --format=JobID,State -n -P |
        awk -F'|' 'NF >= 2 {split($2, a, "+"); state[$1]=a[1]} END {for (id in state) print id, state[id]}'
    )"

    completed=0
    bad_count=0
    live_count=0
    for row in "${ROWS[@]}"; do
        IFS=$'\t' read -r job_id _target _budget _out_dir _log_out _log_err <<< "${row}"
        state="$(state_for "${job_id}")"
        if [[ "${state}" == "COMPLETED" ]]; then
            completed=$((completed + 1))
        elif is_live_state "${state}"; then
            live_count=$((live_count + 1))
        else
            bad_count=$((bad_count + 1))
        fi
    done

    echo "[$(date)] rescue monitor: completed=${completed}/${#ROWS[@]} live=${live_count} bad=${bad_count}"

    if [[ "${completed}" -eq "${#ROWS[@]}" ]]; then
        if [[ "${FINALIZE_ON_COMPLETE}" == "1" ]]; then
            echo "[$(date)] rescued real TAVO manifest completed; launching final materialization monitor."
            PROJECT_ROOT="${PROJECT_ROOT}" \
            CONDA_ENV="${CONDA_ENV}" \
            SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT}" \
            "${SCRIPT_DIR}/monitor_real_tavo_cmaes_then_train.sh" "${MANIFEST}"
        else
            echo "[$(date)] original real TAVO jobs completed; existing monitor can finalize."
        fi
        exit 0
    fi

    if [[ "${bad_count}" -gt 0 ]]; then
        retry_manifest="${REPO_ROOT}/logs/mamamia/real_tavo_cmaes/manifest_${RUN_TAG}.tsv"
        echo -e "job_id\ttarget\tbudget\toutput_dir\tlog_out\tlog_err" > "${retry_manifest}"
        for row in "${ROWS[@]}"; do
            IFS=$'\t' read -r job_id target budget out_dir log_out log_err <<< "${row}"
            state="$(state_for "${job_id}")"
            if [[ "${state}" == "COMPLETED" ]]; then
                printf '%s\n' "${row}" >> "${retry_manifest}"
            elif is_live_state "${state}"; then
                printf '%s\n' "${row}" >> "${retry_manifest}"
            else
                submit_retry "${target}" "${budget}" "${out_dir}" >> "${retry_manifest}"
            fi
        done

        echo "[$(date)] rescue retries submitted; continuing rescue monitoring with ${retry_manifest}"
        PROJECT_ROOT="${PROJECT_ROOT}" \
        CONDA_ENV="${CONDA_ENV}" \
        SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT}" \
        FINALIZE_ON_COMPLETE=1 \
        exec "${SCRIPT_DIR}/rescue_real_tavo_cmaes.sh" "${retry_manifest}"
    fi

    sleep "${POLL_SECONDS}"
done
