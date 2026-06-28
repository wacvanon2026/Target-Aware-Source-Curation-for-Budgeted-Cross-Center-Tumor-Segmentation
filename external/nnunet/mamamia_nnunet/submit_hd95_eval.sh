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
EXPERIMENTS=(all)
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-YOUR_SLURM_ACCOUNT}"
SBATCH_COLLECT_ACCOUNT="${SBATCH_COLLECT_ACCOUNT:-YOUR_SLURM_ACCOUNT}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-p100}"
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_MEM="${SBATCH_MEM:-24G}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
CONDA_ENV="${CONDA_ENV:-mamamia_nnunet}"

usage() {
    cat <<'EOF'
Usage: submit_hd95_eval.sh [--submit] [--targets T1 T2 ...] [experiments...]

Computes foreground HD95 from existing test_preds/test_gt NIfTI files and writes
summary_hd95.json next to each existing Dice summary.json. One Slurm job is
submitted per target, followed by one dependent collector job.

Defaults:
  targets: NACT ISPY1 DUKE ISPY2
  experiments: all
  Slurm: gpu partition, p100 constraint, 12h

Environment overrides:
  RUN_TAG PROJECT_ROOT CONDA_ENV
  SBATCH_ACCOUNT SBATCH_COLLECT_ACCOUNT SBATCH_PARTITION SBATCH_CONSTRAINT SBATCH_TIME SBATCH_MEM SBATCH_CPUS
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
                if normalized_target="$(normalize_target "$1")"; then
                    TARGETS+=("${normalized_target}")
                    shift
                else
                    break
                fi
            done
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXPERIMENTS+=("$1")
            shift
            ;;
    esac
done

if [[ "${#EXPERIMENTS[@]}" -gt 1 && "${EXPERIMENTS[0]}" == "all" ]]; then
    EXPERIMENTS=("${EXPERIMENTS[@]:1}")
fi

mkdir -p "${REPO_ROOT}/logs/mamamia/hd95"
MANIFEST="${REPO_ROOT}/logs/mamamia/hd95/manifest_${RUN_TAG}.tsv"
echo -e "job_id\ttarget\tlog_out\tlog_err" > "${MANIFEST}"

echo "SUBMIT=${SUBMIT} RUN_TAG=${RUN_TAG}"
echo "Targets: ${TARGETS[*]}"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Slurm: account=${SBATCH_ACCOUNT} partition=${SBATCH_PARTITION} constraint=${SBATCH_CONSTRAINT} time=${SBATCH_TIME}"

job_ids=()
for target in "${TARGETS[@]}"; do
    log_prefix="${REPO_ROOT}/logs/mamamia/hd95/${RUN_TAG}_${target}"
    log_out="${log_prefix}_%j.out"
    log_err="${log_prefix}_%j.err"
    cmd=$(cat <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
module purge
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/compute_hd95.py" "${EXPERIMENTS[@]}" --targets "${target}" --project-root "${PROJECT_ROOT}"
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
            --job-name="mamamia_hd95_${target}" \
            --output="${log_out}" \
            --error="${log_err}" \
            --wrap="${cmd}")"
        job_ids+=("${job_id}")
        echo -e "${job_id}\t${target}\t${log_out//%j/${job_id}}\t${log_err//%j/${job_id}}" >> "${MANIFEST}"
        echo "Submitted ${target}: ${job_id}"
    else
        echo
        echo "Would submit ${target}:"
        printf '%s\n' "${cmd}" | sed 's/^/    /'
    fi
done

if [[ "${SUBMIT}" == "1" && "${#job_ids[@]}" -gt 0 ]]; then
    deps="$(IFS=:; echo "${job_ids[*]}")"
    collect_log="${REPO_ROOT}/logs/mamamia/hd95/${RUN_TAG}_collect_%j.out"
    collect_err="${REPO_ROOT}/logs/mamamia/hd95/${RUN_TAG}_collect_%j.err"
    collect_cmd=$(cat <<EOF
set -euo pipefail
module purge
module load conda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/collect_results.py" --metric HD95 --project-root "${PROJECT_ROOT}" --csv reports/mamamia_nnunet_lodo_hd95_results.csv --markdown reports/mamamia_nnunet_lodo_hd95_results.md
EOF
)
    collect_job="$(sbatch --parsable \
        --account="${SBATCH_COLLECT_ACCOUNT}" \
        --partition=main \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=2G \
        --time=01:00:00 \
        --dependency="afterany:${deps}" \
        --job-name="mamamia_hd95_collect" \
        --output="${collect_log}" \
        --error="${collect_err}" \
        --wrap="${collect_cmd}")"
    echo -e "${collect_job}\tCOLLECT\t${collect_log//%j/${collect_job}}\t${collect_err//%j/${collect_job}}" >> "${MANIFEST}"
    echo "Submitted collector: ${collect_job}"
fi

echo "Manifest: ${MANIFEST}"
