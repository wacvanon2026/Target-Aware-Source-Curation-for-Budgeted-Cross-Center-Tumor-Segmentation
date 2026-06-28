#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${MAMAMIA_REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd -P "${MAMAMIA_REPO_ROOT}" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts/mamamia_nnunet" ]]; then
    REPO_ROOT="$(cd -P "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
    SCRIPT_DIR_FALLBACK="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    REPO_ROOT="$(cd -P "${SCRIPT_DIR_FALLBACK}/../.." && pwd -P)"
fi
SCRIPT_DIR="${REPO_ROOT}/scripts/mamamia_nnunet"

MANIFEST="${1:-${REAL_TAVO_MANIFEST:-}}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"
CONDA_ENV="${CONDA_ENV:-data_selection_3_10}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100|a40|l40s|v100}"

if [[ -z "${MANIFEST}" ]]; then
    echo "Usage: monitor_real_tavo_cmaes_then_train.sh MANIFEST.tsv" >&2
    exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing real TAVO CMA-ES manifest: ${MANIFEST}" >&2
    exit 1
fi

mapfile -t JOB_IDS < <(awk -F'\t' 'NR > 1 && $1 ~ /^[0-9]+$/ {print $1}' "${MANIFEST}")
if [[ "${#JOB_IDS[@]}" -eq 0 ]]; then
    echo "No job ids in ${MANIFEST}" >&2
    exit 1
fi

ids_csv="$(IFS=,; echo "${JOB_IDS[*]}")"

while true; do
    states="$(
        sacct -X -j "${ids_csv}" --format=JobID,State -n -P |
        awk -F'|' 'NF >= 2 {split($2, a, "+"); state[$1]=a[1]} END {for (id in state) print id, state[id]}'
    )"
    completed="$(printf '%s\n' "${states}" | awk '$2=="COMPLETED"{c++} END{print c+0}')"
    bad="$(
        printf '%s\n' "${states}" |
        awk '$2!="COMPLETED" && $2!="RUNNING" && $2!="PENDING" && $2!="CONFIGURING" && $2!="COMPLETING" {print}'
    )"
    echo "[$(date)] real TAVO CMA-ES monitor: completed=${completed}/${#JOB_IDS[@]} bad=${bad:-none}"

    if [[ -n "${bad}" ]]; then
        echo "Refusing to materialize/submit final TAVO because a CMA-ES job has a bad state:" >&2
        printf '%s\n' "${bad}" >&2
        exit 1
    fi

    if [[ "${completed}" -eq "${#JOB_IDS[@]}" ]]; then
        break
    fi

    sleep "${POLL_SECONDS}"
done

echo "[$(date)] real TAVO CMA-ES complete; materializing fresh selections."
python "${SCRIPT_DIR}/materialize_method_selections.py" --methods tavo --budgets 50 150 250 --strict

RAW_DIR="${PROJECT_ROOT}/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_raw"
PREPROCESSED_DIR="${PROJECT_ROOT}/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_preprocessed"
RESULTS_DIR="${PROJECT_ROOT}/externals/MAMA-MIA/nnUNet/nnunetv2/nnUNet_results_scratch"

echo "[$(date)] removing old TAVO nnUNet raw/preprocessed/results/output directories."
for target in NACT ISPY1 DUKE ISPY2; do
    target_lc="$(echo "${target}" | tr '[:upper:]' '[:lower:]')"
    for exp in tavo50 tavo150 tavo250; do
        resolved="$(python "${SCRIPT_DIR}/resolve_experiment.py" --shell "${target}" "${exp}")"
        eval "${resolved}"
        rm -rf \
            "${RAW_DIR}/${DATASET_BASENAME}" \
            "${PREPROCESSED_DIR}/${DATASET_BASENAME}" \
            "${RESULTS_DIR}/${DATASET_BASENAME}" \
            "${PROJECT_ROOT}/outputs/tavo_mamamia_${target_lc}_nnunet_${exp}/repeat_01"
    done
done

echo "[$(date)] rebuilding TAVO nnUNet raw datasets from fresh CMA-ES selections."
python "${SCRIPT_DIR}/build_datasets.py" tavo --targets NACT ISPY1 DUKE ISPY2 --overwrite

echo "[$(date)] submitting fresh final TAVO training/eval rows."
PROJECT_ROOT="${PROJECT_ROOT}" \
CONDA_ENV="${CONDA_ENV}" \
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT}" \
"${SCRIPT_DIR}/submit_table2_subset.sh" --submit --skip-build --force tavo
