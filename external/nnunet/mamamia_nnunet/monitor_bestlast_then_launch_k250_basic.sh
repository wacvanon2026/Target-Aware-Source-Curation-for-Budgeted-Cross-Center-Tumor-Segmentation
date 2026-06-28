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

JOBS_FILE="${JOBS_FILE:-${REPO_ROOT}/logs/mamamia/bestlast_reeval_jobs_20260623_corrected.tsv}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
CONDA_ENV="${CONDA_ENV:-mamamia_nnunet}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100|a40|l40s|v100}"

if [[ ! -f "${JOBS_FILE}" ]]; then
    echo "Missing best-last jobs file: ${JOBS_FILE}" >&2
    exit 1
fi

mapfile -t JOB_IDS < <(cut -f1 "${JOBS_FILE}" | awk 'NF')
if [[ "${#JOB_IDS[@]}" -eq 0 ]]; then
    echo "No job ids in ${JOBS_FILE}" >&2
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
    echo "[$(date)] best-last monitor: completed=${completed}/${#JOB_IDS[@]} bad=${bad:-none}"

    if [[ -n "${bad}" ]]; then
        echo "Refusing to launch K250 basic methods because a best-last job has a bad state:" >&2
        printf '%s\n' "${bad}" >&2
        exit 1
    fi

    if [[ "${completed}" -eq "${#JOB_IDS[@]}" ]]; then
        echo "[$(date)] best-last complete; launching K250 basic methods only."
        PROJECT_ROOT="${PROJECT_ROOT}" \
        CONDA_ENV="${CONDA_ENV}" \
        SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT}" \
        "${SCRIPT_DIR}/submit_table2_subset.sh" --submit k250_methods
        exit 0
    fi

    sleep "${POLL_SECONDS}"
done
