#!/bin/bash
set -Eeuo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/scripts/mamamia_nnunet" ]]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
    SCRIPT_DIR="${REPO_ROOT}/scripts/mamamia_nnunet"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ "$(basename "$(dirname "${SCRIPT_DIR}")")" == "nnunet" && "$(basename "$(dirname "$(dirname "${SCRIPT_DIR}")")")" == "external" ]]; then
        REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    else
        REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    fi
fi

TARGET="${1:?usage: post_tavo_job.sh TARGET EXPERIMENT [attempt]}"
EXPERIMENT="${2:?usage: post_tavo_job.sh TARGET EXPERIMENT [attempt]}"
ATTEMPT="${3:-1}"

MAX_RESUBMIT="${MAX_RESUBMIT:-2}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-YOUR_SLURM_ACCOUNT}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100|a40|l40s|v100}"
SBATCH_TIME="${SBATCH_TIME:-07:30:00}"
POST_PARTITION="${POST_PARTITION:-main}"
POST_TIME="${POST_TIME:-00:15:00}"
LOCK_FILE="${LOCK_FILE:-${REPO_ROOT}/logs/mamamia/tavo_collect.lock}"
PUSH_REMOTE="${PUSH_REMOTE:-origin}"
PUSH_BRANCH="${PUSH_BRANCH:-main}"
POST_DRY_RUN="${POST_DRY_RUN:-0}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

summary_exists() {
    SCRIPT_DIR="${SCRIPT_DIR}" TARGET="${TARGET}" EXPERIMENT="${EXPERIMENT}" python - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["SCRIPT_DIR"])

from collect_results import read_foreground_dice, summary_path
from core import project_root

path = summary_path(project_root(), os.environ["TARGET"], os.environ["EXPERIMENT"])
value = read_foreground_dice(path)
if value is None:
    raise SystemExit(1)
print(f"{path} Dice={value:.4f}")
PY
}

collect_and_push_reports() {
    mkdir -p "$(dirname "${LOCK_FILE}")"
    flock "${LOCK_FILE}" bash -lc "
        set -Eeuo pipefail
        cd '${REPO_ROOT}'
        python '${SCRIPT_DIR}/collect_results.py' \
            --csv reports/mamamia_nnunet_lodo_results.csv \
            --markdown reports/mamamia_nnunet_lodo_results.md
        if [[ '${POST_DRY_RUN}' == '1' ]]; then
            echo dry_run_no_report_commit
            exit 0
        fi
        git add reports/mamamia_nnunet_lodo_results.csv reports/mamamia_nnunet_lodo_results.md
        if git diff --cached --quiet; then
            echo no_report_changes
        else
            git commit -m 'Update MAMAMIA nnUNet TAVO results'
            git push '${PUSH_REMOTE}' '${PUSH_BRANCH}'
        fi
    "
}

active_matching_training_job() {
    local prefix="mamamia_${TARGET}_${EXPERIMENT}"
    squeue -u "${USER}" -h -o '%j|%T' | awk -F'|' -v prefix="${prefix}" '
        index($1, prefix) == 1 && $1 !~ /collect|post/ && ($2 == "PENDING" || $2 == "RUNNING") {
            print $0
        }
    '
}

resubmit_training() {
    local submit_output
    local new_job_id
    local next_attempt=$((ATTEMPT + 1))
    local train_job_name="mamamia_${TARGET}_${EXPERIMENT}_resume${ATTEMPT}"
    local post_job_name="mamamia_post_${TARGET}_${EXPERIMENT}_${next_attempt}"

    if [[ "${POST_DRY_RUN}" == "1" ]]; then
        log "DRY RUN: would submit ${train_job_name} and dependent ${post_job_name}"
        return 0
    fi

    submit_output="$(
        cd "${REPO_ROOT}" && sbatch \
            --account="${SBATCH_ACCOUNT}" \
            --partition="${SBATCH_PARTITION}" \
            --time="${SBATCH_TIME}" \
            --constraint="${SBATCH_CONSTRAINT}" \
            --job-name="${train_job_name}" \
            --export="ALL,CLEAN_PREPROCESSED_ON_SUCCESS=0,CLEAN_PREPROCESSED_ON_ERROR=0" \
            "${SCRIPT_DIR}/run_one.sh" "${TARGET}" "${EXPERIMENT}"
    )"
    log "${submit_output}"
    new_job_id="$(awk '{print $4}' <<< "${submit_output}")"

    cd "${REPO_ROOT}"
    sbatch \
        --account="${SBATCH_ACCOUNT}" \
        --partition="${POST_PARTITION}" \
        --time="${POST_TIME}" \
        --dependency="afterany:${new_job_id}" \
        --job-name="${post_job_name}" \
        --output="logs/mamamia/%x-%j.out" \
        "${SCRIPT_DIR}/post_tavo_job.sh" "${TARGET}" "${EXPERIMENT}" "${next_attempt}"
}

main() {
    cd "${REPO_ROOT}"
    log "Post-job check for ${TARGET}/${EXPERIMENT}, attempt ${ATTEMPT}/${MAX_RESUBMIT}"

    collect_and_push_reports

    if summary_exists; then
        log "Summary exists for ${TARGET}/${EXPERIMENT}; no resubmit needed."
        exit 0
    fi

    log "Summary is missing for ${TARGET}/${EXPERIMENT}."
    if active_matching_training_job | grep -q .; then
        log "A matching training job is already queued or running:"
        active_matching_training_job
        exit 0
    fi

    if (( ATTEMPT > MAX_RESUBMIT )); then
        log "Reached MAX_RESUBMIT=${MAX_RESUBMIT}; leaving ${TARGET}/${EXPERIMENT} for manual inspection."
        exit 1
    fi

    log "Resubmitting ${TARGET}/${EXPERIMENT}; run_one.sh will resume from checkpoint_latest if present."
    resubmit_training
}

main "$@"
