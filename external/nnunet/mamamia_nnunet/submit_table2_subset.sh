#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$(dirname "${SCRIPT_DIR}")")" == "nnunet" && "$(basename "$(dirname "$(dirname "${SCRIPT_DIR}")")")" == "external" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SUBMIT=0
FORCE="${FORCE:-0}"
SKIP_BUILD=0
CHAIN=0
SBATCH_TIME="${SBATCH_TIME:-12:00:00}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-YOUR_SLURM_ACCOUNT}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100|a40|l40s|v100}"
TARGETS=(NACT ISPY1 DUKE ISPY2)
EXPERIMENTS=()

usage() {
    cat <<'EOF'
Usage: submit_table2_subset.sh [--targets T1 T2 ...] [--submit] [--force] [--skip-build] [--chain] [experiments...]

Targets:
  NACT ISPY1 DUKE ISPY2

Default experiments:
  target_only source_only target_full_source random50 random150 random250

Experiment groups:
  baselines basic_methods methods k50_methods k150_methods tavo all_methods all

Default mode creates/validates datasets and prints sbatch commands only.
Use --submit to actually submit training jobs.
Use --chain to submit jobs with afterany dependencies so only one runs at a time.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)
            SUBMIT=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --chain)
            CHAIN=1
            shift
            ;;
        --targets|--target)
            shift
            TARGETS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                target_arg="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
                case "${target_arg}" in
                    ALL)
                        TARGETS=(NACT ISPY1 DUKE ISPY2)
                        shift
                        ;;
                    NACT|DUKE)
                        TARGETS+=("${target_arg}")
                        shift
                        ;;
                    ISPY1|I-SPY1)
                        TARGETS+=("ISPY1")
                        shift
                        ;;
                    ISPY2|I-SPY2)
                        TARGETS+=("ISPY2")
                        shift
                        ;;
                    *)
                        break
                        ;;
                esac
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

if [[ "${#EXPERIMENTS[@]}" -eq 0 ]]; then
    EXPERIMENTS=(target_only source_only target_full_source random50 random150 random250)
fi

EXPANDED_EXPERIMENTS=()
for exp in "${EXPERIMENTS[@]}"; do
    case "${exp}" in
        all|baselines|basic_methods|methods|k50_methods|k150_methods|k250_methods|tavo|tavo_methods|all_methods)
            read -r -a group_experiments <<< "$(python "${SCRIPT_DIR}/resolve_experiment.py" --list "${exp}")"
            EXPANDED_EXPERIMENTS+=("${group_experiments[@]}")
            ;;
        *)
            EXPANDED_EXPERIMENTS+=("${exp}")
            ;;
    esac
done
EXPERIMENTS=("${EXPANDED_EXPERIMENTS[@]}")

if [[ "${CHAIN}" == "1" ]]; then
    CLEAN_PREPROCESSED_ON_SUCCESS="${CLEAN_PREPROCESSED_ON_SUCCESS:-1}"
    CLEAN_PREPROCESSED_ON_ERROR="${CLEAN_PREPROCESSED_ON_ERROR:-1}"
else
    CLEAN_PREPROCESSED_ON_SUCCESS="${CLEAN_PREPROCESSED_ON_SUCCESS:-0}"
    CLEAN_PREPROCESSED_ON_ERROR="${CLEAN_PREPROCESSED_ON_ERROR:-0}"
fi

echo "SUBMIT=${SUBMIT} FORCE=${FORCE} SKIP_BUILD=${SKIP_BUILD} CHAIN=${CHAIN}"
echo "SBATCH_TIME=${SBATCH_TIME} SBATCH_ACCOUNT=${SBATCH_ACCOUNT} SBATCH_PARTITION=${SBATCH_PARTITION} SBATCH_CONSTRAINT=${SBATCH_CONSTRAINT}"
echo "Targets: ${TARGETS[*]}"
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Repo root: ${REPO_ROOT}"
echo "Project root: ${PROJECT_ROOT:-${REPO_ROOT}}"
echo "CLEAN_PREPROCESSED_ON_SUCCESS=${CLEAN_PREPROCESSED_ON_SUCCESS} CLEAN_PREPROCESSED_ON_ERROR=${CLEAN_PREPROCESSED_ON_ERROR}"

mkdir -p "${REPO_ROOT}/logs/mamamia"

if [[ "${SKIP_BUILD}" != "1" ]]; then
    echo "Generating paper-style LODO splits."
    python "${SCRIPT_DIR}/build_splits.py" --targets "${TARGETS[@]}"
    echo "Preparing nnUNet raw datasets with symlinks."
    python "${SCRIPT_DIR}/build_datasets.py" "${EXPERIMENTS[@]}" --targets "${TARGETS[@]}" --skip-existing
fi

previous_job_id=""
for target in "${TARGETS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        echo
        echo "Validating ${target}/${exp}"
        DRY_RUN=1 FORCE="${FORCE}" \
            CLEAN_PREPROCESSED_ON_SUCCESS="${CLEAN_PREPROCESSED_ON_SUCCESS}" \
            CLEAN_PREPROCESSED_ON_ERROR="${CLEAN_PREPROCESSED_ON_ERROR}" \
            "${SCRIPT_DIR}/run_one.sh" "${target}" "${exp}"

        if [[ "${SUBMIT}" == "1" ]]; then
            echo "Submitting ${target}/${exp}"
            sbatch_args=(
                --account="${SBATCH_ACCOUNT}"
                --partition="${SBATCH_PARTITION}"
                --time="${SBATCH_TIME}"
                --constraint="${SBATCH_CONSTRAINT}"
                --job-name="mamamia_${target}_${exp}"
                --export="ALL,FORCE=${FORCE},CLEAN_PREPROCESSED_ON_SUCCESS=${CLEAN_PREPROCESSED_ON_SUCCESS},CLEAN_PREPROCESSED_ON_ERROR=${CLEAN_PREPROCESSED_ON_ERROR}"
            )
            if [[ "${CHAIN}" == "1" && -n "${previous_job_id}" ]]; then
                sbatch_args+=(--dependency="afterany:${previous_job_id}")
            fi
            sbatch_output="$(cd "${REPO_ROOT}" && sbatch "${sbatch_args[@]}" "${SCRIPT_DIR}/run_one.sh" "${target}" "${exp}")"
            echo "${sbatch_output}"
            previous_job_id="$(awk '{print $4}' <<< "${sbatch_output}")"
        else
            dep_text=""
            if [[ "${CHAIN}" == "1" && -n "${previous_job_id}" ]]; then
                dep_text=" --dependency=afterany:${previous_job_id}"
            fi
            echo "Would submit from ${REPO_ROOT}: sbatch --account=${SBATCH_ACCOUNT} --partition=${SBATCH_PARTITION} --time=${SBATCH_TIME} --constraint=${SBATCH_CONSTRAINT}${dep_text} --job-name=mamamia_${target}_${exp} --export=ALL,FORCE=${FORCE},CLEAN_PREPROCESSED_ON_SUCCESS=${CLEAN_PREPROCESSED_ON_SUCCESS},CLEAN_PREPROCESSED_ON_ERROR=${CLEAN_PREPROCESSED_ON_ERROR} ${SCRIPT_DIR}/run_one.sh ${target} ${exp}"
            previous_job_id="DRYRUN_${target}_${exp}"
        fi
    done
done
