#!/bin/bash
set -euo pipefail

ROOT="external/efficientvit"
ACCOUNT="${ACCOUNT:-YOUR_SLURM_ACCOUNT}"
MANIFEST="${MANIFEST:-${ROOT}/configs_cls_revise/officehome_da/manifest_seed00.csv}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs_cls_revise/officehome_da_seed0_20260615}"
JOB_NAME="${JOB_NAME:-oh_da_s0}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
THROTTLE="${THROTTLE:-12}"
START="${START:-0}"
COUNT="${COUNT:-999999}"
SKIP_DONE="${SKIP_DONE:-1}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

n=$(($(wc -l < "${MANIFEST}") - 1))
if [[ "${n}" -le 0 ]]; then
  echo "No rows in ${MANIFEST}" >&2
  exit 1
fi
if [[ "${START}" -ge "${n}" ]]; then
  echo "Nothing to submit: START=${START} >= rows=${n}"
  exit 0
fi

remaining=$((n - START))
if [[ "${COUNT}" -gt "${remaining}" ]]; then
  COUNT="${remaining}"
fi
array_spec="0-$((COUNT - 1))%${THROTTLE}"

jobid=$(sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --output="${LOG_DIR}/${JOB_NAME}_%A_%a.out" \
  --error="${LOG_DIR}/${JOB_NAME}_%A_%a.err" \
  --partition=gpu \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --constraint="a40|a100|v100" \
  --cpus-per-task=8 \
  --mem=48G \
  --time="${TIME_LIMIT}" \
  --array="${array_spec}" \
  --export=ALL,MANIFEST="${MANIFEST}",OFFSET="${START}",SKIP_DONE="${SKIP_DONE}" \
  --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
eval "$(python - "${MANIFEST}" "$((SLURM_ARRAY_TASK_ID + OFFSET))" <<'"'"'PY'"'"'
import csv, shlex, sys
with open(sys.argv[1], newline="") as f:
    rows = list(csv.DictReader(f))
row = rows[int(sys.argv[2])]
for key in ["target", "method", "da_method", "budget", "train_seed", "config", "output_dir", "source_split", "target_split"]:
    print(f"{key}={shlex.quote(row[key])}")
PY
)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} manifest_row=$((SLURM_ARRAY_TASK_ID + OFFSET))"
echo "target=${target} method=${method} da=${da_method} budget=${budget} seed=${train_seed}"
echo "config=${config}"
echo "output=${output_dir}"
echo "source_split=${source_split}"
echo "target_split=${target_split}"
if [[ "${SKIP_DONE}" == "1" && -f "${output_dir}/final_results.json" ]]; then
  echo "Skipping completed output: ${output_dir}"
  exit 0
fi
python -u scripts_cls/train_cls_da.py --config "${config}"
')

echo "Submitted ${JOB_NAME}: ${jobid} (rows ${START}-$((START + COUNT - 1)) of ${n}, time=${TIME_LIMIT}, throttle=${THROTTLE}, log_dir=${LOG_DIR})"
