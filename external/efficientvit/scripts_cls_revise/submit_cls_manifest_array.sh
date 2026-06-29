#!/bin/bash
set -euo pipefail

ROOT="external/efficientvit"
cd "${ROOT}"
source ~/envs/brain310/bin/activate

MANIFEST="${MANIFEST:...MANIFEST is required}"
JOB_NAME="${JOB_NAME:...JOB_NAME is required}"
LOG_DIR="${LOG_DIR:...LOG_DIR is required}"
TIME_LIMIT="${TIME_LIMIT:...TIME_LIMIT is required}"
ACCOUNT="${ACCOUNT:-YOUR_SLURM_ACCOUNT}"
START="${START:-0}"
COUNT="${COUNT:-}"

mkdir -p "${LOG_DIR}"

ROWS=$(python - "${MANIFEST}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
if [[ -z "${COUNT}" ]]; then
  COUNT=$((ROWS - START))
fi
if [[ "${COUNT}" -le 0 ]]; then
  echo "Nothing to submit: START=${START}, COUNT=${COUNT}, ROWS=${ROWS}"
  exit 0
fi

sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --output="${LOG_DIR}/%A_%a.out" \
  --error="${LOG_DIR}/%A_%a.err" \
  --partition=gpu \
  --account="${ACCOUNT}" \
  --gres=gpu:1 \
  --constraint="a40|a100|v100" \
  --cpus-per-task=8 \
  --mem=48G \
  --time="${TIME_LIMIT}" \
  --array="0-$((COUNT - 1))" \
  --export=ALL,MANIFEST="${MANIFEST}",OFFSET="${START}" \
  --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
IDX=$((SLURM_ARRAY_TASK_ID + OFFSET))
CONFIG=$(python - "${MANIFEST}" "${IDX}" <<'"'"'PY'"'"'
import json, sys
rows=json.load(open(sys.argv[1]))
print(rows[int(sys.argv[2])]["config"])
PY
)
python - "${MANIFEST}" "${IDX}" <<'"'"'PY'"'"'
import json, os, sys
row=json.load(open(sys.argv[1]))[int(sys.argv[2])]
print("SLURM_JOB_ID =", os.environ.get("SLURM_JOB_ID"))
print("SLURM_ARRAY_TASK_ID =", os.environ.get("SLURM_ARRAY_TASK_ID"))
print("manifest_row =", sys.argv[2])
for key in ["target", "method", "budget", "train_seed", "category", "config"]:
    print(f"{key} =", row.get(key))
PY
python -u scripts_cls/train_cls.py --config "${CONFIG}"
'
