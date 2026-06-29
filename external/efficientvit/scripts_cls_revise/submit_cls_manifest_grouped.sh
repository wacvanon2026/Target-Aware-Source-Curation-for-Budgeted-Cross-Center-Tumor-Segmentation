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
GROUP_SIZE="${GROUP_SIZE:-36}"
SKIP_DONE="${SKIP_DONE:-1}"
DEPENDENCY="${DEPENDENCY:-}"

mkdir -p "${LOG_DIR}"

ROWS=$(python - "${MANIFEST}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
NUM_GROUPS=$(( (ROWS + GROUP_SIZE - 1) / GROUP_SIZE ))
SBATCH_DEP_ARGS=()
if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_DEP_ARGS=(--dependency="${DEPENDENCY}")
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
  "${SBATCH_DEP_ARGS[@]}" \
  --array="0-$((NUM_GROUPS - 1))" \
  --export=ALL,MANIFEST="${MANIFEST}",GROUP_SIZE="${GROUP_SIZE}",SKIP_DONE="${SKIP_DONE}" \
  --wrap='
set -euo pipefail
cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
START=$((SLURM_ARRAY_TASK_ID * GROUP_SIZE))
END=$((START + GROUP_SIZE))
ROWS=$(python - "${MANIFEST}" <<'"'"'PY'"'"'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
if [[ "${END}" -gt "${ROWS}" ]]; then
  END="${ROWS}"
fi
echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Running manifest rows ${START}-$((END - 1)) of ${ROWS}"
for IDX in $(seq "${START}" "$((END - 1))"); do
  eval "$(python - "${MANIFEST}" "${IDX}" <<'"'"'PY'"'"'
import json, shlex, sys, yaml
row=json.load(open(sys.argv[1]))[int(sys.argv[2])]
cfg=yaml.safe_load(open(row["config"]))
print("CONFIG=" + shlex.quote(row["config"]))
print("SAVE_DIR=" + shlex.quote(cfg["experiment"]["save_dir"]))
desc = "{} {} {} seed{}".format(row.get("target"), row.get("method"), row.get("budget"), row.get("train_seed"))
print("DESC=" + shlex.quote(desc))
PY
)"
  echo "===== ROW ${IDX}: ${DESC} ====="
  echo "config=${CONFIG}"
  echo "save_dir=${SAVE_DIR}"
  if [[ "${SKIP_DONE}" == "1" && -f "${SAVE_DIR}/last_best.pt" ]]; then
    echo "Skipping completed checkpoint: ${SAVE_DIR}/last_best.pt"
    continue
  fi
  python -u scripts_cls/train_cls.py --config "${CONFIG}"
done
'
