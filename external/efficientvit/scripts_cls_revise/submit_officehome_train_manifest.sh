#!/bin/bash
#SBATCH --job-name=cls_oh_train
#SBATCH --output=logs_cls_revise/officehome_train_seed0/%A_%a.out
#SBATCH --error=logs_cls_revise/officehome_train_seed0/%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-119%24

set -euo pipefail

cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs_cls_revise/officehome_train_seed0

MANIFEST=${MANIFEST:-configs_cls_revise/officehome_train_seed0_manifest.json}

CONFIG=$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1]))
idx = int(sys.argv[2])
print(rows[idx]["config"])
PY
)

python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1]))
row = rows[int(sys.argv[2])]
print("SLURM_JOB_ID =", __import__("os").environ.get("SLURM_JOB_ID"))
print("SLURM_ARRAY_TASK_ID =", __import__("os").environ.get("SLURM_ARRAY_TASK_ID"))
print("target =", row["target"])
print("method =", row["method"])
print("budget =", row["budget"])
print("train_seed =", row["train_seed"])
print("category =", row["category"])
print("config =", row["config"])
PY

python -u scripts_cls/train_cls.py --config "${CONFIG}"
