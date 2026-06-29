#!/bin/bash
#SBATCH --job-name=cls_oh_warmup
#SBATCH --output=logs_cls_revise/officehome_warmup/%A_%a.out
#SBATCH --error=logs_cls_revise/officehome_warmup/%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4

set -euo pipefail

cd external/efficientvit
source ~/envs/brain310/bin/activate
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs_cls_revise/officehome_warmup

MANIFEST=${MANIFEST:-configs_cls_revise/officehome_warmup/manifest_split00.json}
CONFIG=$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
idx = int(sys.argv[2])
print(manifest[idx]["config"])
PY
)

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "config=${CONFIG}"

python -u scripts_cls/train_cls.py --config "${CONFIG}"
