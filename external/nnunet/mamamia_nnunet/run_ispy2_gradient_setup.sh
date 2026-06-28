#!/bin/bash
#SBATCH --job-name=mamamia_ISPY2_gradients
#SBATCH --output=logs/mamamia/%x_%j.out
#SBATCH --error=logs/mamamia/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a100|a40|l40s|v100"
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -Eeuo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/scripts/mamamia_nnunet" ]]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/data_selection}"
NNUNET_ROOT="${NNUNET_ROOT:-${PROJECT_ROOT}/externals/MAMA-MIA/nnUNet/nnunetv2}"
export REPO_ROOT PROJECT_ROOT
export nnUNet_raw="${NNUNET_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_ROOT}/nnUNet_results_scratch"
export PYTHONPATH="${PROJECT_ROOT}/externals/MAMA-MIA/nnUNet:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate data_selection_3_10

TARGET_DS="Dataset1101_MAMAMIA_ISPY2_TARGET_ONLY"
EXT_DS="Dataset1111_MAMAMIA_ISPY2_EXT_ONLY"
TARGET_PLAN="${nnUNet_preprocessed}/${TARGET_DS}/nnUNetPlans_2d"
EXT_PLAN="${nnUNet_preprocessed}/${EXT_DS}/nnUNetPlans_2d"
POOL_CASES_FILE="${PROJECT_ROOT}/mamamia_ispy2/data/splits/ISPY2/pool_external.txt"
QUERY_CASES_FILE="${PROJECT_ROOT}/mamamia_ispy2/data/splits/ISPY2/target_query_all.txt"

if ! TARGET_PLAN="${TARGET_PLAN}" EXT_PLAN="${EXT_PLAN}" POOL_CASES_FILE="${POOL_CASES_FILE}" QUERY_CASES_FILE="${QUERY_CASES_FILE}" python - <<'PY'
import os
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


checks = [
    (Path(os.environ["TARGET_PLAN"]), read_ids(Path(os.environ["QUERY_CASES_FILE"]))),
    (Path(os.environ["EXT_PLAN"]), read_ids(Path(os.environ["POOL_CASES_FILE"]))),
]

missing = []
for plan_dir, case_ids in checks:
    if not plan_dir.is_dir():
        missing.append(str(plan_dir))
        continue
    for case_id in case_ids:
        if not (plan_dir / f"{case_id}.npz").exists():
            missing.append(str(plan_dir / f"{case_id}.npz"))
            if len(missing) >= 20:
                break
    if len(missing) >= 20:
        break

if missing:
    print("Preprocessing incomplete; missing examples:")
    for item in missing[:20]:
        print(item)
    raise SystemExit(1)
PY
then
    nnUNetv2_plan_and_preprocess -d 1101 1111 -c 2d
fi

CHECKPOINT="${CHECKPOINT:-${nnUNet_results}/Dataset1303_MAMAMIA_NACT_LODO_SEED42_TAVO_TARGET_FULL_SOURCE_2d_3ch/nnUNetTrainerTAVOSaveEveryEpoch__nnUNetPlans__2d/fold_0/checkpoint_final.pth}"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi
GRAD_DIR="${PROJECT_ROOT}/mamamia_ispy2/data/gradients"

python "${REPO_ROOT}/scripts/mamamia_nnunet/extract_case_gradients.py" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-dirs "${TARGET_PLAN}" "${EXT_PLAN}" \
    --pool-cases-file "${POOL_CASES_FILE}" \
    --query-cases-file "${QUERY_CASES_FILE}" \
    --output-dir "${GRAD_DIR}" \
    --proj-dim 4096 \
    --seed 42

python "${REPO_ROOT}/scripts/mamamia_nnunet/derive_tumorseg2025_selections.py" \
    --targets ISPY2 \
    --methods orient gradmatch craig less \
    --max-rank 250

python "${REPO_ROOT}/scripts/mamamia_nnunet/materialize_method_selections.py" \
    --targets ISPY2 \
    --methods orient gradmatch craig less \
    --budgets 50 150 \
    --strict

python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
sys.path.insert(0, str(repo / "scripts" / "mamamia_nnunet"))
from core import EXPERIMENTS, dataset_basename, nnunet_preprocessed_root, nnunet_raw_root, nnunet_results_root

for exp_key in [
    "gradmatch50",
    "less50",
    "orient50",
    "craig50",
    "gradmatch150",
    "less150",
    "orient150",
    "craig150",
]:
    name = dataset_basename("ISPY2", EXPERIMENTS[exp_key])
    for root in [nnunet_raw_root(), nnunet_preprocessed_root(), nnunet_results_root()]:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
PY

python "${REPO_ROOT}/scripts/mamamia_nnunet/build_datasets.py" \
    gradmatch50 less50 orient50 craig50 gradmatch150 less150 orient150 craig150 \
    --targets ISPY2 \
    --overwrite
