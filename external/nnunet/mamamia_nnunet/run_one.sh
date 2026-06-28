#!/bin/bash
#SBATCH --job-name=mamamia_nnunet
#SBATCH --output=logs/mamamia/%x_%j.out
#SBATCH --error=logs/mamamia/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a100|a40|l40s|v100"
#SBATCH --mem=64G
#SBATCH --time=12:00:00

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
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
NNUNET_STORAGE_ROOT="${NNUNET_STORAGE_ROOT:-${PROJECT_ROOT}/outputs/nnunet}"
RAW_DIR="${NNUNET_RAW:-${NNUNET_STORAGE_ROOT}/nnUNet_raw}"
PREPROCESSED_DIR="${NNUNET_PREPROCESSED:-${NNUNET_STORAGE_ROOT}/nnUNet_preprocessed}"
RESULTS_DIR="${NNUNET_RESULTS:-${NNUNET_STORAGE_ROOT}/nnUNet_results_scratch}"
GT_SOURCE_DIR="${GT_SOURCE_DIR:-${MAMAMIA_DATASET_ROOT:-${PROJECT_ROOT}/data/mamamia}/segmentations/expert}"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/splits/mamamia_lodo_seed42}"

TARGET="${TARGET:-NACT}"
EXPERIMENT="${EXPERIMENT:-}"
if [[ $# -eq 1 ]]; then
    EXPERIMENT="$1"
elif [[ $# -ge 2 ]]; then
    TARGET="$1"
    EXPERIMENT="$2"
fi

CONFIG="${CONFIG:-2d}"
FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainerTAVOSaveEveryEpoch}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_BUILD_DATASET="${AUTO_BUILD_DATASET:-0}"
SKIP_PLAN_PREPROCESS="${SKIP_PLAN_PREPROCESS:-0}"
CLEAN_PREPROCESSED_ON_SUCCESS="${CLEAN_PREPROCESSED_ON_SUCCESS:-0}"
CLEAN_PREPROCESSED_ON_ERROR="${CLEAN_PREPROCESSED_ON_ERROR:-0}"
CLEAN_UNPACKED_PREPROCESSED_BEFORE_TRAIN="${CLEAN_UNPACKED_PREPROCESSED_BEFORE_TRAIN:-0}"
BEST_LAST_WINDOW="${BEST_LAST_WINDOW:-10}"
PREDICT_CHECKPOINT="${PREDICT_CHECKPOINT:-checkpoint_best_last.pth}"
CONDA_ENV="${CONDA_ENV:-mamamia_nnunet}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -x "${CONDA_ENV}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_ENV}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
STOP_AFTER_TRAIN="${STOP_AFTER_TRAIN:-0}"
USE_COMPRESSED="${USE_COMPRESSED:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
OUTPUT_EXPERIMENT="${OUTPUT_EXPERIMENT:-${EXPERIMENT}${OUTPUT_SUFFIX}}"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %Z'
}

log() {
    echo "[$(timestamp)] $*"
}

section() {
    echo
    log "===== $* ====="
}

snapshot_storage() {
    log "Storage snapshot"
    df -h "${PROJECT_ROOT}" "${REPO_ROOT}" 2>/dev/null || true
    du -sh "${OUT_DIR:-${PROJECT_ROOT}/outputs}" "${TRAIN_OUT_DIR:-${RESULTS_DIR}}" 2>/dev/null || true
}

snapshot_gpu() {
    log "GPU snapshot"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
}

checkpoint_inventory() {
    local dir="${TRAIN_OUT_DIR:-}"
    if [[ -z "${dir}" || ! -d "${dir}" ]]; then
        log "Checkpoint inventory: no training directory yet (${dir:-unset})"
        return 0
    fi
    log "Checkpoint inventory for ${dir}"
    find "${dir}" -maxdepth 1 -type f \( -name 'checkpoint_latest.pth' -o -name 'checkpoint_final.pth' -o -name 'checkpoint_best.pth' -o -name 'checkpoint_best_last.pth' -o -name 'checkpoint_epoch_*.pth' \) -printf '%TY-%Tm-%Td %TH:%TM %9s %f\n' 2>/dev/null | sort | tail -20 || true
}

cleanup_preprocessed_dataset() {
    local reason="${1:-manual}"
    local dir="${PREPROCESSED_DATASET_DIR:-}"
    if [[ -z "${dir}" || "${dir}" == "/" || ! -d "${dir}" ]]; then
        log "Preprocessed cleanup skipped (${reason}); directory not present: ${dir:-unset}"
        return 0
    fi
    log "Removing preprocessed dataset after ${reason}: ${dir}"
    rm -rf "${dir}"
    snapshot_storage
}

on_error() {
    local exit_code=$?
    local line_no=${1:-unknown}
    log "ERROR: command failed at line ${line_no} with exit code ${exit_code}: ${BASH_COMMAND}"
    snapshot_storage
    snapshot_gpu
    checkpoint_inventory
    if [[ "${CLEAN_PREPROCESSED_ON_ERROR}" == "1" ]]; then
        cleanup_preprocessed_dataset "error"
    fi
    exit "${exit_code}"
}

on_signal() {
    local signal_name="${1:-TERM}"
    log "Received ${signal_name}; collecting diagnostics before exit."
    snapshot_storage
    snapshot_gpu
    checkpoint_inventory
    if [[ "${CLEAN_PREPROCESSED_ON_ERROR}" == "1" ]]; then
        cleanup_preprocessed_dataset "signal_${signal_name}"
    fi
    exit 143
}

trap 'on_error ${LINENO}' ERR
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

usage() {
    cat <<'EOF'
Usage:
  run_one.sh <experiment>
  run_one.sh <target> <experiment>

Targets:
  NACT ISPY1 DUKE ISPY2

Experiments:
  target_only source_only target_full_source random50 random150 random250
  rds50 gradmatch50 less50 orient50 diversity50 kmeans50 craig50 kcenter50
  rds150 gradmatch150 less150 orient150 diversity150 kmeans150 craig150 kcenter150
  tavo50 tavo150

Environment:
  DRY_RUN=1             validate paths and print the resolved command plan
  FORCE=1               rerun even if the test summary exists
  AUTO_BUILD_DATASET=1  create the nnUNet raw dataset first if it is missing
  SKIP_PLAN_PREPROCESS=1 skip nnUNet planning/preprocessing when preprocessed files already exist
  CONDA_ENV=name        conda environment to activate for nnUNet; default mamamia_nnunet
  TRAIN_DEVICE=cuda|cpu device passed to nnUNetv2_train; default cuda
  STOP_AFTER_TRAIN=1    stop after best-last checkpoint selection; useful for smoke tests
  CLEAN_PREPROCESSED_ON_SUCCESS=1  remove this dataset's preprocessed dir after successful evaluation
  CLEAN_PREPROCESSED_ON_ERROR=1    remove this dataset's preprocessed dir after a runner error

Path defaults are repo-relative:
  PROJECT_ROOT defaults to the release repository root
EOF
}

if [[ -z "${EXPERIMENT}" ]]; then
    usage
    exit 2
fi

case "${TARGET}" in
    -h|--help|help)
        usage
        exit 0
        ;;
esac
case "${EXPERIMENT}" in
    -h|--help|help)
        usage
        exit 0
        ;;
esac

if ! RESOLVED_EXPERIMENT="$(${PYTHON_BIN} "${SCRIPT_DIR}/resolve_experiment.py" --shell "${TARGET}" "${EXPERIMENT}")"; then
    echo "Unknown target or experiment: ${TARGET}/${EXPERIMENT}" >&2
    usage >&2
    exit 2
fi
eval "${RESOLVED_EXPERIMENT}"

DATASET_DIR="${RAW_DIR}/${DATASET_BASENAME}"
PREPROCESSED_DATASET_DIR="${PREPROCESSED_DIR}/${DATASET_BASENAME}"
TARGET_TEST_SPLIT="${SPLIT_ROOT}/${TARGET}/target_test.txt"
TARGET_LC="$(echo "${TARGET}" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="${PROJECT_ROOT}/outputs/tavo_mamamia_${TARGET_LC}_nnunet_${OUTPUT_EXPERIMENT}/repeat_01"
PRED_DIR="${OUT_DIR}/test_preds"
GT_DIR="${OUT_DIR}/test_gt"
SUMMARY_JSON="${PRED_DIR}/summary.json"

if [[ ! -d "${DATASET_DIR}" ]]; then
    if [[ "${AUTO_BUILD_DATASET}" == "1" && "${DRY_RUN}" != "1" ]]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/build_splits.py" --targets "${TARGET}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/build_datasets.py" "${EXPERIMENT}" --targets "${TARGET}" --skip-existing
    else
        echo "Missing nnUNet raw dataset: ${DATASET_DIR}" >&2
        echo "Build datasets first: python ${SCRIPT_DIR}/build_datasets.py ${EXPERIMENT} --targets ${TARGET} --skip-existing" >&2
        if [[ "${DRY_RUN}" == "1" ]]; then
            exit 0
        fi
        exit 1
    fi
fi

for required in "${DATASET_DIR}/dataset.json" "${DATASET_DIR}/splits_final.json" "${DATASET_DIR}/imagesTr" "${DATASET_DIR}/imagesTs" "${TARGET_TEST_SPLIT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required path: ${required}" >&2
        exit 1
    fi
done

print_summary() {
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

summary = Path(os.environ["SUMMARY_JSON"])
data = json.loads(summary.read_text())
fg_dice = data.get("foreground_mean", {}).get("Dice")
print(f"Test Dice (foreground_mean): {fg_dice}")
PY
}

echo "Target: ${TARGET}"
echo "Experiment: ${EXPERIMENT}"
echo "Label: ${LABEL}"
echo "Dataset: ${DATASET_BASENAME}"
echo "Repo root: ${REPO_ROOT}"
echo "Project root: ${PROJECT_ROOT}"
echo "nnUNet raw: ${RAW_DIR}"
echo "nnUNet preprocessed: ${PREPROCESSED_DIR}"
echo "nnUNet results: ${RESULTS_DIR}"
echo "Target test split: ${TARGET_TEST_SPLIT}"
echo "Output: ${OUT_DIR}"
echo "DRY_RUN=${DRY_RUN} FORCE=${FORCE} SKIP_PLAN_PREPROCESS=${SKIP_PLAN_PREPROCESS}"
echo "CLEAN_PREPROCESSED_ON_SUCCESS=${CLEAN_PREPROCESSED_ON_SUCCESS} CLEAN_PREPROCESSED_ON_ERROR=${CLEAN_PREPROCESSED_ON_ERROR}"
echo "CLEAN_UNPACKED_PREPROCESSED_BEFORE_TRAIN=${CLEAN_UNPACKED_PREPROCESSED_BEFORE_TRAIN}"
echo "BEST_LAST_WINDOW=${BEST_LAST_WINDOW} PREDICT_CHECKPOINT=${PREDICT_CHECKPOINT}"
echo "TRAINER=${TRAINER} OUTPUT_EXPERIMENT=${OUTPUT_EXPERIMENT} CONDA_ENV=${CONDA_ENV} TRAIN_DEVICE=${TRAIN_DEVICE} STOP_AFTER_TRAIN=${STOP_AFTER_TRAIN}"
TRAIN_OUT_DIR="${RESULTS_DIR}/${DATASET_BASENAME}/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}"
LATEST_CKPT="${TRAIN_OUT_DIR}/checkpoint_latest.pth"
FINAL_CKPT="${TRAIN_OUT_DIR}/checkpoint_final.pth"
PREDICT_CKPT="${TRAIN_OUT_DIR}/${PREDICT_CHECKPOINT}"
echo "Training output: ${TRAIN_OUT_DIR}"

if [[ -f "${SUMMARY_JSON}" && "${FORCE}" != "1" ]]; then
    echo "summary.json already exists; skipping because FORCE!=1."
    export SUMMARY_JSON
    print_summary
    exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run only. Would plan/preprocess, train, predict, and evaluate ${TARGET}/${LABEL}."
    exit 0
fi

mkdir -p "${PROJECT_ROOT}/logs/mamamia" "${OUT_DIR}"

section "Initial environment"
log "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
log "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-none}"
log "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-none}"
log "TMPDIR=${TMPDIR:-none}"
snapshot_storage

module purge
module load gcc/13.3.0
module load cuda/12.6.3
module load cudnn/8.9.7.29-12-cuda
module load conda

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
log "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

if [[ -n "${EXTRA_PATH:-}" ]]; then
    export PATH="${EXTRA_PATH}:${PATH}"
fi

export nnUNet_raw="${RAW_DIR}"
export nnUNet_preprocessed="${PREPROCESSED_DIR}"
export nnUNet_results="${RESULTS_DIR}"
export nnUNet_n_proc_DA="1"
export nnUNet_compile="false"
export PYTHONUNBUFFERED=1
export MAMAMIA_TARGET="${TARGET}"
export MAMAMIA_EXPERIMENT="${EXPERIMENT}"
export MAMAMIA_OUTPUT_EXPERIMENT="${OUTPUT_EXPERIMENT}"
export SPLIT_ROOT="${SPLIT_ROOT}"
if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

NNUNET_TRAINER_DIR="${NNUNET_TRAINER_DIR:-$(python - <<'PY'
from pathlib import Path
import nnunetv2.training.nnUNetTrainer as trainer_pkg
print(Path(trainer_pkg.__file__).resolve().parent)
PY
)}"

for TRAINER_SRC in "${SCRIPT_DIR}"/nnUNetTrainerTAVO*.py; do
    TRAINER_DST="${NNUNET_TRAINER_DIR}/$(basename "${TRAINER_SRC}")"
    mkdir -p "$(dirname "${TRAINER_DST}")"
    if ! cmp -s "${TRAINER_SRC}" "${TRAINER_DST}"; then
        log "Installing trainer into nnU-Net package: ${TRAINER_DST}"
        TRAINER_TMP="${TRAINER_DST}.${SLURM_JOB_ID:-$$}.tmp"
        cp "${TRAINER_SRC}" "${TRAINER_TMP}"
        chmod 0644 "${TRAINER_TMP}"
        mv -f "${TRAINER_TMP}" "${TRAINER_DST}"
    else
        log "Trainer already up to date: ${TRAINER_DST}"
    fi
done

if [[ "${SKIP_PLAN_PREPROCESS}" == "1" ]]; then
    section "Plan and preprocess"
    log "Skipping plan/preprocess because SKIP_PLAN_PREPROCESS=1."
    for required_preprocessed in "${PREPROCESSED_DATASET_DIR}/nnUNetPlans.json" "${PREPROCESSED_DATASET_DIR}/dataset.json" "${PREPROCESSED_DATASET_DIR}/nnUNetPlans_${CONFIG}"; do
        if [[ ! -e "${required_preprocessed}" ]]; then
            echo "Missing preprocessed path while SKIP_PLAN_PREPROCESS=1: ${required_preprocessed}" >&2
            exit 1
        fi
    done
    export PREPROCESSED_DATASET_DIR CONFIG FOLD
    python - <<'PY'
import json
import os
from pathlib import Path

preprocessed_dir = Path(os.environ["PREPROCESSED_DATASET_DIR"])
config = os.environ["CONFIG"]
fold = int(os.environ["FOLD"])
split_file = preprocessed_dir / "splits_final.json"
case_dir = preprocessed_dir / f"nnUNetPlans_{config}"
splits = json.loads(split_file.read_text())
if fold >= len(splits):
    raise SystemExit(f"Fold {fold} not found in {split_file}")
cases = list(splits[fold]["train"]) + list(splits[fold]["val"])
missing = [case for case in cases if not (case_dir / f"{case}.pkl").exists()]
if missing:
    preview = ", ".join(missing[:10])
    raise SystemExit(
        f"SKIP_PLAN_PREPROCESS=1 but {len(missing)} split cases are not preprocessed in {case_dir}: {preview}"
    )
print(f"Preprocessed split check passed for {len(cases)} train/val cases.")
PY
else
    section "Plan and preprocess"
    nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -c "${CONFIG}"
    snapshot_storage
fi

if [[ "${CLEAN_UNPACKED_PREPROCESSED_BEFORE_TRAIN}" == "1" ]]; then
    section "Clean unpacked preprocessed cache"
    if [[ -z "${PREPROCESSED_DATASET_DIR:-}" || "${PREPROCESSED_DATASET_DIR}" == "/" || ! -d "${PREPROCESSED_DATASET_DIR}/nnUNetPlans_${CONFIG}" ]]; then
        echo "Cannot clean unpacked cache; invalid preprocessed directory: ${PREPROCESSED_DATASET_DIR:-unset}" >&2
        exit 1
    fi
    find "${PREPROCESSED_DATASET_DIR}/nnUNetPlans_${CONFIG}" -maxdepth 1 -type f \( -name '*.npy' -o -name '*_seg.npy' \) -delete
    log "Removed unpacked .npy cache files from ${PREPROCESSED_DATASET_DIR}/nnUNetPlans_${CONFIG}."
fi

export nnUNet_n_proc_DA="0"
section "Train ${TARGET}/${LABEL}"
python - <<'PY'
import os
print(f"Python sees nnUNet_n_proc_DA={os.environ.get('nnUNet_n_proc_DA', 'NOT SET')}")
PY
snapshot_gpu
checkpoint_inventory
TRAIN_ARGS=("${DATASET_ID}" "${CONFIG}" "${FOLD}" -tr "${TRAINER}" -device "${TRAIN_DEVICE}")
if [[ "${USE_COMPRESSED}" == "1" ]]; then
    TRAIN_ARGS+=(--use_compressed)
fi
if [[ -f "${FINAL_CKPT}" ]]; then
    log "Final checkpoint exists; skipping training: ${FINAL_CKPT}"
elif [[ -f "${LATEST_CKPT}" ]]; then
    log "Resuming from latest checkpoint: ${LATEST_CKPT}"
    nnUNet_n_proc_DA=0 nnUNetv2_train "${TRAIN_ARGS[@]}" --c
else
    log "Starting fresh training. ${TRAINER} will save checkpoint_latest/checkpoint_final, per-epoch checkpoints, and lr_scheduler_state."
    nnUNet_n_proc_DA=0 nnUNetv2_train "${TRAIN_ARGS[@]}"
fi
checkpoint_inventory
snapshot_storage

section "Select best-last checkpoint and report train/val metrics"
export DATASET_ID DATASET_NAME RESULTS_DIR CONFIG FOLD TRAINER BEST_LAST_WINDOW PREDICT_CHECKPOINT
python - <<'PY'
import json
import os
import shutil
from pathlib import Path
import torch

dataset_id = os.environ["DATASET_ID"]
dataset_name = os.environ["DATASET_NAME"]
results_dir = Path(os.environ["RESULTS_DIR"])
config = os.environ["CONFIG"]
fold = os.environ["FOLD"]
trainer = os.environ["TRAINER"]
best_last_window = int(os.environ["BEST_LAST_WINDOW"])
predict_checkpoint = os.environ["PREDICT_CHECKPOINT"]
output_folder = results_dir / f"Dataset{dataset_id}_{dataset_name}" / f"{trainer}__nnUNetPlans__{config}" / f"fold_{fold}"
ckpt = output_folder / "checkpoint_final.pth"
if not ckpt.exists():
    ckpt = output_folder / "checkpoint_latest.pth"
if not ckpt.exists():
    raise SystemExit(f"No checkpoint found in {output_folder}")

checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
log = checkpoint["logging"]
print(f"Last epoch train loss: {log['train_losses'][-1]}")
print(f"Last epoch val loss: {log['val_losses'][-1]}")
print(f"Last epoch val dice (mean_fg_dice): {log['mean_fg_dice'][-1]}")
print(f"Last epoch val dice (ema_fg_dice): {log['ema_fg_dice'][-1]}")

val_dice = list(log["mean_fg_dice"])
if not val_dice:
    raise SystemExit(f"No mean_fg_dice entries in {ckpt}")

start = max(0, len(val_dice) - best_last_window)
candidates = []
for epoch_idx in range(start, len(val_dice)):
    epoch_ckpt = output_folder / f"checkpoint_epoch_{epoch_idx:03d}.pth"
    if not epoch_ckpt.exists():
        continue
    candidates.append((float(val_dice[epoch_idx]), epoch_idx, epoch_ckpt))

if not candidates:
    raise SystemExit(
        f"No checkpoint_epoch_*.pth files available in the last {best_last_window} epochs under {output_folder}"
    )

best_val, best_epoch, best_ckpt = max(candidates, key=lambda item: (item[0], item[1]))
target = output_folder / predict_checkpoint
shutil.copy2(best_ckpt, target)
metadata = {
    "policy": "best_mean_fg_dice_in_last_n_epochs",
    "window": best_last_window,
    "source_checkpoint": best_ckpt.name,
    "predict_checkpoint": target.name,
    "epoch_index_zero_based": best_epoch,
    "epoch_number_one_based": best_epoch + 1,
    "mean_fg_dice": best_val,
    "reference_checkpoint": ckpt.name,
}
(output_folder / "checkpoint_best_last.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(
    f"Selected {target.name} from {best_ckpt.name}: "
    f"mean_fg_dice={best_val:.6f}, best of last {len(candidates)} epoch checkpoints"
)
PY

if [[ ! -f "${PREDICT_CKPT}" ]]; then
    echo "Missing prediction checkpoint after best-last selection: ${PREDICT_CKPT}" >&2
    exit 1
fi
checkpoint_inventory

if [[ "${STOP_AFTER_TRAIN}" == "1" ]]; then
    log "STOP_AFTER_TRAIN=1; stopping after training and best-last checkpoint selection."
    exit 0
fi

section "Predict on target test set"
snapshot_gpu
log "Predicting with checkpoint: ${PREDICT_CHECKPOINT}"
if [[ "${FORCE}" == "1" && -d "${PRED_DIR}" ]]; then
    log "FORCE=1; clearing previous prediction directory: ${PRED_DIR}"
    rm -rf "${PRED_DIR}"
fi
mkdir -p "${PRED_DIR}"
nnUNetv2_predict -i "${DATASET_DIR}/imagesTs" -o "${PRED_DIR}" -d "${DATASET_ID}" -c "${CONFIG}" -f "${FOLD}" -tr "${TRAINER}" -chk "${PREDICT_CHECKPOINT}"
snapshot_storage

section "Prepare target test GT folder"
export TARGET_TEST_SPLIT GT_SOURCE_DIR GT_DIR
python - <<'PY'
import os
from pathlib import Path
import shutil

target_test_split = Path(os.environ["TARGET_TEST_SPLIT"])
gt_source_dir = Path(os.environ["GT_SOURCE_DIR"])
gt_dir = Path(os.environ["GT_DIR"])
gt_dir.mkdir(parents=True, exist_ok=True)

for line in target_test_split.read_text().splitlines():
    case_id = line.strip()
    if not case_id:
        continue
    src = gt_source_dir / f"{case_id}.nii.gz"
    dst = gt_dir / f"{case_id}.nii.gz"
    if dst.exists() or dst.is_symlink():
        continue
    try:
        dst.symlink_to(src)
    except OSError:
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
PY

section "Evaluate target test predictions"
PLANS_FILE="${PREPROCESSED_DIR}/${DATASET_BASENAME}/nnUNetPlans.json"
DATASET_JSON="${PREPROCESSED_DIR}/${DATASET_BASENAME}/dataset.json"
nnUNetv2_evaluate_folder "${GT_DIR}" "${PRED_DIR}" -djfile "${DATASET_JSON}" -pfile "${PLANS_FILE}" -o "${SUMMARY_JSON}"

section "Test dice summary"
export SUMMARY_JSON
print_summary

if [[ "${CLEAN_PREPROCESSED_ON_SUCCESS}" == "1" ]]; then
    cleanup_preprocessed_dataset "success"
fi

log "Done"
