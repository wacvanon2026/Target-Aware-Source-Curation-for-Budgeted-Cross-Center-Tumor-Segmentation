#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/data_selection}"
CONDA_ENV="${CONDA_ENV:-mamamia_nnunet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
TORCH_INSTALL_CMD="${TORCH_INSTALL_CMD:-python -m pip install torch}"

usage() {
    cat <<'EOF'
Usage:
  scripts/mamamia_nnunet/setup_env.sh

Environment:
  CONDA_ENV=name        environment to create/update; default mamamia_nnunet
  PROJECT_ROOT=path    data_selection root; default <repo>/../data_selection
  INSTALL_TORCH=1      run TORCH_INSTALL_CMD before installing nnU-Net
  TORCH_INSTALL_CMD=... site-specific PyTorch install command

This helper installs the Python side only. It does not copy raw MAMAMIA data.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

NNUNET_REPO="${PROJECT_ROOT}/externals/MAMA-MIA/nnUNet"
if [[ ! -d "${NNUNET_REPO}" ]]; then
    echo "Missing nnU-Net checkout: ${NNUNET_REPO}" >&2
    echo "Set PROJECT_ROOT to the data_selection directory that contains externals/MAMA-MIA/nnUNet." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "Using existing conda env: ${CONDA_ENV}"
else
    echo "Creating conda env: ${CONDA_ENV}"
    conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV}"
python -m pip install --upgrade pip

if [[ "${INSTALL_TORCH}" == "1" ]]; then
    echo "Installing PyTorch with: ${TORCH_INSTALL_CMD}"
    eval "${TORCH_INSTALL_CMD}"
else
    echo "Skipping PyTorch install. Set INSTALL_TORCH=1 if this env does not already have GPU PyTorch."
fi

python -m pip install -e "${NNUNET_REPO}"
python -m pip install cma submodlib

python - <<'PY'
import importlib

required = [
    "torch",
    "nnunetv2",
    "batchgenerators",
    "SimpleITK",
    "nibabel",
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "tqdm",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing required modules:\n" + "\n".join(missing))
print("Environment import check passed.")
PY

echo
echo "Use this environment for jobs with:"
echo "  CONDA_ENV=${CONDA_ENV} python scripts/mamamia_nnunet/submit_domain_alignment.py --targets all --experiments target_full_source --submit"
