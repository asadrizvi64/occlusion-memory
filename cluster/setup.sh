#!/bin/bash
# One-time setup on a Capella login node. Run from anywhere:
#   bash setup.sh            (after copying it), or, from a checkout: bash cluster/setup.sh
# Creates its own checkout at ~/occlusion-memory (never touches other projects' checkouts) and a virtualenv in it.
# Override: OM_REPO (checkout), OM_BRANCH, OM_VENV, PYTHON (an interpreter >= 3.11).
set -euo pipefail
OM_REPO="${OM_REPO:-$HOME/occlusion-memory}"
OM_BRANCH="${OM_BRANCH:-stage1-prep/gpu-smoke}"
OM_VENV="${OM_VENV:-$OM_REPO/.venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$OM_REPO/.git" ]; then
  git clone --branch "$OM_BRANCH" https://github.com/asadrizvi64/occlusion-memory.git "$OM_REPO"
fi
cd "$OM_REPO"
echo "checkout: $OM_REPO at $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || {
  echo "need Python >= 3.11; set PYTHON=/path/to/python3.11+ (or load a Python module first)" >&2; exit 1; }
[ -d "$OM_VENV" ] || "$PYTHON" -m venv "$OM_VENV"
# shellcheck disable=SC1091
source "$OM_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt  # the PyPI torch wheel for Linux is a CUDA build
mkdir -p cluster/logs results
python -c 'import torch, mujoco, numpy; print("python ok: torch", torch.__version__, "cuda build", torch.version.cuda, "| mujoco", mujoco.__version__)'
echo "next: cd $OM_REPO && sbatch cluster/gpu_smoke.sbatch"
