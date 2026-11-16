#!/bin/bash
# One-time setup on a Capella login node. Run from anywhere:
#   bash setup.sh            (after copying it), or, from a checkout: bash cluster/setup.sh
# Creates its own checkout at ~/occlusion-memory (never touches other projects' checkouts) and a virtualenv in it.
# Works from a non-login shell (ssh host 'bash setup.sh'): Lmod is initialised if needed (cluster/modules.sh).
# Override:
#   OM_REPO, OM_BRANCH, OM_VENV  checkout, branch, virtualenv location
#   OM_MODULES   modules that provide Python >= 3.11 (on Capella: "GCC/13.3.0 Python/3.12.3"). Loaded here and
#                recorded in the virtualenv, so every job loads them too (cluster/env.sh). Rerunning setup on an
#                existing virtualenv reuses the recorded modules.
#   PYTHON       interpreter to build the virtualenv from (default: the first of python3.13 .. python3.11, python,
#                python3 that is >= 3.11 once the modules are loaded; on Capella `python3` stays the system 3.9)
#   TORCH_INDEX  a PyTorch wheel index for the CUDA version the GPU driver supports, e.g.
#                https://download.pytorch.org/whl/cu128 (default: PyPI's torch, a CUDA 13 build)
set -euo pipefail
OM_REPO="${OM_REPO:-$HOME/occlusion-memory}"
OM_BRANCH="${OM_BRANCH:-stage1-prep/gpu-smoke}"
OM_VENV="${OM_VENV:-$OM_REPO/.venv}"

if [ ! -d "$OM_REPO/.git" ]; then
  git clone --branch "$OM_BRANCH" https://github.com/asadrizvi64/occlusion-memory.git "$OM_REPO"
fi
cd "$OM_REPO"
echo "checkout: $OM_REPO at $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

source cluster/modules.sh
if [ -z "${OM_MODULES:-}" ] && [ -s "$OM_VENV/om_modules" ]; then OM_MODULES="$(cat "$OM_VENV/om_modules")"; fi
# shellcheck disable=SC2086
om_module_load ${OM_MODULES:-}

ok311() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; }
if [ -z "${PYTHON:-}" ]; then
  for c in python3.13 python3.12 python3.11 python python3; do
    if command -v "$c" >/dev/null 2>&1 && ok311 "$c"; then PYTHON="$c"; break; fi
  done
fi
if [ -z "${PYTHON:-}" ] || ! ok311 "$PYTHON"; then
  echo "need Python >= 3.11: set OM_MODULES to a Python module (or PYTHON to an interpreter)" >&2
  exit 1
fi
echo "python: $(command -v "$PYTHON") ($("$PYTHON" -V 2>&1))"
[ -d "$OM_VENV" ] || "$PYTHON" -m venv "$OM_VENV"
if [ -n "${OM_MODULES:-}" ]; then echo "$OM_MODULES" > "$OM_VENV/om_modules"; fi
# shellcheck disable=SC1091
source "$OM_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt  # the PyPI torch wheel for Linux is a CUDA build
if [ -n "${TORCH_INDEX:-}" ]; then
  pip install -q --force-reinstall --no-deps "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')" --index-url "$TORCH_INDEX"
fi
mkdir -p cluster/logs results
python -c 'import torch, mujoco, numpy; print("python ok: torch", torch.__version__, "cuda build", torch.version.cuda, "| mujoco", mujoco.__version__)'
echo "next: cd $OM_REPO && sbatch cluster/gpu_smoke.sbatch"
