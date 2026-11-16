# Sourced by the job scripts. Activates the occlusion-memory virtualenv; override the location with OM_VENV.
OM_VENV="${OM_VENV:-$HOME/occlusion-memory/.venv}"
if [ ! -f "$OM_VENV/bin/activate" ]; then
  echo "no virtualenv at $OM_VENV; run cluster/setup.sh on a login node first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$OM_VENV/bin/activate"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  # render on the GPU node; gen.py defaults to OSMesa otherwise
