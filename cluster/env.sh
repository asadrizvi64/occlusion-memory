# Sourced by the job scripts. Loads the modules recorded at setup (the virtualenv's interpreter may need the
# Python module's libraries), then activates the virtualenv. Override its location with OM_VENV.
OM_VENV="${OM_VENV:-$HOME/occlusion-memory/.venv}"
if [ ! -f "$OM_VENV/bin/activate" ]; then
  echo "no virtualenv at $OM_VENV; run cluster/setup.sh on a login node first" >&2
  exit 1
fi
if [ -s "$OM_VENV/om_modules" ] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2046
  module load $(cat "$OM_VENV/om_modules")
fi
# shellcheck disable=SC1091
source "$OM_VENV/bin/activate"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  # render on the GPU node; gen.py defaults to OSMesa otherwise
