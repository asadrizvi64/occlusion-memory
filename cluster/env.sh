# Sourced by the job scripts. Loads the modules recorded at setup (the virtualenv's interpreter belongs to the
# cluster's Python module and needs its libraries), then activates the virtualenv. Override its location with OM_VENV.
OM_VENV="${OM_VENV:-$HOME/occlusion-memory/.venv}"
if [ ! -f "$OM_VENV/bin/activate" ]; then
  echo "no virtualenv at $OM_VENV; run cluster/setup.sh on a login node first" >&2
  exit 1
fi
# shellcheck source=cluster/modules.sh
source "$(dirname "${BASH_SOURCE[0]}")/modules.sh"
if [ -s "$OM_VENV/om_modules" ]; then
  read -r -a _om_mods < "$OM_VENV/om_modules" || true
  om_module_load "${_om_mods[@]}" || exit 1
fi
# shellcheck disable=SC1091
source "$OM_VENV/bin/activate"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  # render on the GPU node; gen.py defaults to OSMesa otherwise
