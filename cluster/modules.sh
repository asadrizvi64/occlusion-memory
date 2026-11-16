# Sourced by env.sh and setup.sh. Defines om_module_load, which loads environment modules from any shell.
# A non-login shell (`ssh host 'sbatch ...'`, or a Slurm job submitted from one) has no `module` command on
# Capella, so Lmod is initialised first. Fails loudly if modules were asked for and cannot be loaded.
om_module_load() {
  [ "$#" -gt 0 ] || return 0
  local had_u=0
  case $- in *u*) had_u=1; set +u ;; esac  # Lmod's scripts are not written for `set -u`
  if ! type module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/00-modulepath.sh ]; then . /etc/profile.d/00-modulepath.sh; fi
    if [ -n "${MODULESHOME:-}" ] && [ -f "$MODULESHOME/init/bash" ]; then
      . "$MODULESHOME/init/bash"
    elif [ -f /software/foundation/x86_64/lmod/lmod/init/bash ]; then
      . /software/foundation/x86_64/lmod/lmod/init/bash
    fi
  fi
  if ! type module >/dev/null 2>&1; then
    echo "cannot load modules ($*): no 'module' command and no Lmod init script found" >&2
    return 1
  fi
  module load "$@" || { echo "module load $* failed" >&2; return 1; }
  if [ "$had_u" = 1 ]; then set -u; fi
}
