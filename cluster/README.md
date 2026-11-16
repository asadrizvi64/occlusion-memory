# Running on Capella (TU Dresden)

Jobs for this project run from their own checkout, `~/occlusion-memory`. They never touch another project's checkout or its jobs.

## Before connecting

- **VPN.** Capella is reachable only through TU Dresden's VPN.
- **Host keys.** Check the login node's SSH host key against the fingerprints [published by ZIH](https://compendium.hpc.tu-dresden.de/access/key_fingerprints/). Do not accept a key that does not match.

## One-time setup (login node)

1. Copy `cluster/setup.sh` to the login node, or fetch it from the repository.
2. Run it. It clones this repository into `~/occlusion-memory`, creates a virtualenv there and installs `requirements.txt`. On Linux, the PyPI torch wheel is a CUDA build. It needs Python 3.11 or later: load a Python module first, or set `PYTHON`.

```bash
bash setup.sh
```

Settings:

- **`OM_MODULES`.** Modules that provide Python 3.11 or later. On Capella:

  ```bash
  OM_MODULES="GCC/13.3.0 Python/3.12.3" bash setup.sh
  ```

  Setup records them in the virtualenv, and every job loads them through `cluster/env.sh`. This is required: the virtualenv's interpreter belongs to the module, and without it fails with `libpython3.12.so.1.0: cannot open shared object file`.
  - Both scripts work from non-login shells, such as `ssh host 'sbatch …'`. `cluster/modules.sh` initializes Lmod when `module` is not defined.
  - When modules are recorded but cannot be loaded, the job stops with an error instead of carrying on without them.
  - Setup picks a Python 3.11+ interpreter itself after loading the modules (on Capella, `python3` stays the system 3.9). Set `PYTHON` only to override that choice.
- **`TORCH_INDEX`.** Use it if the smoke test reports *CUDA unavailable*. PyPI's torch wheel is a CUDA 13 build, and it needs a recent driver. Point this at the PyTorch wheel index for the CUDA version that `nvidia-smi` on a GPU node reports, for example `https://download.pytorch.org/whl/cu128`, and rerun setup. The first smoke job found driver 615.71.09, so this is not needed now.

## GPU smoke test

Runs on one H100 for about 5 minutes; the job's limit is 20 minutes.

```bash
cd ~/occlusion-memory && sbatch cluster/gpu_smoke.sbatch
```

The test is not an experiment. It checks:

- **CUDA:** the GPU is visible, and a bf16 matmul benchmark runs.
- **Rendering:** MuJoCo renders through EGL at 64×64 and 96×96. The physics must not depend on the resolution.
- **Throughput:** training speed of a 0.8M-parameter recurrent image model (CNN, GRU, Gaussian head) at both resolutions. From it the test estimates the hours for 10,000 episodes × 20 epochs, and the dataset size.

The log goes to `cluster/logs/om-gpu-smoke-<jobid>.out` and the results to `results/gpu_smoke_<jobid>.json`. Check the job with:

```bash
squeue -u $USER
```

```bash
sacct -j <jobid> -X
```

## Account and partition

`p_scads_mlhpc` on partition `capella`, set in the `.sbatch` headers.
