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

Optional settings:

- **`OM_MODULES`.** Modules that provide the Python, for example the cluster's Python 3.12 module. Setup loads them and records them in the virtualenv, and every job loads them again through `cluster/env.sh`. The virtualenv's interpreter belongs to the module and may not start without it. You can rerun setup on an existing checkout just to record them:

  ```bash
  OM_MODULES="<the Python 3.12 module>" bash ~/occlusion-memory/cluster/setup.sh
  ```

- **`TORCH_INDEX`.** Use it if the smoke test reports *CUDA unavailable*. PyPI's torch wheel is a CUDA 13 build, and it needs a recent driver. Point this at the PyTorch wheel index for the CUDA version that `nvidia-smi` on a GPU node reports, for example `https://download.pytorch.org/whl/cu128`, and rerun setup.

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
