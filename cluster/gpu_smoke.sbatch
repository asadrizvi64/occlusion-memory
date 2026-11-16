#!/bin/bash
#SBATCH --job-name=om-gpu-smoke
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out
# Submit from the occlusion-memory checkout:  cd ~/occlusion-memory && sbatch cluster/gpu_smoke.sbatch
# The log path is relative to the submit directory; cluster/setup.sh creates cluster/logs.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is), commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python cluster/gpu_smoke.py
