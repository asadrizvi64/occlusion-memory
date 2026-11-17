#!/bin/bash
#SBATCH --job-name=om-b3-eval
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
# B3 rebuild, step 3: score all runs and B1d on the eval block (b3_eval.py; CPU work, the GPU is idle).
# Writes results/b3_eval_{periods.csv,summary.json,tables.md} in this checkout. Submit through cluster/submit_b3.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${OM_DATA:?set OM_DATA to a directory in the occlusion-memory workspace}"
source cluster/env.sh
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is), commit $(git rev-parse --short HEAD)"
python b3_eval.py "$OM_DATA" "$OM_DATA/runs" results/b3_eval "$SLURM_CPUS_PER_TASK"
