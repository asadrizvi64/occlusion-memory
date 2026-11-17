#!/bin/bash
#SBATCH --job-name=om-b3-data
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
# B3 rebuild, step 1: render the development episodes (make_rgb_data.py) into $OM_DATA, about 37 GB:
# train 10,000 / val 500 / eval 1,000, generator version 3, at 64x64 and 96x96. The GPU is used only for EGL
# rendering. Rerunning resumes from the last complete shard. Submit through cluster/submit_b3.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${OM_DATA:?set OM_DATA to a directory in the occlusion-memory workspace}"
source cluster/env.sh
echo "job $SLURM_JOB_ID on $(hostname), $(date -Is), commit $(git rev-parse --short HEAD), OM_DATA=$OM_DATA"
python make_rgb_data.py "$OM_DATA" train,val,eval "$SLURM_CPUS_PER_TASK"
du -sh "$OM_DATA"/*
