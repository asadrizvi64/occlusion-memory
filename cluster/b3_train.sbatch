#!/bin/bash
#SBATCH --job-name=om-b3-train
#SBATCH --account=p_scads_mlhpc
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=cluster/logs/%x-%A_%a.out
# B3 rebuild, step 2: one B3-RGB training run per array task (b3_rgb.py), then its predictions on the eval block.
# Task i: resolution 64 (i < 6) or 96; inputs rgb (i mod 6 < 3) or rgb+meas; seed i mod 3.
# Runs go to $OM_DATA/runs/b3-<inputs>-<res>-s<seed>/. Submit through cluster/submit_b3.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${OM_DATA:?set OM_DATA to a directory in the occlusion-memory workspace}"
source cluster/env.sh
i=$SLURM_ARRAY_TASK_ID
res=$(( i < 6 ? 64 : 96 ))
if (( (i % 6) < 3 )); then inputs=rgb; else inputs=rgb+meas; fi
seed=$(( i % 3 ))
run="$OM_DATA/runs/b3-${inputs}-${res}-s${seed}"
echo "job ${SLURM_ARRAY_JOB_ID}_${i} on $(hostname), $(date -Is), commit $(git rev-parse --short HEAD): res $res, inputs $inputs, seed $seed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python b3_rgb.py train "$OM_DATA" "$run" --res "$res" --inputs "$inputs" --seed "$seed"
