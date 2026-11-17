#!/bin/bash
# Submits the B3 rebuild as one chain: data -> 12 training runs (after the data job succeeds) -> evaluation (after all
# training runs succeed). Prints the job IDs. Run only after the user's go-ahead, from this project's own checkout
# (~/occlusion-memory), with OM_DATA pointing into this project's workspace. Never touches other projects' checkouts.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OM_DATA:?set OM_DATA to a directory in the occlusion-memory workspace}"
mkdir -p cluster/logs "$OM_DATA"
d=$(sbatch --parsable --export=ALL,OM_DATA="$OM_DATA" cluster/b3_data.sbatch)
t=$(sbatch --parsable --dependency=afterok:"$d" --export=ALL,OM_DATA="$OM_DATA" cluster/b3_train.sbatch)
e=$(sbatch --parsable --dependency=afterok:"$t" --export=ALL,OM_DATA="$OM_DATA" cluster/b3_eval.sbatch)
echo "submitted: data $d, training array $t (12 tasks), evaluation $e"
