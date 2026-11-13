#!/bin/bash
# 6× RNN, 1 epoka, preset ~125M (jak src/main.py) — array na GPU V100.
source "$(dirname "${BASH_SOURCE[0]}")/_common_"

mkdir -p grid/output
export LANGUAGE_RNN_ROOT="$GRID_ROOT"

sbatch --parsable grid/job_rnn_array | tee grid/.latest_job_started
echo "Array job id: $(cat grid/.latest_job_started)"
squeue -u "$USER" | head -12
