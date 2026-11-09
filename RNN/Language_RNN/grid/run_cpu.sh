#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common_"

# ./grid/run_cpu.sh [NTASKS] [CPUS_PER_TASK]           — batch (domyślnie 8 1)
# ./grid/run_cpu.sh {jupyter|jupy|j|jp} [NTASKS] [CPUS_PER_TASK]  — interaktywny węzeł, max 2 h

MODE="${1:-batch}"
TIME="${GRID_JUPYTER_TIME:-02:00:00}"
MEM="${GRID_CPU_MEM:-16G}"
ACCOUNT="${GRID_CPU_ACCOUNT:-plgar2026-cpu}"
PARTITION="${GRID_CPU_PARTITION:-cpu-lowprio}"

if is_jupyter_mode "$MODE"; then
  NTASKS="${2:-1}"
  CPUS_PER_TASK="${3:-4}"
  echo "Interaktywny CPU (Jupyter): ntasks=${NTASKS} cpus-per-task=${CPUS_PER_TASK} time=${TIME}"
  exec srun \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name=lrnn_jupyter_cpu \
    --ntasks="$NTASKS" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --export=ALL,OMP_NUM_THREADS="$CPUS_PER_TASK" \
    --pty "$GRID_ROOT/local/jupyter_shell.sh" cpu
fi

clear
rm -f output/*

NTASKS="${1:-8}"
CPUS_PER_TASK="${2:-1}"

sbatch --parsable --ntasks="$NTASKS" --cpus-per-task="$CPUS_PER_TASK" \
  --export=ALL,OMP_NUM_THREADS="$CPUS_PER_TASK" \
  grid/job_cpu | tee grid/.latest_job_started
