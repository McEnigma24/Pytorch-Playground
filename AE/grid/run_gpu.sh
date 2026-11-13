#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/_common_"

# ./grid/run_gpu.sh [GPUS] [CPUS_PER_TASK]           — batch (domyślnie 1 4)
# ./grid/run_gpu.sh {jupyter|jupy|j|jp} [GPUS] [CPUS_PER_TASK]  — interaktywny węzeł, max 2 h

MODE="${1:-batch}"
TIME="${GRID_JUPYTER_TIME:-02:00:00}"
MEM="${GRID_GPU_MEM:-32G}"
ACCOUNT="${GRID_GPU_ACCOUNT:-plgar2026-gpu}"
PARTITION="${GRID_GPU_PARTITION:-plgrid-gpu-v100}"

if is_jupyter_mode "$MODE"; then
  GPUS="${2:-1}"
  CPUS_PER_TASK="${3:-4}"
  if [[ "$GPUS" -lt 1 || "$GPUS" -gt 8 ]]; then
    echo "GPUS musi być 1–8" >&2
    exit 1
  fi
  echo "Interaktywny GPU (Jupyter): gpus=${GPUS} cpus-per-task=${CPUS_PER_TASK} time=${TIME}"
  exec srun \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name=lrnn_jupyter_gpu \
    --ntasks=1 \
    --gres="gpu:${GPUS}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time="$TIME" \
    --export=ALL,GPUS="$GPUS",OMP_NUM_THREADS="$CPUS_PER_TASK" \
    --pty "$GRID_ROOT/local/jupyter_shell.sh" gpu
fi

clear
rm -f output/*

GPUS="${1:-1}"
CPUS_PER_TASK="${2:-4}"

sbatch --parsable --ntasks=1 --gres=gpu:"$GPUS" --cpus-per-task="$CPUS_PER_TASK" \
  --export=ALL,GPUS="$GPUS",OMP_NUM_THREADS="$CPUS_PER_TASK" \
  grid/job_gpu | tee grid/.latest_job_started
