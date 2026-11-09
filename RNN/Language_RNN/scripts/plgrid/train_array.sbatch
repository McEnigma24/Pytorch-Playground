#!/bin/bash
#SBATCH --job-name=lrnn
#SBATCH --output=logs/lrnn_%A_%a.out
#SBATCH --error=logs/lrnn_%A_%a.err
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
#   mkdir -p logs
#   export LANGUAGE_RNN_ROOT=/ścieżka/do/Language_RNN
#   sbatch scripts/plgrid/train_array.sbatch
#
# Na klastrze: skopiuj tokenizer + corpus (albo pierwszy job zbuduje BPE ~30 min).

set -euo pipefail

ARCHITECTURES=(
  rnnRNN rnnGRU rnnLSTM
  rnnBaseLSTM rnnAWDLSTM rnnGoldenLSTM
  transformer_nanoGPT transformer_miniLLaMA transformer_GPT2small
)
ARCH="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT="${LANGUAGE_RNN_ROOT:-$HOME/CODE_SPACE/_my_stuff_/ML/Pytorch-Playground/RNN/Language_RNN}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src"
export TENSORBOARD_LOGDIR="${SCRATCH:-/tmp}/lrnn_runs/${SLURM_ARRAY_JOB_ID}"
mkdir -p logs

# module load python/3.12 cuda/12.4
# uv sync

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID → $ARCH"

uv run python scripts/plgrid/run_train.py \
  --architecture "$ARCH" \
  --preset auto \
  --tensorboard-root "$TENSORBOARD_LOGDIR"
