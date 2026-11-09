#!/usr/bin/env bash
# 1 epoka × 9 architektur (mini — mieści się na słabszym GPU).
#   ./scripts/smoke_all_architectures.sh [--cpu]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "${1:-}" == "--cpu" ]] && export CUDA_VISIBLE_DEVICES=""
unset LANGUAGE_RNN_SAVE_DIR

LOG_ROOT="${TENSORBOARD_LOGDIR:-$ROOT/runs/smoke_all}"
mkdir -p "$LOG_ROOT"

for arch in rnnRNN rnnGRU rnnLSTM rnnBaseLSTM rnnAWDLSTM rnnGoldenLSTM \
  transformer_nanoGPT transformer_miniLLaMA transformer_GPT2small; do
  echo "======== $arch ========"
  export TENSORBOARD_LOGDIR="$LOG_ROOT/$arch"
  uv run python scripts/plgrid/run_train.py --architecture "$arch" --preset smoke \
    --tensorboard-root "$LOG_ROOT/$arch"
done
