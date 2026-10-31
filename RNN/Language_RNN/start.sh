#!/bin/bash
# Użycie:
#   ./start                          — nowy run (saved_models/rnnLSTM-YYYY.MM.DD-HHMMSS/)
#   experiment w src/main.py: rnnLSTM | rnnGRU | rnnRNN
#   ./start <checkpoint.pt> <save_dir> — wznowienie: wagi z arg1, zapis do arg2
#
# Przykład:
#   ./start saved_models/rnnLSTM-2026.10.30/checkpoints/epoch_0010.pt saved_models/rnnLSTM-2026.10.30-cont
#
# Porównanie metryk (multi-line PNG): python src/metrics/plot_metrics.py

set -euo pipefail

clear
export LANGUAGE_RNN_CHECKPOINT="${1:-}"
export LANGUAGE_RNN_SAVE_DIR="${2:-}"
python src/main.py
echo ""
