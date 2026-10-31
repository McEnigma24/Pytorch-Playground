#!/bin/bash
# Użycie:
#   ./start                          — nowy run (saved_models/lstm-YYYY.MM.DD-HHMMSS/)
#   ./start <checkpoint.pt> <save_dir> — wznowienie: wagi z arg1, zapis do arg2
#
# Przykład:
#   ./start saved_models/lstm-2026.10.30/checkpoints/epoch_0010.pt saved_models/lstm-2026.10.30-cont

set -euo pipefail

clear
export LANGUAGE_RNN_CHECKPOINT="${1:-}"
export LANGUAGE_RNN_SAVE_DIR="${2:-}"
python src/main.py
echo ""
