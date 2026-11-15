#!/usr/bin/env bash
# Krótki smoke: każdy preset DUAL3060 (~1.5 min, mały korpus).
# W kontenerze: ./scripts/smoke_dual3060.sh
# Z hosta:      ./scripts/drun run ./scripts/smoke_dual3060.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LANGUAGE_RNN_LIGHTNING=1
export LANGUAGE_RNN_TIME_BUDGET_MIN="${LANGUAGE_RNN_TIME_BUDGET_MIN:-1.5}"
export LANGUAGE_RNN_MAX_CORPUS_CHARS="${LANGUAGE_RNN_MAX_CORPUS_CHARS:-500000}"
unset LANGUAGE_RNN_SAVE_DIR

if [[ -f /workspace/build/venv/bin/python ]]; then
  exec uv run python scripts/smoke_dual3060.py
fi
exec ./scripts/drun run uv run python scripts/smoke_dual3060.py