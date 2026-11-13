#!/usr/bin/env bash
# Trening + porównanie: rnnAWDLSTM vs transformer_miniLLaMA (preset local6G, ~6 GB VRAM).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset LANGUAGE_RNN_SAVE_DIR
export LANGUAGE_RNN_MAX_CORPUS_CHARS="${LANGUAGE_RNN_MAX_CORPUS_CHARS:-12000000}"
exec uv run python src/main.py
