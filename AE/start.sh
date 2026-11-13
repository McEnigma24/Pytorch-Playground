#!/bin/bash
# Użycie:
#   ./start                    — nowy run
#   ./start latest             — wznowienie z najnowszego runu tej architektury (nowy katalog)
#   ./start saved_models/...   — wznowienie z podanego katalogu runu
#
# Zmienna LANGUAGE_RNN_RESUME (latest lub ścieżka) — to samo co pierwszy argument.
# Porównanie metryk (multi-line PNG): python src/metrics/plot_metrics.py

set -euo pipefail

clear

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate.sh"

export AE_RESUME="${1:-${AE_RESUME:-}}"
export LANGUAGE_RNN_RESUME="${AE_RESUME}"
unset AE_SAVE_DIR LANGUAGE_RNN_CHECKPOINT

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
python src/main.py

echo ""
