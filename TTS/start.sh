#!/bin/bash

clear

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate.sh"

uv sync --quiet
uv run -m tts.infer_Qwen_3_tts

echo ""
