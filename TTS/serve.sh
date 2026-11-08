#!/bin/bash

clear

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate.sh"

export PYTHONPATH="${SCRIPT_DIR}/src"
exec python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
