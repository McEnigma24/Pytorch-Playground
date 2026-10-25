#!/usr/bin/env bash
# Wspólna konfiguracja docker compose dla drun / dsh.
set -euo pipefail

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# shellcheck source=_config_/load_config.sh
source "${ROOT}/_config_/load_config.sh"

ENV_FILE="${ROOT}/_config_/config.env"
COMPOSE=(docker compose --env-file "${ENV_FILE}" -f "${ROOT}/docker-compose.yml")
GPU_OK=0

detect_gpu() {
  if [[ "${ML_TEMPLATE_GPU:-auto}" == "0" ]]; then
    return 1
  fi
  if [[ "${ML_TEMPLATE_GPU:-auto}" == "1" ]]; then
    return 0
  fi
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}

if detect_gpu; then
  COMPOSE+=(-f "${ROOT}/docker-compose.gpu.yml")
  GPU_OK=1
fi
