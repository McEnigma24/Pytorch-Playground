#!/usr/bin/env bash
# Source from host scripts: source "$ROOT/_config_/load_config.sh"
set -euo pipefail

_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CONFIG_FILE="${_CONFIG_DIR}/config.env"

if [[ ! -f "${_CONFIG_FILE}" ]]; then
  echo "Brak pliku konfiguracji: ${_CONFIG_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${_CONFIG_FILE}"
set +a

if [[ -z "${PROJECT_NAME:-}" ]]; then
  echo "Ustaw PROJECT_NAME w ${_CONFIG_FILE}" >&2
  exit 1
fi

export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export IMAGE_NAME="${PROJECT_NAME}:dev"
export UV_PROJECT_NAME="${PROJECT_NAME//_/-}"
export KERNEL_NAME="${UV_PROJECT_NAME}"
export KERNEL_DISPLAY_NAME="${PROJECT_DISPLAY_NAME:-${PROJECT_NAME} (Docker)}"
export SHELL_CONTAINER_NAME="${PROJECT_NAME}-shell"

_PYPROJECT="${_CONFIG_DIR}/../pyproject.toml"
if [[ -f "${_PYPROJECT}" ]] && ! grep -q "^name = \"${UV_PROJECT_NAME}\"$" "${_PYPROJECT}"; then
  sed -i -E "0,/^name = \".*\"$/s//name = \"${UV_PROJECT_NAME}\"/" "${_PYPROJECT}"
  echo "[config] pyproject.toml: name = ${UV_PROJECT_NAME}" >&2
fi
