#!/usr/bin/env bash
# Minimalna instalacja NA WSL (poza Dockerem).
# Python, uv i pakiety — w kontenerze (uv sync).
#
# Wymaga: Ubuntu na WSL2 + sterownik NVIDIA na Windows (Game Ready / Studio).
#
# Uruchom:  sudo ./_config_/first_time_init.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=_config_/load_config.sh
source "${ROOT}/_config_/load_config.sh"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Uruchom: sudo $0" >&2
  exit 1
fi

echo "=== 1/4 — pakiety bazowe na WSL (tylko to, co musi być na hoście) ==="
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  gnupg \
  apt-transport-https

echo ""
echo "=== 2/4 — Docker Engine (jeśli brak) ==="
if ! command -v docker >/dev/null 2>&1; then
  echo "Instaluję Docker CE…"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  echo "Dodaj użytkownika do grupy docker:  sudo usermod -aG docker \$USER"
else
  echo "Docker już zainstalowany: $(docker --version)"
fi

echo ""
echo "=== 3/4 — NVIDIA Container Toolkit (CUDA w Dockerze) ==="
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker 2>/dev/null || service docker restart

echo ""
echo "=== 4/4 — weryfikacja GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "UWAGA: brak nvidia-smi w WSL."
  echo "Zainstaluj sterownik NVIDIA na Windows (WSL2 CUDA support)."
fi

echo ""
echo "Test CUDA w Dockerze:"
docker run --rm --gpus all nvidia/cuda:12.9.2-base-ubuntu22.04 nvidia-smi

cat <<EOF

=== Gotowe na hoście WSL ===
Zainstalowane / wymagane POZA kontenerem:
  • Docker Engine + compose plugin
  • NVIDIA Container Toolkit
  • Sterownik NVIDIA (Windows → WSL2)

NIE instaluj na hoście (kontener + uv):
  • Python, pip, PyTorch, Jupyter

Następne kroki (bez sudo, z katalogu projektu):
  cd ${ROOT}
  docker compose --env-file _config_/config.env build
  ./scripts/drun sync
  ./scripts/drun verify   # wymaga GPU w Dockerze
  ./shell.sh              # w środku: tb, jnb, run_main

Projekt (compose): ${PROJECT_NAME}
EOF
