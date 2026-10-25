#!/usr/bin/env bash
set -euo pipefail

cd /workspace

VENV="${UV_PROJECT_ENVIRONMENT:-/workspace/build/venv}"
mkdir -p /workspace/build/uv-cache /workspace/build/python /workspace/build/run /workspace/build/logs

export UV_PROJECT_ENVIRONMENT="${VENV}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/build/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/build/python}"
export PATH="/workspace/bin:${VENV}/bin:/root/.local/bin:${PATH}"

_kernel_name="${KERNEL_NAME:-}"
if [[ -z "${_kernel_name}" ]]; then
  _kernel_name="${PROJECT_NAME//_/-}"
fi
_kernel_display="${PROJECT_DISPLAY_NAME:-${PROJECT_NAME:-ml_template} (Docker)}"

_skip_sync=0
if [[ "${1:-}" == "bash" && "${2:-}" == "-c" ]]; then
  _skip_sync=1
fi
if [[ $_skip_sync -eq 0 ]] && [[ -f pyproject.toml ]]; then
  echo "[entrypoint] uv sync (pyproject.toml → ${VENV})"
  uv sync
fi

_kernel_dir="${VENV}/share/jupyter/kernels/${_kernel_name}"
if [[ -x "${VENV}/bin/python" ]] && [[ ! -f "${_kernel_dir}/kernel.json" ]]; then
  echo "[entrypoint] Rejestruję kernel: ${_kernel_name}"
  "${VENV}/bin/python" -m ipykernel install --sys-prefix \
    --name "${_kernel_name}" \
    --display-name "${_kernel_display}"
fi

print_banner() {
  cat <<EOF

=== ML dev shell (${PROJECT_NAME:-project}) ===
  tb         TensorBoard w tle  -> http://127.0.0.1:${TENSORBOARD_PORT:-6006}/
  jnb        Jupyter Lab w tle  -> http://127.0.0.1:${JUPYTER_PORT:-8888}/
  run_main   python src/main.py (pierwszy plan)
  tb|jnb: stop | status | logs
  exit       zamyka kontener (pierwszy terminal)

EOF
}

if [[ "${1:-}" == "bash" && $# -eq 1 ]]; then
  clear
  print_banner
  exec bash --rcfile /workspace/docker/shell_init.sh
fi

exec "$@"
