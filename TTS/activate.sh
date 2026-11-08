# shellcheck shell=bash
# Per-repo: który wspólny env używamy (tylko: source ./activate.sh)

export PYTHON_SHARED_ENVS___ML="${HOME}/CODE_SPACE/python_shared_envs/ml"
export UV_PROJECT_ENVIRONMENT="${PYTHON_SHARED_ENVS___ML}/.venv"
export UV_PYTHON=3.12

if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
  echo "Brak venv: ${UV_PROJECT_ENVIRONMENT}" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${VIRTUAL_ENV:-}" && "${VIRTUAL_ENV}" != "${UV_PROJECT_ENVIRONMENT}" ]]; then
  type deactivate &>/dev/null && deactivate
fi

# shellcheck source=/dev/null
. "${UV_PROJECT_ENVIRONMENT}/bin/activate"
# jeśli chcemy coś dodać do shared VENV   to musimy mieć activate i KONKRETNIE zmienną { UV_PROJECT_ENVIRONMENT }
# albo wejść i tam dodać
# -> go_to_venv.sh
