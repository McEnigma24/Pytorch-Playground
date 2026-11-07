# shellcheck shell=bash
# Per-repo: który wspólny env używamy (tylko: source ./activate.sh)

export PYTHON_SHARED_ENVS___ML="${HOME}/CODE_SPACE/python_shared_envs/ml"
export UV_PROJECT_ENVIRONMENT="${PYTHON_SHARED_ENVS___ML}/.venv"
export UV_PYTHON=3.12

. "${UV_PROJECT_ENVIRONMENT}/bin/activate"

# jeśli chcemy coś dodać do VENV
# to trzeba tam wejść
# -> go_to_venv.sh
