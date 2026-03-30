#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

"$ROOT_DIR/setup.sh"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export RMBG_HOST="${RMBG_HOST:-127.0.0.1}"
export RMBG_PORT="${RMBG_PORT:-7860}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  export RMBG_DEVICE="${RMBG_DEVICE:-cpu}"
fi

exec python "$ROOT_DIR/app/app.py"
