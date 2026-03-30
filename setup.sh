#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/app"
VENV_DIR="$ROOT_DIR/.venv"
STAMP_FILE="$VENV_DIR/.rmbg_setup_complete"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 no esta instalado o no esta en PATH." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

install_torch() {
  local platform
  platform="$(uname -s)-$(uname -m)"

  case "$platform" in
    Darwin-arm64)
      python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
      ;;
    Darwin-x86_64)
      python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
      ;;
    Linux-*)
      python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
      ;;
    *)
      python -m pip install torch torchvision torchaudio
      ;;
  esac
}

if [[ ! -f "$STAMP_FILE" || "$APP_DIR/requirements.txt" -nt "$STAMP_FILE" || "$APP_DIR/app.py" -nt "$STAMP_FILE" ]]; then
  install_torch
  python -m pip install -r "$APP_DIR/requirements.txt"
  touch "$STAMP_FILE"
fi

echo "Entorno listo en $VENV_DIR"
