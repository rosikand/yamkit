#!/usr/bin/env bash
# One-shot bootstrap on a fresh Linux box:
#     git clone <repo> yamkit && cd yamkit && ./setup.sh
# Installs everything *inside this directory*: uv (if missing), Python 3.12, the venv, all deps.
# Needs: internet, gcc/g++/make (one dependency builds from source), a SocketCAN kernel (any Ubuntu).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

step "build tools"
missing=""
for t in gcc g++ make curl; do command -v "$t" >/dev/null || missing="$missing $t"; done
if [ -n "$missing" ]; then
  echo "missing:$missing"
  echo "install them first:  sudo apt install build-essential curl"
  exit 1
fi
echo "ok"

step "uv"
if [ -x "$ROOT/.tools/uv" ]; then
  export PATH="$ROOT/.tools:$PATH"
elif ! command -v uv >/dev/null; then
  echo "uv not found; installing into $ROOT/.tools (nothing outside the repo is touched)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$ROOT/.tools" UV_NO_MODIFY_PATH=1 sh
  export PATH="$ROOT/.tools:$PATH"
fi
uv --version

step "python 3.12 (project-local)"
uv python install 3.12

step "dependencies (this takes a few minutes the first time)"
uv sync --extra dev

step "checks"
lsmod 2>/dev/null | grep -q '^gs_usb' || echo "note: gs_usb kernel module not loaded yet — it loads automatically when a CANable adapter is plugged in"
command -v candump >/dev/null || echo "optional: sudo apt install can-utils   (candump/cansend for bus diagnostics)"
.venv/bin/yamkit doctor || true

cat <<MSG

Done. Next steps:
  source scripts/env.sh          # activate (per terminal)
  scripts/can_up.sh              # bring CAN adapters up at 1 Mbit/s (asks for sudo)
  yamkit discover --write        # only if the CAN adapters differ from configs/rig.yaml
  yamkit read left_leader        # first contact with an arm
MSG
