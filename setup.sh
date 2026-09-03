#!/usr/bin/env bash
# One-shot bootstrap on a fresh Linux box:
#     git clone <repo> yamkit && cd yamkit && ./setup.sh
# Installs everything *inside this directory*: uv (if missing), Python 3.12, the venv, all deps.
# Then offers the one system change yamkit needs (boot-time CAN bring-up, one sudo prompt) and
# creates configs/rig.yaml from the attached arms and cameras. Safe to re-run.
# Needs: internet, gcc/g++/make (one dependency builds from source), a SocketCAN kernel (any Ubuntu).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export UV_PYTHON_INSTALL_DIR="$ROOT/.uv-python"
NO_SYSTEM=0; [ "${1:-}" = "--no-system" ] && NO_SYSTEM=1   # skip the boot-time CAN install

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

step "boot-time CAN bring-up (the one system change yamkit makes)"
# Installs system/80-yam-can.network + enables systemd-networkd so the adapters come up at boot and
# on hot-plug. Skipped with --no-system, or when there is no terminal to ask on.
if [ "$NO_SYSTEM" = 1 ]; then
  echo "skipped (--no-system); later:  scripts/install_system.sh"
elif [ -f /etc/systemd/network/80-yam-can.network ] && systemctl is-active --quiet systemd-networkd 2>/dev/null; then
  echo "already installed"
elif [ -t 0 ]; then
  printf 'Install it now? It asks for your sudo password once. [Y/n] '
  read -r ans
  case "${ans:-Y}" in
    [Yy]*) scripts/install_system.sh || echo "install failed — you can retry later with scripts/install_system.sh, or use scripts/can_up.sh after each boot";;
    *) echo "skipped; later:  scripts/install_system.sh   (or scripts/can_up.sh after each boot)";;
  esac
else
  echo "no terminal to ask on; run  scripts/install_system.sh  once (or scripts/can_up.sh after each boot)"
fi

step "rig file (configs/rig.yaml)"
# Machine-specific, not in git. Created here by passive discovery (no motor is enabled) when the
# adapters are up; otherwise created later with `yamkit discover --write`.
if [ -f configs/rig.yaml ]; then
  echo "exists — keeping it (re-run 'yamkit discover --write' if you changed cables)"
elif ip -brief link show type can 2>/dev/null | grep -q " UP "; then
  .venv/bin/yamkit discover --write || echo "discovery failed — run 'yamkit discover --write' once the arms are powered"
else
  echo "no CAN adapter is up yet — power the arms, then run:  yamkit discover --write"
fi

step "checks"
lsmod 2>/dev/null | grep -q '^gs_usb' || echo "note: gs_usb kernel module not loaded yet — it loads automatically when a CANable adapter is plugged in"
command -v candump >/dev/null || echo "optional: sudo apt install can-utils   (candump/cansend for bus diagnostics)"
.venv/bin/yamkit doctor || true

cat <<MSG

Done. Next steps:
  source scripts/env.sh          # activate (per terminal)
  yamkit ui                      # camera feeds + arm status (never energises a motor)
  yamkit read left_follower      # first contact: the arm stays free to move — check it IS the left one
  yamkit swap left_follower right_follower   # if it was not
MSG
