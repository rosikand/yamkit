#!/usr/bin/env bash
# One-time system setup so the CAN adapters come up by themselves — at boot, on hot-plug and after
# a bus-off — instead of running scripts/can_up.sh after every reboot.
#
# What it does (the ONLY thing yamkit ever installs outside its own directory):
#   * copies system/80-yam-can.network to /etc/systemd/network/
#   * enables systemd-networkd (it only touches interfaces named can*; NetworkManager keeps
#     managing wifi/ethernet exactly as before)
# Needs sudo. Safe to re-run. Undo with:  scripts/install_system.sh --uninstall
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/system/80-yam-can.network"
DST="/etc/systemd/network/80-yam-can.network"
SUDO=""; [[ "$(id -u)" != "0" ]] && SUDO="sudo"

if ! command -v systemctl >/dev/null; then
  echo "no systemd on this machine — bring the adapters up with scripts/can_up.sh instead"; exit 1
fi

if [[ "${1:-}" == "--uninstall" ]]; then
  $SUDO rm -f "$DST"
  $SUDO networkctl reload 2>/dev/null || true
  echo "removed $DST (systemd-networkd itself was left enabled; 'sudo systemctl disable --now systemd-networkd' if you want it gone)"
  exit 0
fi

if cmp -s "$SRC" "$DST" && systemctl is-active --quiet systemd-networkd; then
  echo "already installed: $DST, systemd-networkd active"
else
  echo "installing $DST and enabling systemd-networkd (sudo)"
  $SUDO install -D -m 644 "$SRC" "$DST"
  $SUDO systemctl enable --now systemd-networkd >/dev/null
  $SUDO networkctl reload 2>/dev/null || $SUDO systemctl restart systemd-networkd
fi

# Adapters that are plugged in right now get configured by the reload; show the result.
for _ in 1 2 3 4 5 6; do
  down=0
  for d in /sys/class/net/*; do
    [[ "$(cat "$d/type" 2>/dev/null)" == "280" ]] || continue
    ip link show "$(basename "$d")" | grep -q "state UP" || down=1
  done
  [[ $down -eq 0 ]] && break
  sleep 0.5
done
ip -brief link show type can 2>/dev/null || true
echo "done — CAN adapters now come up automatically (at boot, on hot-plug, after bus-off)."
