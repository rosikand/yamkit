#!/usr/bin/env bash
# Bring every SocketCAN adapter up at the YAM bitrate (1 Mbit/s). Needs sudo. Idempotent.
#   scripts/can_up.sh          # bring up interfaces that are down
#   scripts/can_up.sh --reset  # down + up every interface (recovers a wedged adapter)
set -euo pipefail
BITRATE="${BITRATE:-1000000}"
RESET=0; [[ "${1:-}" == "--reset" ]] && RESET=1
SUDO=""; [[ "$(id -u)" != "0" ]] && SUDO="sudo"
found=0
for d in /sys/class/net/*; do
  [[ "$(cat "$d/type" 2>/dev/null)" == "280" ]] || continue   # ARPHRD_CAN
  n="$(basename "$d")"; found=1
  if [[ $RESET -eq 0 ]] && ip link show "$n" | grep -q "state UP"; then
    echo "$n: already UP"; continue
  fi
  echo "$n: configuring @ ${BITRATE} bit/s"
  $SUDO ip link set "$n" down || true
  $SUDO ip link set "$n" up type can bitrate "$BITRATE"
done
[[ $found -eq 1 ]] || { echo "no CAN interfaces found (is the gs_usb module loaded / adapters plugged in?)"; exit 1; }
