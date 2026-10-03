#!/usr/bin/env bash
# USB/video plus DH5 serial. Requires DH5_SERIAL_DEVICE (or /dev/ttyUSB0) on the host.
# Usage: bash ros2_ws/docker/franka_humble/scripts/compose_dh5.sh run --rm franka_humble bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_dir="$(dirname -- "$script_dir")"

exec bash "$script_dir/compose_devices.sh" \
  -f "$config_dir/docker-compose.dh5.yml" \
  "$@"
