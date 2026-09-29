#!/usr/bin/env bash
# USB/video plus DH5 serial. Requires DH5_SERIAL_DEVICE (or /dev/ttyUSB0) on the host.
# Usage: bash ros2_ws/docker/franka_humble/scripts/compose_dh5.sh run --rm franka_humble bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_dir="$(dirname -- "$script_dir")"

exec docker compose \
  -f "$config_dir/docker-compose.yml" \
  -f "$config_dir/docker-compose.proxy.yml" \
  -f "$config_dir/docker-compose.build-host-network.yml" \
  -f "$config_dir/docker-compose.host-proxy.yml" \
  -f "$config_dir/docker-compose.runtime-safe.yml" \
  -f "$config_dir/docker-compose.uv.yml" \
  -f "$config_dir/docker-compose.overlay.yml" \
  -f "$config_dir/docker-compose.devices.yml" \
  -f "$config_dir/docker-compose.dh5.yml" \
  --env-file "$config_dir/.env" \
  "$@"
