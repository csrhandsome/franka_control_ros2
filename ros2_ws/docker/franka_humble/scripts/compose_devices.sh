#!/usr/bin/env bash
# Runtime overlay that also passes USB/video devices. Do not use as the default.
# Usage: bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm franka_humble bash
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
  --env-file "$config_dir/.env" \
  "$@"
