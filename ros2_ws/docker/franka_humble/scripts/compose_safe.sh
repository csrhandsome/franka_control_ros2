#!/usr/bin/env bash
# Wrapper for the complete local configuration. Usage:
#   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble ...
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
  --env-file "$config_dir/.env" \
  "$@"
