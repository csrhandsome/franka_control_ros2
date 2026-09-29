#!/usr/bin/env bash
# Run this as: bash ros2_ws/docker/franka_humble/scripts/compose.sh <compose arguments>
# If the current terminal has not yet picked up the docker group, prefix it with
# `sg docker -c 'bash ros2_ws/docker/franka_humble/scripts/compose.sh …'`.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_dir="$(dirname -- "$script_dir")"

exec docker compose \
  -f "$config_dir/docker-compose.yml" \
  -f "$config_dir/docker-compose.proxy.yml" \
  -f "$config_dir/docker-compose.build-host-network.yml" \
  -f "$config_dir/docker-compose.host-proxy.yml" \
  --env-file "$config_dir/.env" \
  "$@"
