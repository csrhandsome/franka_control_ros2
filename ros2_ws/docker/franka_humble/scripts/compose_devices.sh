#!/usr/bin/env bash
# Runtime overlay that also passes USB/video devices. Do not use as the default.
# Usage: bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm franka_humble bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_dir="$(dirname -- "$script_dir")"

# Compose devices entries do not expand globs. Discover video nodes on every
# invocation so both cameras are mapped even when their device numbers change.
video_overlay="$(mktemp "${TMPDIR:-/tmp}/franka-video-devices.XXXXXX.yml")"
trap 'rm -f -- "$video_overlay"' EXIT
video_nodes=()
declare -A video_groups=()
for video_node in /dev/video[0-9]*; do
  [[ -c "$video_node" ]] || continue
  video_nodes+=("$video_node")
  video_groups["$(stat -c '%g' "$video_node")"]=1
done

{
  printf 'services:\n  franka_humble:\n'
  if (( ${#video_nodes[@]} )); then
    printf '    devices:\n'
    for video_node in "${video_nodes[@]}"; do
      printf '      - "%s:%s:rw"\n' "$video_node" "$video_node"
    done
    # Numeric host GIDs also work if video has a different GID in the image.
    printf '    group_add:\n'
    for video_gid in "${!video_groups[@]}"; do
      printf '      - "%s"\n' "$video_gid"
    done
  else
    printf '    devices: []\n'
  fi
} > "$video_overlay"

docker compose \
  -f "$config_dir/docker-compose.yml" \
  -f "$config_dir/docker-compose.proxy.yml" \
  -f "$config_dir/docker-compose.build-host-network.yml" \
  -f "$config_dir/docker-compose.host-proxy.yml" \
  -f "$config_dir/docker-compose.runtime-safe.yml" \
  -f "$config_dir/docker-compose.uv.yml" \
  -f "$config_dir/docker-compose.overlay.yml" \
  -f "$config_dir/docker-compose.devices.yml" \
  -f "$video_overlay" \
  --env-file "$config_dir/.env" \
  "$@"
