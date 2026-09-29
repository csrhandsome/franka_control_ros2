#!/usr/bin/env bash
# Separate, device-free Jazzy experiment. Uses the existing host .env values.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_dir="$(dirname -- "$script_dir")"
env_file="${config_dir}/../franka_humble/.env"

exec docker compose -f "${config_dir}/docker-compose.yml" --env-file "$env_file" "$@"
