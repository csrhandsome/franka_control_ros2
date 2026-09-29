#!/usr/bin/env bash
# Host-side wrapper: run the repository's Python tests inside the Humble container.
#
#   bash ros2_ws/docker/franka_humble/scripts/run_tests.sh
#   bash ros2_ws/docker/franka_humble/scripts/run_tests.sh tests/test_inference.py
#
# The inner script is the same file tree mounted at /workspace/data_collect, so it
# is executed from there instead of being quoted into `bash -c`.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
container_scripts="/workspace/data_collect/ros2_ws/docker/franka_humble/scripts"

exec bash "${script_dir}/compose_safe.sh" run --rm franka_humble \
  bash "${container_scripts}/pytest_in_container.sh" "$@"
