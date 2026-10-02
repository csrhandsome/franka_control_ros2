#!/usr/bin/env bash
# ROS 2 Humble setup scripts read optional variables that may be unset, so do
# not enable `set -u` here.  Keep this entrypoint minimal and shell-compatible.
set -eo pipefail

source /opt/ros/humble/setup.bash
if [[ -f /opt/panda_ws/install/setup.bash ]]; then
  source /opt/panda_ws/install/setup.bash
  export LD_LIBRARY_PATH="/opt/panda_libfranka/lib:${LD_LIBRARY_PATH:-}"
elif [[ -f /opt/franka_ros2_ws/install/setup.bash ]]; then
  source /opt/franka_ros2_ws/install/setup.bash
fi
if [[ -f /opt/overlay_ws/install/local_setup.bash ]]; then
  # Do not source setup.bash: its recorded underlay could reintroduce FR3 libraries.
  source /opt/overlay_ws/install/local_setup.bash
fi

export UV_PROJECT="${UV_PROJECT:-/workspace/data_collect/ros2_ws/docker/franka_humble/python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/uv/venv}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-/opt/uv/venv}"
export UV_PYTHON="${UV_PYTHON:-/usr/bin/python3}"
if [[ -d /opt/uv/venv/bin ]]; then
  export PATH="/opt/uv/venv/bin:${PATH}"
fi

exec "$@"
