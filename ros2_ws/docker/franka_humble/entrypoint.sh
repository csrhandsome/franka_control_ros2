#!/usr/bin/env bash
set -euo pipefail

source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source /opt/franka_ros2_ws/install/setup.bash

exec "$@"
