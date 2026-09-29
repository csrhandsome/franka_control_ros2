#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /opt/cv_bridge_ws/install/setup.bash
exec "$@"
