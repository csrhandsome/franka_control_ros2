#!/usr/bin/env bash
# Build the data_collect_franka overlay into /opt/overlay_ws (not the official ws).
set -eo pipefail

source /opt/ros/humble/setup.bash
if [[ -f /opt/panda_ws/install/setup.bash ]]; then
  source /opt/panda_ws/install/setup.bash
  export LD_LIBRARY_PATH="/opt/panda_libfranka/lib:${LD_LIBRARY_PATH:-}"
elif [[ -f /opt/franka_ros2_ws/install/setup.bash ]]; then
  source /opt/franka_ros2_ws/install/setup.bash
fi

src="${DATA_COLLECT_OVERLAY_SRC:-/workspace/data_collect/ros2_ws/src}"
overlay_ws="${OVERLAY_WS:-/opt/overlay_ws}"

sudo mkdir -p "${overlay_ws}"
sudo chown "$(id -u):$(id -g)" "${overlay_ws}"

echo "[FrankaROS2] colcon build overlay from ${src} -> ${overlay_ws}"
cd "${overlay_ws}"
colcon build \
  --symlink-install \
  --base-paths "${src}" \
  --cmake-clean-cache \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

echo "[FrankaROS2] overlay install: ${overlay_ws}/install"
echo "Source it after the Panda underlay with: source ${overlay_ws}/install/local_setup.bash"
