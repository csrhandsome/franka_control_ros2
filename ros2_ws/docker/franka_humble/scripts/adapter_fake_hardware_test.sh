#!/usr/bin/env bash
# Fake-hardware smoke test for the overlay adapter. Does not launch a real arm.
set -eo pipefail

source /opt/ros/humble/setup.bash
if [[ -f /opt/franka_ros2_ws/install/setup.bash ]]; then
  source /opt/franka_ros2_ws/install/setup.bash
fi
if [[ -f /opt/overlay_ws/install/setup.bash ]]; then
  source /opt/overlay_ws/install/setup.bash
else
  echo "overlay is not built; run ros2_ws/docker/franka_humble/scripts/build_overlay.sh first" >&2
  exit 1
fi

log_file="$(mktemp)"
launch_pid=""

cleanup() {
  if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
  rm -f "${log_file}"
}
trap cleanup EXIT

ros2 launch data_collect_franka franka_data_collect.launch.py \
  robot_type:=fr3 \
  robot_ip:=dont-care \
  use_fake_hardware:=true \
  load_gripper:=true \
  start_cameras:=false >"${log_file}" 2>&1 &
launch_pid=$!

ready=0
for _ in $(seq 1 45); do
  if timeout 3 ros2 topic echo --once /joint_states >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" != 1 ]]; then
  sed -n '1,240p' "${log_file}" >&2
  echo "overlay fake-hardware startup did not publish /joint_states" >&2
  exit 1
fi

echo '--- ROS nodes ---'
ros2 node list || true
echo '--- controllers ---'
ros2 control list_controllers || true

if ! timeout 12 ros2 topic echo --once /joint_states >/dev/null; then
  sed -n '1,240p' "${log_file}" >&2
  echo 'no /joint_states message from fake hardware' >&2
  exit 1
fi

cartesian_ok=0
if ros2 control switch_controllers --activate cartesian_pose_target_controller >/tmp/cartesian_switch.log 2>&1; then
  cartesian_ok=1
  echo "cartesian_pose_target_controller activate: OK"
  python3 - <<'PY'
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

rclpy.init()
node = Node("adapter_fake_pose_pub")
pub = node.create_publisher(
    PoseStamped,
    "/cartesian_pose_target_controller/target_pose",
    qos_profile_sensor_data,
)
msg = PoseStamped()
msg.header.frame_id = "fr3_link0"
msg.pose.orientation.w = 1.0
pub.publish(msg)
rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.shutdown()
print("published one target_pose")
PY
else
  echo "cartesian_pose_target_controller activate: FAILED (expected on some fake setups)"
  sed -n '1,80p' /tmp/cartesian_switch.log || true
fi

if ros2 control switch_controllers --activate joint_position_target_controller >/tmp/joint_switch.log 2>&1; then
  echo "joint_position_target_controller activate: OK"
  python3 - <<'PY'
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

rclpy.init()
node = Node("adapter_fake_joint_pub")
pub = node.create_publisher(
    Float64MultiArray,
    "/joint_position_target_controller/target_joints",
    10,
)
msg = Float64MultiArray()
msg.data = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
pub.publish(msg)
rclpy.spin_once(node, timeout_sec=0.5)
node.destroy_node()
rclpy.shutdown()
print("published one target_joints")
PY
else
  echo "joint_position_target_controller activate: FAILED"
  sed -n '1,80p' /tmp/joint_switch.log >&2 || true
  if [[ "${cartesian_ok}" -eq 0 ]]; then
    exit 1
  fi
fi

if [[ -f /workspace/data_collect/control/franka_ros2_control.py ]]; then
  (cd /workspace/data_collect && python -m control.franka_ros2_control --use-fake-hardware status)
fi

echo "PASS: overlay adapter fake-hardware test completed (cartesian_ok=${cartesian_ok})."
