#!/usr/bin/env bash
# Verify ROS 2, ros2_control and the FR3 model without opening an FCI connection.
set -euo pipefail

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

ros2 launch franka_bringup franka.launch.py \
  robot_type:=fr3 \
  robot_ip:=dont-care \
  use_fake_hardware:=true \
  load_gripper:=false >"${log_file}" 2>&1 &
launch_pid=$!

for _ in $(seq 1 30); do
  if ros2 node list 2>/dev/null | grep -q 'controller_manager'; then
    break
  fi
  sleep 1
done

if ! ros2 node list | grep -q 'controller_manager'; then
  sed -n '1,220p' "${log_file}" >&2
  echo "fake-hardware startup did not expose controller_manager" >&2
  exit 1
fi

echo '--- ROS nodes ---'
ros2 node list
echo '--- controllers ---'
ros2 control list_controllers
echo '--- representative topics ---'
ros2 topic list | grep -E '(^/joint_states$|robot_state|tf)' || true

if ! timeout 8 ros2 topic echo --once /joint_states >/dev/null; then
  sed -n '1,220p' "${log_file}" >&2
  echo 'no /joint_states message from fake hardware' >&2
  exit 1
fi

echo 'PASS: Humble FR3 fake-hardware ROS 2 smoke test completed.'
