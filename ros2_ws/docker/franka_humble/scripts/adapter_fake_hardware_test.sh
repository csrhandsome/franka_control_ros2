#!/usr/bin/env bash
# Strict Panda integration test. All motion targets go through the public Python class.
set -eo pipefail

log_file=/tmp/franka-panda-fake-bringup.log
launch_pid=""
cleanup() {
  result=$?
  if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
    kill -INT "${launch_pid}" 2>/dev/null || true
    wait "${launch_pid}" 2>/dev/null || true
  fi
  if [[ "$result" -ne 0 ]]; then
    cat "${log_file}" >&2
  fi
}
trap cleanup EXIT

ros2 launch data_collect_franka franka_data_collect.launch.py \
  robot_type:=panda robot_ip:=dont-care use_fake_hardware:=true \
  load_gripper:=false start_cameras:=false >"${log_file}" 2>&1 &
launch_pid=$!

FRANKA_FAKE_LAUNCH_PID="$launch_pid" python - <<'CHECK'
import os
import time
from control.robotic_arm_controller_ros import RoboticArmControlerRos
last_error = None
for _ in range(30):
    os.kill(int(os.environ["FRANKA_FAKE_LAUNCH_PID"]), 0)
    try:
        with RoboticArmControlerRos(use_fake_hardware=True, motion_config={"gripper": {"enabled": False}}) as arm:
            arm.connect(timeout_s=2.0)
            arm.wait_ready(timeout_s=2.0)
            capabilities = arm.get_capabilities()
            for key in ("joint_stream", "joint_trajectory", "ee_stream", "ee_trajectory"):
                assert capabilities[key]["ready"], capabilities[key]
            print("All four Panda arm capabilities ready on GenericSystem")
            break
    except (RuntimeError, TimeoutError, AssertionError) as exc:
        last_error = exc
        print(f"Waiting for fake bringup: {exc}", flush=True)
        time.sleep(0.5)
else:
    raise RuntimeError("Fake Panda bringup did not become ready") from last_error
CHECK

FRANKA_TEST_FAKE_HARDWARE=true python tests/docker_arm_connection_test.py
