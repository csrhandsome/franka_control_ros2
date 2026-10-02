"""Exercise the synchronous ROS trajectory action on Panda fake hardware only."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.robotic_arm_controller_ros import RoboticArmControlerRos  # noqa: E402


def main():
    if os.environ.get("ROS_DOMAIN_ID") != "145":
        raise RuntimeError("Run this smoke test in the isolated ROS_DOMAIN_ID=145")
    with tempfile.TemporaryFile(mode="w+") as log:
        launch = subprocess.Popen(
            [
                "ros2",
                "launch",
                "franka_bringup",
                "franka.launch.py",
                "robot_ip:=dont-care",
                "use_fake_hardware:=true",
                "load_gripper:=false",
                "use_rviz:=false",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        arm = None
        try:
            for _ in range(20):
                if launch.poll() is not None:
                    raise RuntimeError("Fake bringup exited early")
                ready = subprocess.run(
                    ["timeout", "2", "ros2", "topic", "echo", "--once", "/joint_states"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if ready.returncode == 0:
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Fake bringup did not publish joints")
            subprocess.run(
                [
                    "ros2",
                    "run",
                    "controller_manager",
                    "spawner",
                    "joint_trajectory_controller",
                    "--inactive",
                    "--controller-type",
                    "joint_trajectory_controller/JointTrajectoryController",
                    "--param-file",
                    "/workspace/data_collect/ros2_ws/src/data_collect_franka/config/panda_trajectory_controller.yaml",
                ],
                check=True,
                timeout=30,
            )
            arm = RoboticArmControlerRos(
                config_path="config/planer/panda.yaml",
                use_fake_hardware=True,
                motion_config={"gripper": {"enabled": False}},
            )
            arm.connect()
            start = arm.wait_ready(required=("joints",))["joint_positions"]
            target = start.copy()
            target[0] += 0.02
            samples = []
            result = arm.execute_joint_trajectory(
                [
                    {
                        "time_s": t,
                        "positions": q,
                        "velocities": np.zeros(7),
                        "accelerations": np.zeros(7),
                    }
                    for t, q in ((0.0, start), (2.0, target), (4.0, start))
                ],
                on_sample=lambda state: samples.append(state["joint_positions"].copy()),
            )
            assert result["status"] == "succeeded", result
            assert len(samples) > 20, len(samples)
            measured = np.asarray(samples)
            assert np.ptp(measured[:, 0]) > 0.015, measured[:, 0]
            assert (
                np.max(np.abs(arm.get_state(required=("joints",))["joint_positions"] - start))
                < 0.005
            )
            cancelled_samples = []
            cancellation_due = threading.Event()

            def cancel_sample(state):
                cancelled_samples.append(state["joint_positions"].copy())
                if len(cancelled_samples) == 20:
                    cancellation_due.set()

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    arm.execute_joint_trajectory,
                    [
                        {
                            "time_s": t,
                            "positions": q,
                            "velocities": np.zeros(7),
                            "accelerations": np.zeros(7),
                        }
                        for t, q in ((0.0, start), (4.0, target))
                    ],
                    on_sample=cancel_sample,
                )
                if not cancellation_due.wait(5.0):
                    future.result(timeout=1.0)
                    raise RuntimeError("Trajectory did not produce cancellation samples")
                acknowledged = arm.cancel_motion(timeout_s=5.0)
                cancelled = future.result(timeout=5.0)
                assert acknowledged == cancelled
            assert cancelled["status"] == "cancelled", cancelled
            print(f"PASS: fake Panda synchronous trajectory, {len(samples)} samples, {result}")
            print(f"PASS: ROS action cancellation acknowledged: {cancelled}")
        except BaseException:
            log.seek(0)
            print(log.read()[-12000:])
            raise
        finally:
            if arm is not None:
                arm.close()
            if launch.poll() is None:
                os.killpg(launch.pid, signal.SIGINT)
                try:
                    launch.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(launch.pid, signal.SIGTERM)
                    launch.wait(timeout=5)


if __name__ == "__main__":
    main()
