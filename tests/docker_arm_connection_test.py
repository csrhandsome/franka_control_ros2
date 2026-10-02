"""Ordinary live tests of RoboticArmControlerRos; run only in Humble Docker.

Start the existing ROS bringup, then run from the repository root:
  bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
    franka_humble python tests/docker_arm_connection_test.py

All movement uses this class. Tests use a 0.006 rad joint-1 offset and a
0.004 m base-frame +Z offset, then return to the measured starting state.
The extended EE stream follows two smooth X/Z ellipses (0.04 m wide,
0.05 m high) over 40 seconds at 100 Hz, keeping the initial orientation.
Select only EE tests with ``-k ee -f``; select only the extended stream with
``RoboticArmControlerRosTest.test_09_ee_continuous_motion``.
Errors are reported by unittest with their original tracebacks.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EE_LOOP_DURATION_S = 40.0
EE_LOOP_CYCLES = 2
EE_LOOP_RADII_M = np.array([0.02, 0.025])


def ee_loop_offset(fraction):
    """Start at the ellipse's bottom with zero velocity and acceleration."""
    smooth = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
    phase = 2.0 * math.pi * EE_LOOP_CYCLES * smooth
    return np.array([
        EE_LOOP_RADII_M[0] * math.sin(phase),
        0.0,
        EE_LOOP_RADII_M[1] * (1.0 - math.cos(phase)),
    ])


class RoboticArmControlerRosTest(unittest.TestCase):
    def setUp(self):
        from control.robotic_arm_controller_ros import RoboticArmControlerRos

        self.arm = RoboticArmControlerRos(
            config_path="config/planer/panda.yaml",
            use_fake_hardware=os.environ.get("FRANKA_TEST_FAKE_HARDWARE", "false").lower() == "true",
            # Request completion inside the existing 1 mm arrival assertion.
            motion_config={"ee": {"completion": {"translation_tolerance_m": 0.0008}}},
        )
        self.addCleanup(self.arm.close)
        self.arm.connect(timeout_s=5.0)
        self.arm.wait_ready(timeout_s=5.0)

    def tearDown(self):
        import rclpy

        self.arm.close()
        self.arm.close()  # close() must be idempotent.
        self.assertFalse(self.arm.get_status()["connected"])
        self.assertFalse(any(self.arm.get_state(required=())["valid"].values()))
        self.assertEqual(self.arm._ros_entities, [])
        self.assertFalse(self.arm._spin_thread.is_alive())
        self.assertFalse(self.arm._gripper_thread.is_alive())
        self.assertEqual(self.arm._executor.get_nodes(), [])
        self.assertIsNone(self.arm._node.executor)
        self.assertFalse(rclpy.ok(), "Owned ROS context was not shut down")
        for name in ("publishers", "subscriptions", "clients", "waitables"):
            self.assertEqual(list(getattr(self.arm._node, name)), [], f"Remaining {name}")
        threads = list(self.arm._executor._executor._threads)
        deadline = time.monotonic() + 1.0
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        self.assertEqual([thread.name for thread in threads if thread.is_alive()], [],
                         "Executor worker threads were not released by close()")

    def start_and_target(self, space):
        measured = self.arm.get_state()
        if space == "joint":
            start = measured["joint_positions"].copy()
            target = start.copy()
            target[0] += 0.006
            return start, target, None
        start = measured["ee_position"].copy()
        target = start.copy()
        target[2] += 0.004
        return start, target, measured["ee_quaternion_xyzw"].copy()

    def measured(self, space):
        key = "joint_positions" if space == "joint" else "ee_position"
        return self.arm.get_state()[key]

    def assert_arrived(self, space, target):
        self.arm.wait_until_stopped(timeout_s=5.0)
        np.testing.assert_allclose(self.measured(space), target, atol=0.001, rtol=0)

    def stream(self, space):
        start, target, quaternion = self.start_and_target(space)
        self.arm.start_stream(space)
        samples = []
        begin = time.monotonic()
        while True:
            fraction = min((time.monotonic() - begin) / 4.0, 1.0)
            blend = (1.0 - math.cos(2.0 * math.pi * fraction)) / 2.0
            command = start + blend * (target - start)
            if space == "joint":
                receipt = self.arm.send_joint_target(command)
            else:
                receipt = self.arm.send_ee_target(command, quaternion)
            self.assertEqual(receipt["space"], space)
            samples.append(self.measured(space).copy())
            if fraction == 1.0:
                break
            time.sleep(0.02)
        self.arm.stop_stream(timeout_s=5.0)
        self.assertIsNone(self.arm.get_status()["stream_space"])
        self.assertGreater(len(samples), 20)
        self.assertGreater(np.max(np.abs(np.asarray(samples) - start)),
                           np.max(np.abs(target - start)) / 2)
        self.assert_arrived(space, start)

    def direct_move(self, space):
        start, target, quaternion = self.start_and_target(space)
        begin = time.monotonic()
        if space == "joint":
            result = self.arm.move_joints(target, duration_s=4.0)
        else:
            result = self.arm.move_ee(target, quaternion, duration_s=4.0)
        self.assertEqual(result["status"], "succeeded")
        self.assertGreaterEqual(time.monotonic() - begin, 4.0)
        self.assert_arrived(space, target)
        if space == "joint":
            result = self.arm.move_joints(start, duration_s=4.0)
        else:
            result = self.arm.move_ee(start, quaternion, duration_s=4.0)
        self.assertEqual(result["status"], "succeeded")
        self.assert_arrived(space, start)

    def trajectory_points(self, start, target, quaternion):
        values = ((0.0, start), (2.0, target), (4.0, start))
        if quaternion is None:
            return [{"time_s": t, "positions": value, "velocities": np.zeros(7),
                     "accelerations": np.zeros(7)} for t, value in values]
        return [{"time_s": t, "position": value, "quaternion_xyzw": quaternion}
                for t, value in values]

    def trajectory(self, space):
        start, target, quaternion = self.start_and_target(space)
        samples = []
        key = "joint_positions" if space == "joint" else "ee_position"
        method = (self.arm.execute_joint_trajectory if space == "joint"
                  else self.arm.execute_ee_trajectory)
        result = method(self.trajectory_points(start, target, quaternion),
                        on_sample=lambda measured: samples.append(measured[key].copy()))
        self.assertEqual(result["status"], "succeeded")
        self.assertGreater(len(samples), 10)
        self.assertGreater(np.max(np.abs(np.asarray(samples) - start)),
                           np.max(np.abs(target - start)) / 2)
        self.assert_arrived(space, start)

    def cancellation(self, space):
        start, target, quaternion = self.start_and_target(space)
        started = threading.Event()
        method = (self.arm.execute_joint_trajectory if space == "joint"
                  else self.arm.execute_ee_trajectory)
        with ThreadPoolExecutor(max_workers=1) as pool:
            execution = pool.submit(method, self.trajectory_points(start, target, quaternion),
                                    on_sample=lambda _: started.set())
            if not started.wait(5.0):
                execution.result(timeout=5.0)  # Report the original movement exception.
                self.fail("Trajectory did not produce any state samples")
            cancelled = self.arm.cancel_motion(timeout_s=5.0)
            result = execution.result(timeout=5.0)
        self.assertEqual(cancelled, result)
        self.assertEqual(result["status"], "cancelled")
        self.arm.wait_until_stopped(timeout_s=5.0)
        if space == "joint":
            self.arm.move_joints(start, duration_s=4.0)
        else:
            self.arm.move_ee(start, quaternion, duration_s=4.0)
        self.assert_arrived(space, start)

    def test_01_joint_stream(self):
        self.stream("joint")

    def test_02_joint_direct_move(self):
        self.direct_move("joint")

    def test_03_ee_stream(self):
        self.stream("ee")

    def test_04_ee_direct_move(self):
        self.direct_move("ee")

    def test_05_joint_trajectory(self):
        self.trajectory("joint")

    def test_06_ee_trajectory(self):
        self.trajectory("ee")

    def test_07_joint_cancel(self):
        self.cancellation("joint")

    def test_08_ee_cancel(self):
        self.cancellation("ee")

    def test_09_ee_continuous_motion(self):
        from control.util.pose import quat_angle_xyzw

        initial = self.arm.get_state()
        start = initial["ee_position"].copy()
        quaternion = initial["ee_quaternion_xyzw"].copy()
        samples = []
        rotation_errors = []
        trace_path = os.environ.get("FRANKA_TEST_EE_TRACE_PATH")
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        self.arm.start_stream("ee")
        begin = time.monotonic()
        period = 0.01
        try:
            while True:
                elapsed = time.monotonic() - begin
                fraction = min(elapsed / EE_LOOP_DURATION_S, 1.0)
                command = start + ee_loop_offset(fraction)
                receipt = self.arm.send_ee_target(command, quaternion)
                self.assertEqual(receipt["space"], "ee")
                measured = self.arm.get_state()
                samples.append(np.concatenate([
                    [elapsed, measured["stamp_s"]["ee"]],
                    command, measured["ee_position"], measured["ee_quaternion_xyzw"],
                ]))
                rotation_errors.append(quat_angle_xyzw(
                    measured["ee_quaternion_xyzw"], quaternion
                ))
                # Keep publishing the starting pose for two seconds before stop_stream()
                # replaces the target with a measured hold pose.
                if elapsed >= EE_LOOP_DURATION_S + 2.0:
                    break
                time.sleep(max(0.0, period - (time.monotonic() - begin - elapsed)))
        finally:
            try:
                self.arm.stop_stream(timeout_s=5.0)
            finally:
                if trace_path and samples:
                    np.savetxt(trace_path, np.asarray(samples), delimiter=",", header=(
                        "elapsed_s,ee_stamp_s,target_x_m,target_y_m,target_z_m,"
                        "measured_x_m,measured_y_m,measured_z_m,qx,qy,qz,qw"
                    ), comments="")

        trace = np.asarray(samples)
        tracking_error = np.linalg.norm(trace[:, 5:8] - trace[:, 2:5], axis=1)
        spans = np.ptp(trace[:, 5:8], axis=0)
        return_error = np.linalg.norm(self.measured("ee") - start)
        print(
            f"\nEE continuous: {len(samples)} samples, "
            f"X/Z spans {spans[0] * 1000:.2f}/{spans[2] * 1000:.2f} mm, "
            f"max tracking error {tracking_error.max() * 1000:.2f} mm, "
            f"return error {return_error * 1000:.3f} mm, "
            f"max orientation error {max(rotation_errors):.5f} rad",
            flush=True,
        )
        self.assertIsNone(self.arm.get_status()["stream_space"])
        self.assertGreater(len(samples), EE_LOOP_DURATION_S * 80)
        for low, high in ((0.0, EE_LOOP_DURATION_S / 2),
                          (EE_LOOP_DURATION_S / 2, EE_LOOP_DURATION_S)):
            loop = trace[(trace[:, 0] >= low) & (trace[:, 0] < high), 5:8]
            self.assertGreater(np.ptp(loop[:, 0]), 0.8 * 2 * EE_LOOP_RADII_M[0])
            self.assertGreater(np.ptp(loop[:, 2]), 0.8 * 2 * EE_LOOP_RADII_M[1])
        self.assertLess(tracking_error.max(), 0.01)
        self.assertLess(max(rotation_errors), 0.02)
        self.assertLess(return_error, 0.001)
        self.assert_arrived("ee", start)


if __name__ == "__main__":
    if not Path("/.dockerenv").exists() or os.environ.get("ROS_DISTRO") != "humble":
        raise SystemExit("Run this live test in Humble Docker using compose_safe.sh")
    unittest.main(verbosity=2)
