"""Private target publishing, trajectory execution, and motion validation."""

from __future__ import annotations
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
import numpy as np
from numpy.polynomial import polynomial as poly
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from std_msgs.msg import Float64MultiArray
from control.util.pose import (
    as_vector,
    normalize_quat_xyzw,
    slerp_quat_xyzw,
    matrix_to_quat_xyzw,
    quat_angle_xyzw,
)
from . import lifecycle, state, motion


def _joint_points(points: Sequence[dict], current: np.ndarray, config: dict) -> list[dict]:
    """Check the entire quintic path, including extrema between waypoints."""
    result = []
    previous = -1.0
    for point in points:
        time_s = float(point["time_s"])
        if not np.isfinite(time_s) or time_s < 0 or time_s <= previous:
            raise ValueError("Trajectory times must increase strictly and be nonnegative")
        result.append(
            {
                "time_s": time_s,
                **{
                    key: as_vector(point[key], 7, key)
                    for key in ("positions", "velocities", "accelerations")
                },
            }
        )
        previous = time_s
    if len(result) < 2 or result[0]["time_s"] != 0:
        raise ValueError("Trajectory needs two or more waypoints starting at time zero")
    if (
        np.max(np.abs(result[0]["positions"] - current))
        > config["joint"]["trajectory"]["start_tolerance_rad"]
    ):
        raise ValueError("First trajectory waypoint must match measured joints")
    if np.any(np.abs(result[-1]["velocities"]) > 1e-09) or np.any(
        np.abs(result[-1]["accelerations"]) > 1e-09
    ):
        raise ValueError("Trajectory must finish with zero velocity and acceleration")
    limits = config["limits"]
    if not limits["enabled"]:
        return result
    for start, end in zip(result, result[1:]):
        duration = end["time_s"] - start["time_s"]
        coefficients = np.zeros((6, 7))
        coefficients[:3] = [
            start["positions"],
            start["velocities"] * duration,
            start["accelerations"] * duration**2 / 2,
        ]
        coefficients[3:] = np.linalg.solve(
            [[1, 1, 1], [3, 4, 5], [6, 12, 20]],
            [
                end["positions"] - coefficients[:3].sum(axis=0),
                end["velocities"] * duration - coefficients[1] - 2 * coefficients[2],
                end["accelerations"] * duration**2 - 2 * coefficients[2],
            ],
        )
        for axis in range(7):
            for derivative, limit_key in enumerate(
                (
                    "max_joint_excursion_rad",
                    "max_joint_velocity_rad_s",
                    "max_joint_acceleration_rad_s2",
                )
            ):
                curve = poly.polyder(coefficients[:, axis], derivative) / duration**derivative
                roots = poly.polyroots(poly.polyder(curve))
                locations = [0.0, 1.0] + [
                    float(root.real)
                    for root in roots
                    if abs(root.imag) < 1e-08 and 0 < root.real < 1
                ]
                values = poly.polyval(locations, curve)
                if derivative == 0:
                    values -= current[axis]
                if np.max(np.abs(values)) > limits[limit_key] + 1e-09:
                    raise ValueError(f"Joint {axis + 1} trajectory exceeds {limit_key}")
    return result


def _check_available(self, space: str, *, trajectory: bool = False):
    if space not in ("joint", "ee"):
        raise ValueError("Control space must be joint or ee")
    section = self._config[space]
    if space == "ee":
        enabled = section["enabled"]
        controller = section["streaming"]["controller"]
    else:
        section = section["trajectory" if trajectory else "streaming"]
        enabled, controller = (section["enabled"], section["controller"])
    if not enabled:
        mode = "trajectory" if trajectory else "streaming"
        raise RuntimeError(f"{space} {mode} is disabled in the motion profile")
    state._get_state(self, required=("joints", "ee") if space == "ee" else ("joints",))
    state._raise_robot_fault(self)
    return controller


def _require_no_motion(self):
    if self._trajectory_running:
        raise RuntimeError("A trajectory is running; cancel or wait before changing control")


def _validate_stream_step(self, command, *, space):
    limits = self._config[space]["streaming"]["limits"]
    if limits["enabled"]:
        if space == "joint":
            if np.max(np.abs(command - self._stream_origin)) > limits["max_excursion_rad"] + 1e-9:
                raise ValueError("Joint target exceeds the stream excursion limit")
            if (
                self._last_command is not None
                and np.max(np.abs(command - self._last_command)) > limits["max_step_rad"] + 1e-9
            ):
                raise ValueError("Joint target exceeds the stream step limit")
        else:
            origin_pos, origin_quat = self._stream_origin
            if np.linalg.norm(command[:3] - origin_pos) > limits["max_translation_m"] + 1e-9:
                raise ValueError("EE target exceeds the translation limit")
            if quat_angle_xyzw(command[3:], origin_quat) > limits["max_rotation_rad"] + 1e-9:
                raise ValueError("EE target exceeds the rotation limit")
            if self._last_command is not None:
                if (
                    np.linalg.norm(command[:3] - self._last_command[:3])
                    > limits["max_translation_step_m"] + 1e-9
                ):
                    raise ValueError("EE target exceeds the translation step limit")
                if (
                    quat_angle_xyzw(command[3:], self._last_command[3:])
                    > limits["max_rotation_step_rad"] + 1e-9
                ):
                    raise ValueError("EE target exceeds the rotation step limit")
    self._last_command = command.copy()


def _send_joint_target(self, joints):
    with self._command_lock:
        lifecycle._require_connected(self)
        _require_no_motion(self)
        if self._stream_space != "joint":
            raise RuntimeError("Joint stream is not active")
        state._get_state(self, required=("joints",))
        target = as_vector(joints, 7, "joints")
        _validate_stream_step(self, target, space="joint")
        _publish_joint_target(self, target)
        return dict(space="joint", published_monotonic_s=time.monotonic())


def _send_ee_target(self, position, quaternion_xyzw):
    with self._command_lock:
        lifecycle._require_connected(self)
        _require_no_motion(self)
        if self._stream_space != "ee":
            raise RuntimeError("EE stream is not active")
        state._get_state(self, required=("joints", "ee"))
        state._raise_robot_fault(self)
        pos, quat = (as_vector(position, 3, "position"), normalize_quat_xyzw(quaternion_xyzw))
        _validate_stream_step(self, np.concatenate([pos, quat]), space="ee")
        _publish_ee_target(self, pos, quat)
        return dict(space="ee", published_monotonic_s=time.monotonic())


def _publish_ee_target(self, position: np.ndarray, orientation: np.ndarray):
    msg = PoseStamped()
    msg.header.stamp = self._node.get_clock().now().to_msg()
    msg.header.frame_id = self._config["robot"]["base_frame"]
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, position)
    (
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ) = map(float, orientation)
    self._pose_pub.publish(msg)


def _wait_future(self, future, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while not future.done() and (not self._shutdown):
        if time.monotonic() >= deadline:
            raise TimeoutError("ROS action/service response timed out")
        time.sleep(0.01)
    if not future.done():
        raise RuntimeError("Robot client closed while waiting for ROS")
    return future.result()


def _execute_trajectory(self, points, *, space, on_sample=None):
    with self._command_lock:
        lifecycle._require_connected(self)
        _require_no_motion(self)
        if self._stream_space is not None:
            raise RuntimeError(
                "A stream is active; call stop_stream() before executing a trajectory"
            )
        if space not in ("joint", "ee"):
            raise ValueError("Control space must be joint or ee")
        run = _MotionRun(space)
        self._motion_run, self._cancel_requested = (run, run.cancel)
        self._trajectory_running = True
    begin = time.monotonic()
    try:
        if space == "joint":
            _execute_joint_trajectory(self, points, on_sample)
        else:
            _execute_ee_trajectory(self, points, on_sample)
        cancellation_confirmed = False
        while True:
            with self._command_lock:
                if not run.cancel.is_set() or cancellation_confirmed:
                    self._stream_space = None
                    run.result = dict(
                        status="cancelled" if run.cancel.is_set() else "succeeded",
                        space=space,
                        duration_s=time.monotonic() - begin,
                    )
                    self._trajectory_running = False
                    run.done.set()
                    return run.result
            if space == "ee":
                _hold(self)
            state._wait_until_stopped(self, timeout_s=self._command_timeout_s)
            lifecycle._deactivate_controller(self)
            cancellation_confirmed = True
    except BaseException as exc:
        run.error = exc
        raise
    finally:
        with self._command_lock:
            if self._motion_run is run:
                self._trajectory_running = False
            run.done.set()


def _execute_joint_trajectory(self, points: Sequence[dict], on_sample):
    from action_msgs.msg import GoalStatus
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    controller = motion._check_available(self, "joint", trajectory=True)
    validated = _joint_points(points, state._current_qpos(self), self._config)
    # ConfigureController creates the action server, before activation or motion.
    lifecycle._ensure_loaded(self, controller)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = self._config["robot"]["joint_names"]
    for point in validated:
        waypoint = JointTrajectoryPoint()
        for key in ("positions", "velocities", "accelerations"):
            setattr(waypoint, key, point[key].tolist())
        ns = round(point["time_s"] * 1000000000.0)
        waypoint.time_from_start.sec, waypoint.time_from_start.nanosec = divmod(ns, 10**9)
        goal.trajectory.points.append(waypoint)
    client = ActionClient(
        self._node, FollowJointTrajectory, self._config["joint"]["trajectory"]["action"]
    )
    timeout = self._config["execution"]["service_timeout_s"]
    accepted = None
    result_future = None
    try:
        if not client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError("Joint trajectory action server is unavailable")
        if not lifecycle._activate_arm_controller(self, controller):
            raise RuntimeError("Joint trajectory controller did not activate")
        self._stream_space = None
        self._ee_streaming = False
        accepted = motion._wait_future(self, client.send_goal_async(goal), timeout)
        if accepted is None or not accepted.accepted:
            raise RuntimeError("Joint trajectory goal was rejected")
        result_future = accepted.get_result_async()
        deadline = (
            time.monotonic()
            + validated[-1]["time_s"]
            + self._config["execution"]["execution_timeout_s"]
        )
        while not result_future.done():
            snapshot = state._get_state(self, required=("joints",))
            if on_sample is not None:
                on_sample(snapshot)
            if self._cancel_requested.is_set():
                motion._cancel_joint_goal(self, accepted, result_future)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Joint trajectory execution timed out")
            time.sleep(0.01)
        result = result_future.result()
        if result.status == GoalStatus.STATUS_CANCELED:
            self._cancel_requested.set()
            return
        if (
            result.status != GoalStatus.STATUS_SUCCEEDED
            or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            raise RuntimeError(f"Joint trajectory failed: {result.result.error_string}")
        motion._wait_completion(self, "joint", validated[-1]["positions"], on_sample)
    except BaseException:
        if accepted is not None and accepted.accepted and (result_future is not None):
            try:
                motion._cancel_joint_goal(self, accepted, result_future)
            finally:
                lifecycle._deactivate_controller(self)
        elif self._active_arm_controller == controller:
            lifecycle._deactivate_controller(self)
        raise
    finally:
        client.destroy()


def _cancel_joint_goal(self, goal, result_future):
    if result_future.done():
        return
    timeout = self._config["execution"]["service_timeout_s"]
    reply = motion._wait_future(self, goal.cancel_goal_async(), timeout)
    if not reply.goals_canceling and (not result_future.done()):
        raise RuntimeError("Trajectory cancellation was rejected")
    result = motion._wait_future(self, result_future, timeout)
    from action_msgs.msg import GoalStatus

    if result.status not in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED):
        raise RuntimeError("Trajectory failed while awaiting cancellation")


def _execute_ee_trajectory(self, points: Sequence[dict], on_sample):
    controller = motion._check_available(self, "ee", trajectory=True)
    start_pos, start_quat = state._current_ee_pose(self)
    validated = []
    previous = -1.0
    limits = self._config["limits"]
    for point in points:
        t = float(point["time_s"])
        if not math.isfinite(t) or t < 0 or t <= previous:
            raise ValueError("EE trajectory times must increase strictly")
        pos = as_vector(point["position"], 3, "position")
        quat = normalize_quat_xyzw(point["quaternion_xyzw"])
        validated.append((t, pos, quat))
        previous = t
        if limits["enabled"] and (
            np.linalg.norm(pos - start_pos) > limits["max_ee_translation_m"]
            or quat_angle_xyzw(quat, start_quat) > limits["max_ee_rotation_rad"]
        ):
            raise ValueError("EE trajectory exceeds its excursion limits")
    if len(validated) < 2 or validated[0][0] != 0:
        raise ValueError("EE trajectory needs two poses starting at time zero")
    completion = self._config["ee"]["completion"]
    if (
        np.linalg.norm(validated[0][1] - start_pos) > completion["translation_tolerance_m"]
        or quat_angle_xyzw(validated[0][2], start_quat) > completion["rotation_tolerance_rad"]
    ):
        raise ValueError("First EE waypoint must match measured pose")
    if limits["enabled"]:
        for a, b in zip(validated, validated[1:]):
            duration = b[0] - a[0]
            distance, angle = (np.linalg.norm(b[1] - a[1]), quat_angle_xyzw(a[2], b[2]))
            for value, key in (
                (1.875 * distance / duration, "max_ee_velocity_m_s"),
                (5.773503 * distance / duration**2, "max_ee_acceleration_m_s2"),
                (1.875 * angle / duration, "max_ee_angular_velocity_rad_s"),
                (5.773503 * angle / duration**2, "max_ee_angular_acceleration_rad_s2"),
            ):
                if value > limits[key]:
                    raise ValueError(f"EE trajectory exceeds {key}")
    if not lifecycle._activate_arm_controller(self, controller):
        raise RuntimeError("EE controller did not activate")
    self._stream_space = "ee"
    self._ee_streaming = True
    period = 1.0 / self._config["ee"]["streaming"]["publish_rate_hz"]
    begin = time.monotonic()
    segment = 0
    try:
        while not self._cancel_requested.is_set():
            snapshot = state._get_state(self, required=("joints", "ee"))
            state._raise_robot_fault(self)
            if on_sample is not None:
                on_sample(snapshot)
            elapsed = time.monotonic() - begin
            while segment < len(validated) - 2 and elapsed >= validated[segment + 1][0]:
                segment += 1
            a, b = (validated[segment], validated[segment + 1])
            fraction = np.clip((elapsed - a[0]) / (b[0] - a[0]), 0, 1)
            smooth = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
            motion._publish_ee_target(
                self, a[1] + smooth * (b[1] - a[1]), slerp_quat_xyzw(a[2], b[2], smooth)
            )
            if elapsed >= validated[-1][0]:
                motion._wait_completion(self, "ee", validated[-1], on_sample)
                break
            self._cancel_requested.wait(period)
    except BaseException:
        lifecycle._deactivate_controller(self)
        raise


def _wait_completion(self, space: str, target, on_sample):
    completion = self._config[space]["completion"]
    settled = None
    deadline = time.monotonic() + self._config["execution"]["execution_timeout_s"]
    while not self._cancel_requested.is_set():
        snapshot = state._get_state(
            self, required=("joints", "ee") if space == "ee" else ("joints",)
        )
        state._raise_robot_fault(self)
        if on_sample is not None:
            on_sample(snapshot)
        if space == "joint":
            within = (
                np.max(np.abs(snapshot["joint_positions"] - target))
                <= completion["position_tolerance_rad"]
                and snapshot["joint_velocities"] is not None
                and (
                    np.max(np.abs(snapshot["joint_velocities"]))
                    <= completion["velocity_tolerance_rad_s"]
                )
            )
        else:
            matrix = snapshot["end_effector_pose"]
            within = (
                np.linalg.norm(matrix[:3, 3] - target[1]) <= completion["translation_tolerance_m"]
                and quat_angle_xyzw(matrix_to_quat_xyzw(matrix[:3, :3]), target[2])
                <= completion["rotation_tolerance_rad"]
            )
        settled = (settled if settled is not None else time.monotonic()) if within else None
        if settled is not None and time.monotonic() - settled >= completion["settle_time_s"]:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{space} motion did not settle within tolerance")
        time.sleep(0.01)


def _move_joints(self, target: Sequence[float], *, duration_s: float = 4.0):
    current = state._get_state(self, required=("joints",))["joint_positions"]
    return motion._execute_trajectory(
        self,
        [
            {"time_s": t, "positions": q, "velocities": np.zeros(7), "accelerations": np.zeros(7)}
            for t, q in ((0.0, current), (duration_s, as_vector(target, 7, name="target")))
        ],
        space="joint",
    )


def _move_ee(
    self, position: Sequence[float], quaternion_xyzw: Sequence[float], *, duration_s: float = 4.0
):
    state._get_state(self, required=("joints", "ee"))
    pos, quat = state._current_ee_pose(self)
    return motion._execute_trajectory(
        self,
        [
            {"time_s": 0.0, "position": pos, "quaternion_xyzw": quat},
            {"time_s": duration_s, "position": position, "quaternion_xyzw": quaternion_xyzw},
        ],
        space="ee",
    )


def _hold(self):
    if self._stream_space == "joint":
        snapshot = state._get_state(self, required=("joints",))
        _publish_joint_target(self, snapshot["joint_positions"])
    elif self._stream_space == "ee":
        snapshot = state._get_state(self, required=("joints", "ee"))
        _publish_ee_target(self, snapshot["ee_position"], snapshot["ee_quaternion_xyzw"])


def _publish_joint_target(self, qpos: Sequence[float]):
    joints = as_vector(qpos, 7, name="qpos")
    msg = Float64MultiArray()
    msg.data = [float(value) for value in joints.tolist()]
    self._joint_pub.publish(msg)


@dataclass
class _MotionRun:
    space: str
    owner: int = field(default_factory=threading.get_ident)
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: BaseException | None = None


def _cancel_motion(self, *, timeout_s=None):
    timeout = lifecycle._timeout(self._command_timeout_s if timeout_s is None else timeout_s)
    with self._command_lock:
        run = self._motion_run
        if run is None or run.done.is_set():
            return dict(status="idle", space=None, duration_s=0.0)
        if run.owner == threading.get_ident():
            raise RuntimeError(
                "cancel_motion() must run on another thread while a blocking trajectory is executing"
            )
        run.cancel.set()
    if not run.done.wait(timeout):
        raise TimeoutError("Trajectory cancellation was not confirmed within timeout")
    if run.error is not None:
        raise RuntimeError("Trajectory failed while cancelling") from run.error
    return run.result
