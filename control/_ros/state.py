"""Private measured-state callbacks and snapshots."""

from __future__ import annotations
import math
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control.util.pose import (
    as_vector,
    normalize_quat_xyzw,
    matrix_to_quat_xyzw,
    position_quat_to_matrix,
)
from . import lifecycle, state, gripper


def _on_joint_states(self, msg: JointState):
    name_to_pos = dict(zip(msg.name, msg.position))
    name_to_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
    names = self._config["robot"]["joint_names"]
    if not all((name in name_to_pos for name in names)):
        return
    positions = []
    velocities = []
    for index, name in enumerate(names, start=1):
        positions.append(float(name_to_pos.get(name, self._joint_positions[index - 1])))
        velocities.append(float(name_to_vel.get(name, 0.0)))
    finger_names = [f"{self._robot_type}_finger_joint1", f"{self._robot_type}_finger_joint2"]
    fingers = [name_to_pos[name] for name in finger_names if name in name_to_pos]
    if not np.all(np.isfinite(positions + velocities + fingers)):
        return
    with self._state_lock:
        self._joint_positions = np.asarray(positions, dtype=np.float64)
        self._joint_velocities = np.asarray(velocities, dtype=np.float64)
        self._have_joint_velocity = all(name in name_to_vel for name in names)
        if fingers:
            self._gripper_position = float(sum(fingers))
            self._have_gripper_state = True
            state._mark_state(self, "gripper", msg)
        self._have_joint_state = True
        state._mark_state(self, "joints", msg)


def _on_current_pose(self, msg: PoseStamped):
    try:
        position = as_vector(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], 3, "EE position"
        )
        orientation = normalize_quat_xyzw(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
    except ValueError:
        return
    if msg.header.frame_id and msg.header.frame_id != self._config["robot"]["base_frame"]:
        return
    with self._state_lock:
        self._last_ee_pos = position
        self._last_ee_quat = orientation
        self._have_ee_pose = True
        state._mark_state(self, "ee", msg)


def _on_gripper_states(self, msg: JointState):
    if not msg.position:
        return
    width = float(msg.position[0])
    if len(msg.position) > 1:
        width += float(msg.position[1])
    if not math.isfinite(width):
        return
    with self._state_lock:
        self._gripper_position = width
        self._have_gripper_state = True
        state._mark_state(self, "gripper", msg)


def _on_panda_state(self, msg):
    modes = {
        msg.ROBOT_MODE_OTHER: "Other",
        msg.ROBOT_MODE_GUIDING: "Guiding",
        msg.ROBOT_MODE_REFLEX: "Reflex",
        msg.ROBOT_MODE_USER_STOPPED: "User stopped",
        msg.ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY: "Automatic error recovery",
    }
    mode = modes.get(msg.robot_mode)
    errors = [name for name in msg.current_errors.get_fields_and_field_types()
              if getattr(msg.current_errors, name)]
    if mode == "Reflex" and not errors:
        errors = [name for name in msg.last_motion_errors.get_fields_and_field_types()
                  if getattr(msg.last_motion_errors, name)]
    fault = f"Panda motion unavailable: {mode or 'current errors'}" if mode or errors else None
    if fault and errors:
        fault += " (" + ", ".join(errors) + ")"
    with self._state_lock:
        self._robot_fault = fault
    matrix = np.asarray(msg.o_t_ee, dtype=np.float64).reshape(4, 4, order="F")
    if not np.all(np.isfinite(matrix)):
        return
    with self._state_lock:
        self._last_ee_pos = matrix[:3, 3].copy()
        self._last_ee_quat = matrix_to_quat_xyzw(matrix[:3, :3])
        self._have_ee_pose = True
        state._mark_state(self, "ee", msg)


def _raise_robot_fault(self):
    with self._state_lock:
        fault = self._robot_fault
    if fault:
        raise RuntimeError(fault)


def _mark_state(self, kind: str, msg):
    self._state_received[kind] = time.monotonic()
    self._state_stamps[kind] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-09


def _list_controllers(self):
    if not self._connected or self._shutdown:
        return []
    from controller_manager_msgs.srv import ListControllers

    result = lifecycle._call_service(self, self._list_client, ListControllers.Request())
    if result is None:
        raise RuntimeError("Controller manager is unavailable")
    return [(item.name, item.state) for item in result.controller]


def _get_capabilities(self):
    connected = self._connected and (not self._shutdown)
    try:
        controllers = dict(_list_controllers(self))
        discovery_error = None
    except RuntimeError as exc:
        controllers, discovery_error = ({}, str(exc))
    snapshot = _snapshot(self)
    with self._state_lock:
        robot_fault = self._robot_fault
    joint, ee = (self._config["joint"], self._config["ee"])
    specifications = {
        "joint_stream": (
            joint["streaming"]["enabled"],
            joint["streaming"]["controller"],
            ("joints",),
        ),
        "joint_trajectory": (
            joint["trajectory"]["enabled"],
            joint["trajectory"]["controller"],
            ("joints",),
        ),
        "ee_stream": (ee["enabled"], ee["streaming"]["controller"], ("joints", "ee")),
        "ee_trajectory": (ee["enabled"], ee["streaming"]["controller"], ("joints", "ee")),
    }
    result = {}
    for name, (enabled, controller, required) in specifications.items():
        reason = None
        if not enabled:
            reason = "Disabled in motion profile"
        elif not connected:
            reason = "Robot client is not connected"
        elif discovery_error:
            reason = discovery_error
        elif robot_fault:
            reason = robot_fault
        elif controllers.get(controller) not in ("active", "inactive"):
            reason = f"Controller {controller} is not configured"
        elif any((not snapshot["valid"][key] for key in required)):
            reason = "Required measured state is missing or stale"
        elif name == "joint_trajectory":
            from control_msgs.action import FollowJointTrajectory
            from rclpy.action import ActionClient

            client = ActionClient(self._node, FollowJointTrajectory, joint["trajectory"]["action"])
            try:
                if not client.server_is_ready():
                    reason = "Joint trajectory action server is unavailable"
            finally:
                client.destroy()
        result[name] = dict(enabled=enabled, ready=reason is None, reason=reason)
    enabled = self._config["gripper"]["enabled"]
    reason = "Disabled in motion profile" if not enabled else None
    if enabled:
        if not connected:
            reason = "Robot client is not connected"
        elif self._gripper_move is None or self._gripper_grasp is None:
            reason = "Gripper clients are unavailable"
        elif not self._gripper_move.server_is_ready() or not self._gripper_grasp.server_is_ready():
            reason = "Gripper action servers are unavailable"
        elif not self._gripper_stop.service_is_ready():
            reason = "Gripper stop service is unavailable"
    result["gripper"] = dict(enabled=enabled, ready=reason is None, reason=reason)
    return result


def _get_state(self, *, required=("joints", "ee")):
    snapshot = _snapshot(self)
    for name in required:
        if name not in snapshot["valid"]:
            raise ValueError(f"Unknown state field: {name}")
        if not snapshot["valid"][name]:
            raise RuntimeError(f"Measured {name} state is missing or stale")
    return snapshot


def _wait_ready(self, *, required=("joints", "ee"), timeout_s=None):
    lifecycle._require_connected(self)
    timeout = self._config["execution"]["startup_timeout_s"] if timeout_s is None else timeout_s
    deadline = time.monotonic() + lifecycle._timeout(timeout)
    while not self._shutdown:
        try:
            return _get_state(self, required=required)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for fresh {tuple(required)} state") from None
            time.sleep(0.01)
    raise RuntimeError("Robot client is closed")


def _wait_until_stopped(self, *, timeout_s=2.0):
    deadline = time.monotonic() + lifecycle._timeout(timeout_s)
    tolerance = self._config["joint"]["completion"]["velocity_tolerance_rad_s"]
    settle_s = self._config["joint"]["completion"]["settle_time_s"]
    settled = None
    while True:
        snapshot = _get_state(self, required=("joints",))
        velocities = snapshot["joint_velocities"]
        if velocities is None:
            raise RuntimeError("Measured joint velocities are unavailable; cannot confirm stop")
        now = time.monotonic()
        if np.max(np.abs(velocities)) <= tolerance:
            settled = now if settled is None else settled
            if now - settled >= settle_s:
                return
        else:
            settled = None
        if now >= deadline:
            raise TimeoutError("Robot did not stop within timeout")
        time.sleep(0.01)


def _current_qpos(self):
    with self._state_lock:
        return self._joint_positions.copy()


def _current_ee_pose(self):
    with self._state_lock:
        return (self._last_ee_pos.copy(), self._last_ee_quat.copy())


def _snapshot(self):
    with self._state_lock:
        now = time.monotonic()
        ages = {
            name: None if received is None else now - received
            for name, received in self._state_received.items()
        }
        valid = {
            name: not self._shutdown
            and age is not None
            and (age <= self._config["execution"]["state_timeout_s"])
            for name, age in ages.items()
        }
        joints = self._joint_positions.copy() if ages["joints"] is not None else None
        velocities = (
            self._joint_velocities.copy()
            if ages["joints"] is not None and self._have_joint_velocity
            else None
        )
        position = self._last_ee_pos.copy() if ages["ee"] is not None else None
        quaternion = self._last_ee_quat.copy() if ages["ee"] is not None else None
        pose = position_quat_to_matrix(position, quaternion) if position is not None else None
        width = float(self._gripper_position) if ages["gripper"] is not None else None
        return dict(
            joint_positions=joints,
            joint_velocities=velocities,
            ee_position=position,
            ee_quaternion_xyzw=quaternion,
            end_effector_pose=pose,
            gripper_width_m=width,
            base_frame=self._config["robot"]["base_frame"],
            valid=valid,
            age_s=ages,
            stamp_s=self._state_stamps.copy(),
        )


def _get_status(self):
    with self._command_lock:
        return dict(
            connected=self._connected and (not self._shutdown),
            robot_type=self._robot_type,
            use_fake_hardware=self._use_fake_hardware,
            stream_space=self._stream_space,
            trajectory_running=self._trajectory_running,
            active_controller=self._active_arm_controller,
            gripper_busy=gripper._busy(self) if self._connected else False,
            controllers=dict(_list_controllers(self)),
            state=_snapshot(self),
        )
