"""Public ROS arm contracts. Distances are metres and angles are radians."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

ControlSpace = Literal["joint", "ee"]


class RobotState(TypedDict):
    """Copied measured snapshot; missing sources are None, stale ones have valid=False.

    Source timestamps are ROS seconds. age_s uses monotonic receipt time.
    The independently received sources are not time synchronized.
    """

    joint_positions: np.ndarray | None
    joint_velocities: np.ndarray | None
    ee_position: np.ndarray | None
    ee_quaternion_xyzw: np.ndarray | None
    end_effector_pose: np.ndarray | None
    gripper_width_m: float | None
    base_frame: str
    valid: dict[str, bool]
    age_s: dict[str, float | None]
    stamp_s: dict[str, float | None]


class Capability(TypedDict):
    enabled: bool
    ready: bool
    reason: str | None


class RobotCapabilities(TypedDict):
    joint_stream: Capability
    joint_trajectory: Capability
    ee_stream: Capability
    ee_trajectory: Capability
    gripper: Capability


class RobotStatus(TypedDict):
    connected: bool
    robot_type: str
    use_fake_hardware: bool
    stream_space: ControlSpace | None
    trajectory_running: bool
    active_controller: str | None
    gripper_busy: bool
    controllers: dict[str, str]
    state: RobotState


class CommandReceipt(TypedDict):
    """Local publication receipt, without an acknowledgement from the robot."""

    space: ControlSpace
    published_monotonic_s: float


class MotionResult(TypedDict):
    status: Literal["succeeded", "cancelled", "idle"]
    space: ControlSpace | None
    duration_s: float
