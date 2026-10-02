"""Small robot operations shared by collection workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import time

import numpy as np

from control.util.pose import matrix_to_quat_xyzw


def current_joint_position(arm: Any) -> np.ndarray:
    return np.asarray(arm.get_state(required=("joints",))["joint_positions"], dtype=np.float64)


def current_ee_pose(arm: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(arm.get_state(required=("ee",))["end_effector_pose"], dtype=np.float64)
    return matrix[:3, 3].copy(), matrix_to_quat_xyzw(matrix[:3, :3])


def start_control_streaming(arm: Any, control_mode: str, settle_s: float = 0.0) -> None:
    arm.start_stream(control_mode)
    if settle_s > 0:
        time.sleep(settle_s)


def format_joint_position(qpos: np.ndarray) -> str:
    return np.array2string(
        np.asarray(qpos, dtype=np.float64),
        precision=5,
        separator=", ",
        suppress_small=False,
    )


def move_robot_to_start_pose(
    arm: Any, custom_start: Sequence[float], *, motion_config: dict | None = None
) -> None:
    joints = np.asarray(custom_start, dtype=np.float64)
    print(f"[Control] Moving to initial joint pose: {format_joint_position(joints)}")
    duration_s = (
        4.0
        if motion_config is None
        else joint_move_duration(current_joint_position(arm), joints, motion_config)
    )
    arm.move_joints(joints, duration_s=duration_s)
    arm.wait_until_stopped()


def joint_move_duration(
    start: Sequence[float],
    target: Sequence[float],
    motion_config: dict,
    *,
    minimum_s: float = 4.0,
) -> float:
    """Choose a rest-to-rest quintic duration from the configured joint bounds."""
    from control.util.pose import as_vector

    distance = float(np.max(np.abs(as_vector(target, 7, "target") - as_vector(start, 7, "start"))))
    limits = motion_config["limits"]
    if not limits["enabled"]:
        return minimum_s
    return max(
        minimum_s,
        1.01 * 1.875 * distance / limits["max_joint_velocity_rad_s"],
        1.01 * float(np.sqrt(5.773503 * distance / limits["max_joint_acceleration_rad_s2"])),
    )
