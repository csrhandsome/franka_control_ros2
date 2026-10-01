"""Small robot operations shared by collection workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from control.util.pose import matrix_to_quat_xyzw


def current_joint_position(arm: Any) -> np.ndarray:
    return np.asarray(arm.state["joint_positions"], dtype=np.float64)


def current_ee_pose(arm: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(arm.ee_pose_matrix, dtype=np.float64)
    return matrix[:3, 3].copy(), matrix_to_quat_xyzw(matrix[:3, :3])


def start_control_streaming(arm: Any, control_mode: str, settle_s: float = 0.0) -> None:
    if control_mode == "ee":
        arm.start_ee_streaming(settle_s=settle_s)
    elif control_mode == "joint":
        arm.start_joint_streaming(settle_s=settle_s)
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")
    position, quaternion = current_ee_pose(arm)
    arm.set_ee_control(position, quaternion, current_joint_position(arm))


def format_joint_position(qpos: np.ndarray) -> str:
    return np.array2string(
        np.asarray(qpos, dtype=np.float64),
        precision=5,
        separator=", ",
        suppress_small=False,
    )


def move_robot_to_start_pose(
    arm: Any, custom_start: Sequence[float] | None = None
) -> None:
    move_name = "move_to_start"
    if custom_start is None:
        arm.move_to_start()
    else:
        joints = np.asarray(custom_start, dtype=np.float64).flatten()
        if joints.shape != (7,):
            raise ValueError(f"target_qpos must be 7D, got shape {joints.shape}")
        print(f"[Control] Moving to custom joint pose: {format_joint_position(joints)}")
        arm.move_to_joint_position(joints)
        move_name = "custom start joint pose"
    if not arm.wait_until_stopped():
        max_vel = float(np.max(np.abs(np.asarray(arm.state["joint_velocities"]))))
        print(
            f"[Warning] Robot did not fully stop after {move_name}: "
            f"max_vel={max_vel:.4f} rad/s"
        )
