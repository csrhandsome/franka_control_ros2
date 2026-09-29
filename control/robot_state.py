"""Robot arm state representations shared by collection and inference.

The learned 6D pose is [x, y, z, roll, pitch, yaw] in the robot base frame.
Angles use radians and the ZYX (yaw-pitch-roll) Euler convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _as_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return vector.copy()


def _with_gripper(arm_values: np.ndarray, gripper_open: float) -> np.ndarray:
    if not np.isfinite(gripper_open) or not 0.0 <= gripper_open <= 1.0:
        raise ValueError("Logical gripper state must be in 0..1")
    return np.concatenate([arm_values, [float(gripper_open)]]).astype(np.float32)


@dataclass(frozen=True)
class EEPose:
    """Base-frame xyz metres and ZYX roll/pitch/yaw radians."""

    xyz_m: np.ndarray
    rpy_rad: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz_m", _as_vector(self.xyz_m, 3, "EE xyz"))
        object.__setattr__(self, "rpy_rad", _as_vector(self.rpy_rad, 3, "EE rpy"))

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.xyz_m, self.rpy_rad]).astype(np.float32)

    def action(self, gripper_open: float) -> np.ndarray:
        return _with_gripper(self.vector, gripper_open)

    def quaternion_xyzw(self) -> np.ndarray:
        return pose6_to_quat_xyzw(self.vector)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "EEPose":
        pose = matrix_to_pose6(matrix)
        return cls(pose[:3], pose[3:])

    @classmethod
    def from_position_quat(cls, position: np.ndarray, quat_xyzw: np.ndarray) -> "EEPose":
        pose = position_quat_to_pose6(position, quat_xyzw)
        return cls(pose[:3], pose[3:])

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "EEPose":
        pose = _as_vector(vector, 6, "EE pose")
        return cls(pose[:3], pose[3:])


@dataclass(frozen=True)
class JointPose:
    """Seven measured or commanded Franka joint angles in radians."""

    angles_rad: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles_rad", _as_vector(self.angles_rad, 7, "Joint angles"))

    @property
    def vector(self) -> np.ndarray:
        return self.angles_rad.astype(np.float32)

    def action(self, gripper_open: float) -> np.ndarray:
        return _with_gripper(self.vector, gripper_open)


def matrix_to_pose6(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("EE transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    else:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.array([*matrix[:3, 3], roll, pitch, yaw], dtype=np.float64)


def position_quat_to_pose6(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("EE position and xyzw quaternion must be 3D and 4D")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(position).all() or norm < 1e-12:
        raise ValueError("EE pose must be finite with a nonzero quaternion")
    x, y, z, w = quat / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = position
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return matrix_to_pose6(matrix)


def pose6_to_quat_xyzw(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("EE pose must be finite [x,y,z,roll,pitch,yaw]")
    roll, pitch, yaw = pose[3:] / 2.0
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    quat = np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )
    return quat / np.linalg.norm(quat)


def quat_angle_xyzw(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("Quaternions must be xyzw 4D vectors")
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(2.0 * math.acos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0)))
