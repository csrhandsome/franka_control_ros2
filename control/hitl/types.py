"""Shared observation keys and command / step records for robot RL loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FRONT_IMAGE_KEY = "observation.images.front"
WRIST_IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "actions"


@dataclass
class RobotCommand:
    ee_delta_xyz: np.ndarray
    gripper_close: bool = False
    ee_quat_xyzw: np.ndarray | None = None
    intervening: bool = False
    control_source: str = "policy"
    vr_pose_seq: int = 0
    vr_pose_monotonic_ns: int = 0


@dataclass
class LoopEvents:
    success: bool = False
    failure: bool = False


@dataclass
class StepContext:
    episode_index: int
    step_index: int
    elapsed_s: float
    episode_time_s: float


@dataclass
class StepRecord:
    episode_index: int
    step_index: int
    timestamp_monotonic_ns: int
    ee_pos_start_xyz: np.ndarray
    ee_quat_start_xyzw: np.ndarray
    obs: dict[str, Any]
    command: RobotCommand
    executed_xyz: np.ndarray
    gripper_close: bool
    reward: float
    next_obs: dict[str, Any]
    done: bool
    truncated: bool
    clipped: bool
    success: bool


@dataclass
class EpisodeStats:
    episode_index: int
    episodic_reward: float
    success: bool
    steps: int
    clip_steps: int
    extras: dict[str, Any] = field(default_factory=dict)


def parse_ee_gripper_action(payload: dict[str, Any]) -> tuple[np.ndarray, bool]:
    action = np.asarray(payload[ACTION_KEY], dtype=np.float32).reshape(-1)
    if action.shape[0] < 4:
        padded = np.zeros((4,), dtype=np.float32)
        padded[: action.shape[0]] = action
        action = padded
    else:
        action = action[:4]
    return action[:3].astype(np.float64), bool(action[3] > 0.5)
