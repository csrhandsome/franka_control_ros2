"""Clip policy / teleop EE deltas before they reach the ROS controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EESafety:
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    max_step_xyz: np.ndarray

    @classmethod
    def from_config(cls, bounds: dict, step_sizes: dict) -> EESafety:
        minimum = np.asarray(bounds["min"], dtype=np.float64).reshape(3)
        maximum = np.asarray(bounds["max"], dtype=np.float64).reshape(3)
        steps = np.array(
            [step_sizes["x"], step_sizes["y"], step_sizes["z"]],
            dtype=np.float64,
        )
        return cls(min_xyz=minimum, max_xyz=maximum, max_step_xyz=steps)

    def apply_delta(
        self, current_xyz: np.ndarray, delta_xyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        current = np.asarray(current_xyz, dtype=np.float64).reshape(3)
        delta = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        clipped_step = np.clip(delta, -self.max_step_xyz, self.max_step_xyz)
        target = np.clip(current + clipped_step, self.min_xyz, self.max_xyz)
        executed = target - current
        clipped = not np.allclose(executed, delta, atol=1e-9)
        return target, executed.astype(np.float64), clipped
