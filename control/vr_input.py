"""Shared VR state and deadman logic for ROS collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VRInput:
    arm_enabled: bool = False
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0
    quat_w: float = 1.0
    x_pressed: bool = False
    y_pressed: bool = False
    save_pressed: bool = False
    gripper_close: bool = False
    gripper_open: bool = False
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    gripper_velocity_axis: float = 0.0
    timestamp: float = 0.0
    pose_monotonic_ns: int = 0
    pose_seq: int = 0


class DualTriggerDeadman:
    """Enable motion only while both fresh triggers were held long enough."""

    def __init__(self, hold_s: float = 0.5, stale_after_s: float = 0.5) -> None:
        self.hold_s = float(hold_s)
        self.stale_after_s = float(stale_after_s)
        self.left_since = 0.0
        self.right_since = 0.0
        self.left_at = 0.0
        self.right_at = 0.0

    def update_left(self, pressed: bool, now: float) -> None:
        self.left_at = now
        if pressed:
            self.left_since = self.left_since or now
        else:
            self.left_since = 0.0

    def update_right(self, pressed: bool, now: float) -> None:
        self.right_at = now
        if pressed:
            self.right_since = self.right_since or now
        else:
            self.right_since = 0.0

    def fresh(self, now: float) -> bool:
        return (
            self.left_at > 0
            and self.right_at > 0
            and now - self.left_at < self.stale_after_s
            and now - self.right_at < self.stale_after_s
        )

    def active(self, now: float) -> bool:
        return (
            self.fresh(now)
            and self.left_since > 0
            and self.right_since > 0
            and now - self.left_since >= self.hold_s
            and now - self.right_since >= self.hold_s
        )
