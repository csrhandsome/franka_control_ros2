"""Robot state and reset operations used by VR collection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from control.util.robot import (
    current_ee_pose,
    current_joint_position,
    move_robot_to_start_pose,
    start_control_streaming,
)


@dataclass
class HoldTarget:
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    joints: np.ndarray

    @classmethod
    def from_arm(cls, arm: Any) -> HoldTarget:
        position, quaternion = current_ee_pose(arm)
        return cls(position, quaternion, current_joint_position(arm).copy())

    def refresh(self, arm: Any) -> None:
        self.position, self.quaternion_xyzw = current_ee_pose(arm)
        self.joints = current_joint_position(arm).copy()

    def apply(self, arm: Any) -> None:
        arm.set_ee_control(self.position, self.quaternion_xyzw, self.joints)


class CollectionRobot:
    def __init__(
        self,
        *,
        arm: Any,
        vr_mapper: Any,
        soft_gripper: Any,
        gripper_type: str,
        control_mode: str,
        start_joint_position: Any,
    ) -> None:
        self.arm = arm
        self.vr_mapper = vr_mapper
        self.soft_gripper = soft_gripper
        self.gripper_type = gripper_type
        self.control_mode = control_mode
        self.start_joint_position = start_joint_position
        self.hold = HoldTarget.from_arm(arm)
        self.gripper_state = (
            float(soft_gripper.gripper_open_ratio.reshape(-1)[0])
            if soft_gripper is not None
            else 1.0
        )
        self.last_gripper_cmd = self.gripper_state
        self.gripper_busy = False
        self.last_gripper_switch_time = 0.0
        self.gripper_switch_cooldown_s = 0.12

    def request_franka_gripper(self, command: float) -> bool:
        if (
            command == self.last_gripper_cmd
            or self.gripper_busy
            or time.time() - self.last_gripper_switch_time
            < self.gripper_switch_cooldown_s
        ):
            return False
        self.last_gripper_switch_time = time.time()
        self.gripper_state = 1.0 if command > 0.5 else 0.0
        self.last_gripper_cmd = command
        self.gripper_busy = True
        threading.Thread(
            target=self._run_franka_gripper, args=(command,), daemon=True
        ).start()
        return True

    def _run_franka_gripper(self, command: float) -> None:
        try:
            self.hold.apply(self.arm)
            if command > 0.5:
                self.arm.gripper_open()
            else:
                self.arm.gripper_close()
            self.hold.refresh(self.arm)
            self.hold.apply(self.arm)
            self.vr_mapper.reset()
        finally:
            self.gripper_busy = False

    def reset_to_start(self) -> None:
        self.hold.apply(self.arm)
        self.arm.stop_ee_streaming()
        if self.soft_gripper is not None:
            self.soft_gripper.set_gripper_level(0, wait=True)
            self.gripper_state = float(
                self.soft_gripper.gripper_open_ratio.reshape(-1)[0]
            )
        elif self.gripper_type == "franka":
            self.arm.gripper_open()
            self.gripper_state = 1.0
        else:
            self.gripper_state = 1.0
        self.last_gripper_cmd = 1.0
        print("[Control] Moving to start position...")
        move_robot_to_start_pose(self.arm, self.start_joint_position)
        start_control_streaming(self.arm, self.control_mode, settle_s=0.0)
        self.hold.refresh(self.arm)
        self.hold.apply(self.arm)
        self.vr_mapper.reset()
