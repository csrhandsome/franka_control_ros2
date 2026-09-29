"""Map ROS VR poses to continuous Franka end-effector targets.

The input pose uses ROS FLU axes, aligned with the robot base frame.
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from control.vr_input import VRInput


class VREEPoseMapper:
    """Continuously integrate VR controller motion into an EE target pose.

    This class keeps an internal EE command and updates it from adjacent VR
    samples. That means the operator can keep
    moving continuously without releasing / re-engaging the deadman switch.

    Speed and total excursion are decoupled:
      - ``max_translation_step`` / ``max_rotation_step`` control per-cycle EE
        increments, i.e. how fast the command moves.
      - ``translation_limit`` / ``rotation_limit`` optionally bound the total
        commanded offset around the engagement pose. Set either limit to ``0``
        to disable that workspace clamp.
    """

    def __init__(
        self,
        *,
        translation_scale: float,
        rotation_scale: float,
        max_translation_step: float,
        max_rotation_step: float,
        translation_limit: float,
        rotation_limit: float,
        sensitivity: float,
        enable_rotation: bool = True,
    ) -> None:
        self._translation_scale = float(translation_scale)
        self._rotation_scale = float(rotation_scale)
        self._max_translation_step = float(max_translation_step)
        self._max_rotation_step = float(max_rotation_step)
        self._translation_limit = float(translation_limit)
        self._rotation_limit = float(rotation_limit)
        self._sensitivity = float(sensitivity)
        self._enable_rotation = bool(enable_rotation)
        self._translation_alpha = 0.35
        self._last_debug_at = 0.0
        self._state_lock = threading.Lock()
        self.reset()

    @staticmethod
    def _quat_wxyz_to_xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
        return np.array(
            [
                quaternion_wxyz[1],
                quaternion_wxyz[2],
                quaternion_wxyz[3],
                quaternion_wxyz[0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_xyzw_to_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
        return np.array(
            [
                quaternion_xyzw[3],
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
            ],
            dtype=np.float64,
        )

    def reset(self) -> None:
        with self._state_lock:
            self._prev_vr_pos: np.ndarray | None = None
            self._prev_vr_rot: Rotation | None = None
            self._filtered_vr_pos: np.ndarray | None = None
            self._cmd_ee_pos: np.ndarray | None = None
            self._cmd_ee_quat: np.ndarray | None = None
            self._anchor_ee_pos: np.ndarray | None = None
            self._anchor_ee_quat: np.ndarray | None = None

    def map(
        self,
        vr: VRInput,
        ee_position: np.ndarray,
        ee_quaternion_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_pos = np.asarray(ee_position, dtype=np.float64)
        current_quat = np.asarray(ee_quaternion_wxyz, dtype=np.float64)

        if not vr.arm_enabled:
            self.reset()
            return current_pos.copy(), current_quat.copy()

        vr_pos = np.array([vr.pos_x, vr.pos_y, vr.pos_z], dtype=np.float64)
        vr_rot = Rotation.from_quat([vr.quat_x, vr.quat_y, vr.quat_z, vr.quat_w])

        with self._state_lock:
            if (
                self._prev_vr_pos is None
                or self._filtered_vr_pos is None
                or self._cmd_ee_pos is None
                or self._cmd_ee_quat is None
                or self._anchor_ee_pos is None
                or self._anchor_ee_quat is None
            ):
                self._prev_vr_pos = vr_pos.copy()
                self._prev_vr_rot = vr_rot
                self._filtered_vr_pos = vr_pos.copy()
                self._cmd_ee_pos = current_pos.copy()
                self._cmd_ee_quat = current_quat.copy()
                self._anchor_ee_pos = current_pos.copy()
                self._anchor_ee_quat = current_quat.copy()
                return current_pos.copy(), current_quat.copy()

            self._filtered_vr_pos = (
                1.0 - self._translation_alpha
            ) * self._filtered_vr_pos + self._translation_alpha * vr_pos
            pos_delta = self._filtered_vr_pos - self._prev_vr_pos
            scaled_translation = np.clip(
                pos_delta / self._translation_scale,
                -1.0,
                1.0,
            )
            ee_step = (
                scaled_translation * self._max_translation_step * self._sensitivity
            )
            target_pos = self._cmd_ee_pos + ee_step
            if self._translation_limit > 0.0:
                target_offset = np.clip(
                    target_pos - self._anchor_ee_pos,
                    -self._translation_limit,
                    self._translation_limit,
                )
                target_pos = self._anchor_ee_pos + target_offset

            target_quat = self._cmd_ee_quat.copy()
            rot_cmd = np.zeros(3, dtype=np.float64)
            if self._enable_rotation and self._prev_vr_rot is not None:
                rot_delta = (vr_rot * self._prev_vr_rot.inv()).as_rotvec()
                scaled_rotation = np.clip(
                    rot_delta / self._rotation_scale,
                    -1.0,
                    1.0,
                )
                rot_cmd = scaled_rotation * self._max_rotation_step * self._sensitivity

                cmd_rot = Rotation.from_quat(self._quat_wxyz_to_xyzw(self._cmd_ee_quat))
                target_rot = Rotation.from_rotvec(rot_cmd) * cmd_rot
                if self._rotation_limit > 0.0:
                    anchor_rot = Rotation.from_quat(
                        self._quat_wxyz_to_xyzw(self._anchor_ee_quat)
                    )
                    rel_rotvec = (target_rot * anchor_rot.inv()).as_rotvec()
                    rel_rotvec = np.clip(
                        rel_rotvec,
                        -self._rotation_limit,
                        self._rotation_limit,
                    )
                    target_rot = Rotation.from_rotvec(rel_rotvec) * anchor_rot

                target_quat = self._quat_xyzw_to_wxyz(target_rot.as_quat())

            self._cmd_ee_pos = target_pos.copy()
            self._cmd_ee_quat = target_quat.copy()
            self._prev_vr_pos = self._filtered_vr_pos.copy()
            self._prev_vr_rot = vr_rot

        now = time.monotonic()
        if now - self._last_debug_at >= 1.0:
            self._last_debug_at = now
            pos_offset = target_pos - self._anchor_ee_pos
            rot_offset = np.zeros(3, dtype=np.float64)
            if self._enable_rotation:
                anchor_rot = Rotation.from_quat(
                    self._quat_wxyz_to_xyzw(self._anchor_ee_quat)
                )
                cmd_rot = Rotation.from_quat(self._quat_wxyz_to_xyzw(target_quat))
                rot_offset = (cmd_rot * anchor_rot.inv()).as_rotvec()
            print(
                f"[VR→EE] step=({ee_step[0]:+.4f},{ee_step[1]:+.4f},{ee_step[2]:+.4f}) "
                f"cmd=({pos_offset[0]:+.4f},{pos_offset[1]:+.4f},{pos_offset[2]:+.4f}) "
                f"drot=({rot_cmd[0]:+.3f},{rot_cmd[1]:+.3f},{rot_cmd[2]:+.3f}) "
                f"rot=({rot_offset[0]:+.3f},{rot_offset[1]:+.3f},{rot_offset[2]:+.3f})"
            )

        return target_pos.copy(), target_quat.copy()
