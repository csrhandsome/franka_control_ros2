"""VR state, ROS input, and end-effector mapping for collection."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from control.util.pose import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Joy
except ImportError as exc:  # The state and mapper can also be used without ROS.
    rclpy = None
    _ROS_IMPORT_ERROR = exc
else:
    _ROS_IMPORT_ERROR = None

if rclpy is not None:
    _BEST_EFFORT = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


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


class VRInputRos:
    """Read VR pose and require both fresh trigger buttons for takeover."""

    def __init__(
        self,
        *,
        pose_topic: str = "/xr/controller_right/pose",
        left_joy_topic: str = "/xr/controller_left/joy",
        right_joy_topic: str = "/xr/controller_right/joy",
        long_press_s: float = 0.5,
        stale_after_s: float = 0.5,
        assume_arm_enabled: bool = False,
    ) -> None:
        if rclpy is None:
            raise ImportError(
                "VRInputRos requires Humble rclpy. Run it inside the franka_humble container."
            ) from _ROS_IMPORT_ERROR
        self._pose_topic = pose_topic
        self._left_joy_topic = left_joy_topic
        self._right_joy_topic = right_joy_topic
        self._stale_after_s = float(stale_after_s)
        self._deadman = DualTriggerDeadman(long_press_s, stale_after_s)
        self._assume_arm_enabled = bool(assume_arm_enabled)
        self._lock = threading.Lock()
        self._latest = VRInput()
        self._pose_at = 0.0

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("vr_input_ros")
        self._node.create_subscription(
            PoseStamped, pose_topic, self._on_pose, _BEST_EFFORT
        )
        self._node.create_subscription(
            Joy, left_joy_topic, self._on_left_joy, _BEST_EFFORT
        )
        self._node.create_subscription(
            Joy, right_joy_topic, self._on_right_joy, _BEST_EFFORT
        )
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="vr-ros-spin", daemon=True
        )

    def start(self) -> None:
        print(
            f"[VRROS] Subscribing to {self._pose_topic}, "
            f"{self._left_joy_topic}, {self._right_joy_topic}"
        )
        self._spin_thread.start()

    def stop(self) -> None:
        self._executor.shutdown(timeout_sec=1.0)
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._node.destroy_node()

    @staticmethod
    def _button(msg: Joy, index: int) -> bool:
        return index < len(msg.buttons) and msg.buttons[index] > 0

    def _on_pose(self, msg: PoseStamped) -> None:
        now = time.monotonic()
        with self._lock:
            self._latest.pos_x = float(msg.pose.position.x)
            self._latest.pos_y = float(msg.pose.position.y)
            self._latest.pos_z = float(msg.pose.position.z)
            self._latest.quat_x = float(msg.pose.orientation.x)
            self._latest.quat_y = float(msg.pose.orientation.y)
            self._latest.quat_z = float(msg.pose.orientation.z)
            self._latest.quat_w = float(msg.pose.orientation.w)
            self._latest.timestamp = time.time()
            self._latest.pose_monotonic_ns = time.monotonic_ns()
            self._latest.pose_seq += 1
            self._pose_at = now

    def _on_left_joy(self, msg: Joy) -> None:
        now = time.monotonic()
        with self._lock:
            self._deadman.update_left(self._button(msg, 0), now)
            self._latest.x_pressed = self._button(msg, 4)
            self._latest.y_pressed = self._button(msg, 5)
            self._latest.save_pressed = self._latest.y_pressed

    def _on_right_joy(self, msg: Joy) -> None:
        now = time.monotonic()
        with self._lock:
            self._deadman.update_right(self._button(msg, 0), now)
            self._latest.gripper_close = self._button(msg, 4)
            self._latest.gripper_open = self._button(msg, 5)
            # teleop_xr appends each button value after the gamepad axes.
            gamepad_axes = msg.axes[: -len(msg.buttons)] if msg.buttons else msg.axes
            if len(gamepad_axes) >= 2:
                self._latest.right_stick_x = float(gamepad_axes[-2])
                self._latest.right_stick_y = float(gamepad_axes[-1])
                self._latest.gripper_velocity_axis = -self._latest.right_stick_y

    @property
    def latest(self) -> VRInput:
        now = time.monotonic()
        with self._lock:
            pose_fresh = self._pose_at > 0 and now - self._pose_at < self._stale_after_s
            joys_fresh = self._deadman.fresh(now)
            held = self._deadman.active(now)
            arm_enabled = pose_fresh and (
                self._assume_arm_enabled or (joys_fresh and held)
            )
            return VRInput(
                arm_enabled=arm_enabled,
                pos_x=self._latest.pos_x,
                pos_y=self._latest.pos_y,
                pos_z=self._latest.pos_z,
                quat_x=self._latest.quat_x,
                quat_y=self._latest.quat_y,
                quat_z=self._latest.quat_z,
                quat_w=self._latest.quat_w,
                x_pressed=self._latest.x_pressed if joys_fresh else False,
                y_pressed=self._latest.y_pressed if joys_fresh else False,
                save_pressed=self._latest.save_pressed if joys_fresh else False,
                gripper_close=self._latest.gripper_close if joys_fresh else False,
                gripper_open=self._latest.gripper_open if joys_fresh else False,
                right_stick_x=self._latest.right_stick_x if joys_fresh else 0.0,
                right_stick_y=self._latest.right_stick_y if joys_fresh else 0.0,
                gripper_velocity_axis=self._latest.gripper_velocity_axis
                if joys_fresh
                else 0.0,
                timestamp=self._latest.timestamp,
                pose_monotonic_ns=self._latest.pose_monotonic_ns,
                pose_seq=self._latest.pose_seq,
            )


class VREEPoseMapper:
    """Continuously integrate VR controller motion into an EE target pose.

    The input pose uses ROS FLU axes, aligned with the robot base frame.
    This class keeps an internal EE command and updates it from adjacent VR
    samples. That means the operator can keep moving continuously without
    releasing or re-engaging the deadman switch.

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

                cmd_rot = Rotation.from_quat(quat_wxyz_to_xyzw(self._cmd_ee_quat))
                target_rot = Rotation.from_rotvec(rot_cmd) * cmd_rot
                if self._rotation_limit > 0.0:
                    anchor_rot = Rotation.from_quat(
                        quat_wxyz_to_xyzw(self._anchor_ee_quat)
                    )
                    rel_rotvec = (target_rot * anchor_rot.inv()).as_rotvec()
                    rel_rotvec = np.clip(
                        rel_rotvec,
                        -self._rotation_limit,
                        self._rotation_limit,
                    )
                    target_rot = Rotation.from_rotvec(rel_rotvec) * anchor_rot

                target_quat = quat_xyzw_to_wxyz(target_rot.as_quat())

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
                anchor_rot = Rotation.from_quat(quat_wxyz_to_xyzw(self._anchor_ee_quat))
                cmd_rot = Rotation.from_quat(quat_wxyz_to_xyzw(target_quat))
                rot_offset = (cmd_rot * anchor_rot.inv()).as_rotvec()
            print(
                f"[VR→EE] step=({ee_step[0]:+.4f},{ee_step[1]:+.4f},{ee_step[2]:+.4f}) "
                f"cmd=({pos_offset[0]:+.4f},{pos_offset[1]:+.4f},{pos_offset[2]:+.4f}) "
                f"drot=({rot_cmd[0]:+.3f},{rot_cmd[1]:+.3f},{rot_cmd[2]:+.3f}) "
                f"rot=({rot_offset[0]:+.3f},{rot_offset[1]:+.3f},{rot_offset[2]:+.3f})"
            )

        return target_pos.copy(), target_quat.copy()
