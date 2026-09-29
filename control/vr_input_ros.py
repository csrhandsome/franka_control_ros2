"""ROS VR input from one teleop_xr ROS publisher.

The right controller pose and both controllers' Joy messages come from
``python -m teleop_xr.ros2 --mode teleop``. No second VR server is started.
"""

from __future__ import annotations

import threading
import time

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Joy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "VRInputRos requires Humble rclpy. Run it inside the franka_humble container."
    ) from exc

from control.vr_input import DualTriggerDeadman, VRInput

_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
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
