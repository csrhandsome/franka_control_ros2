"""ROS Humble DH5 soft-gripper client.

Commands go to ``dh5_gripper_node`` over ROS topics. Fingertip cameras are
Image subscriptions rather than OpenCV VideoCapture.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32, Int32
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DH5GripperRos requires Humble rclpy. Run it inside the franka_humble container."
    ) from exc

from control.dual_camera_manager_ros import DualRealsenseManagerRos


@dataclass
class DHGripperObservation:
    external_img: np.ndarray | None = None
    wrist_img: np.ndarray | None = None
    gripper_open_ratio: np.ndarray = field(
        default_factory=lambda: np.ones((1,), dtype=np.float32)
    )


class DH5GripperRos:
    """High-level API aligned with ``DH5Gripper``, backed by ROS topics."""

    def __init__(
        self,
        *,
        command_ns: str = "/dh5_gripper",
        left_image_topic: str = "/gripper_left/image_raw",
        right_image_topic: str = "/gripper_right/image_raw",
        enable_cameras: bool = True,
        crop_scale: float = 0.9,
        image_hw: int | None = None,
        discrete_level_count: int = 28,
        vr_axis_threshold: float = 0.55,
        vr_step_interval_s: float = 0.08,
        command_timeout_s: float = 2.0,
        use_fake: bool = False,
    ) -> None:
        if int(discrete_level_count) <= 1:
            raise ValueError("discrete_level_count must be > 1")
        if not 0.0 < float(vr_axis_threshold) <= 1.0:
            raise ValueError("vr_axis_threshold must be in (0, 1]")
        if float(vr_step_interval_s) <= 0.0:
            raise ValueError("vr_step_interval_s must be > 0")

        self._command_ns = command_ns.rstrip("/")
        self._use_fake = bool(use_fake)
        self._command_timeout_s = float(command_timeout_s)
        self._position_command = 0
        self._gripper_level_count = int(discrete_level_count)
        self._gripper_level = 0
        self._vr_axis_threshold = float(vr_axis_threshold)
        self._vr_step_interval_s = float(vr_step_interval_s)
        self._last_gripper_step_time = 0.0
        self._last_gripper_step_dir = 0
        self._open_ratio = 1.0
        self._have_state = bool(use_fake)
        self._lock = threading.Lock()
        self._owns_context = False

        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
        self._node = Node("dh5_gripper_ros")
        self._pos_pub = self._node.create_publisher(
            Int32, f"{self._command_ns}/set_position", 10
        )
        self._force_pub = self._node.create_publisher(
            Int32, f"{self._command_ns}/set_force", 10
        )
        self._vel_pub = self._node.create_publisher(
            Int32, f"{self._command_ns}/set_velocity", 10
        )
        self._node.create_subscription(
            JointState, f"{self._command_ns}/state", self._on_state, 10
        )
        self._node.create_subscription(
            Float32, f"{self._command_ns}/open_ratio", self._on_open_ratio, 10
        )
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="dh5-ros-spin", daemon=True
        )
        self._spin_thread.start()

        self.dual_camera_manager = DualRealsenseManagerRos(
            external_topic=right_image_topic,
            wrist_topic=left_image_topic,
            crop_scale=crop_scale,
            out_hw=image_hw,
            enabled=enable_cameras,
            node_name="dh5_gripper_cameras_ros",
        )
        self.dual_camera_manager.connect()
        self._observation = DHGripperObservation(
            gripper_open_ratio=self.gripper_open_ratio
        )

        if not self._use_fake:
            deadline = time.time() + self._command_timeout_s
            while time.time() < deadline and not self._have_state:
                time.sleep(0.05)
            if not self._have_state:
                print(
                    "[DH5ROS] No /dh5_gripper/state yet. "
                    "Is dh5_gripper.launch.py running?"
                )
        print("[DH5ROS] Client ready")

    def _on_state(self, msg: JointState) -> None:
        with self._lock:
            if msg.position:
                self._position_command = int(round(float(msg.position[0])))
                self._gripper_level = self._position_to_level(self._position_command)
            self._have_state = True

    def _on_open_ratio(self, msg: Float32) -> None:
        with self._lock:
            self._open_ratio = float(np.clip(msg.data, 0.0, 1.0))

    def set_force(self, value: int) -> None:
        if not 20 <= int(value) <= 100:
            raise RuntimeError("Out of range")
        msg = Int32()
        msg.data = int(value)
        self._force_pub.publish(msg)

    def set_velocity(self, value: int) -> None:
        if not 0 <= int(value) <= 1000:
            raise RuntimeError("Out of range")
        msg = Int32()
        msg.data = int(value)
        self._vel_pub.publish(msg)

    def set_position(self, value: int, *, wait: bool = True) -> None:
        if not 0 <= int(value) <= 1000:
            raise RuntimeError("Out of range")
        msg = Int32()
        msg.data = int(value)
        self._pos_pub.publish(msg)
        with self._lock:
            self._position_command = int(value)
            self._gripper_level = self._position_to_level(int(value))
            self._open_ratio = float(1.0 - (int(value) / 1000.0))
        if wait:
            time.sleep(min(self._command_timeout_s, 0.2))

    @property
    def gripper_level_count(self) -> int:
        return self._gripper_level_count

    @property
    def gripper_level(self) -> int:
        return self._gripper_level

    @property
    def position_command(self) -> int:
        return int(self._position_command)

    @property
    def max_gripper_level(self) -> int:
        return self._gripper_level_count - 1

    def _level_to_position(self, level: int) -> int:
        clipped = int(np.clip(level, 0, self.max_gripper_level))
        if self.max_gripper_level <= 0:
            return 0
        return int(round((float(clipped) / float(self.max_gripper_level)) * 1000.0))

    def _position_to_level(self, position: int) -> int:
        if self.max_gripper_level <= 0:
            return 0
        clipped = float(np.clip(position, 0, 1000))
        return int(
            np.clip(
                np.rint((clipped / 1000.0) * float(self.max_gripper_level)),
                0,
                self.max_gripper_level,
            )
        )

    def set_gripper_level(self, level: int, *, wait: bool = True) -> bool:
        next_level = int(np.clip(level, 0, self.max_gripper_level))
        if next_level == self._gripper_level:
            return False
        self.set_position(self._level_to_position(next_level), wait=wait)
        return True

    def update_from_vr(
        self,
        vr_input,
        *,
        axis_threshold: float | None = None,
        step_interval_s: float | None = None,
        now: float | None = None,
        wait: bool = False,
    ) -> bool:
        axis = float(getattr(vr_input, "gripper_velocity_axis", 0.0))
        threshold = (
            self._vr_axis_threshold if axis_threshold is None else float(axis_threshold)
        )
        step_interval = (
            self._vr_step_interval_s
            if step_interval_s is None
            else float(step_interval_s)
        )
        now = time.monotonic() if now is None else float(now)

        if axis >= threshold:
            step_dir = 1
        elif axis <= -threshold:
            step_dir = -1
        else:
            self._last_gripper_step_dir = 0
            return False

        should_step = (
            step_dir != self._last_gripper_step_dir
            or self._last_gripper_step_time == 0.0
            or (now - self._last_gripper_step_time) >= step_interval
        )
        if not should_step:
            return False

        self._last_gripper_step_time = now
        self._last_gripper_step_dir = step_dir
        return self.set_gripper_level(self._gripper_level + step_dir, wait=wait)

    @property
    def gripper_open_ratio(self) -> np.ndarray:
        with self._lock:
            normalized = float(np.clip(self._open_ratio, 0.0, 1.0))
        return np.asarray([normalized], dtype=np.float32)

    @property
    def observation(self) -> DHGripperObservation:
        right_img, left_img, _, _ = self.dual_camera_manager.get_frames()
        self._observation = DHGripperObservation(
            external_img=right_img,
            wrist_img=left_img,
            gripper_open_ratio=self.gripper_open_ratio,
        )
        return self._observation

    def wait_for_frames(self, timeout_s: float = 10.0):
        return self.dual_camera_manager.wait_for_frames(timeout_s=timeout_s)

    def close(self) -> None:
        try:
            self.dual_camera_manager.close()
        except Exception:
            pass
        try:
            self._executor.cancel()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass

    def __enter__(self) -> DH5GripperRos:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
