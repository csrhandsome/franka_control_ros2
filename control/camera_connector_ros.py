"""ROS Humble RealSense image subscriber.

Use inside the Humble container with realsense2_camera.
"""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RealSenseConnectorRos requires Humble rclpy/cv_bridge. "
        "Run it inside the franka_humble container."
    ) from exc


_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class RealSenseConnectorRos:
    """Subscribe to a ROS Image topic and expose the latest RGB frame."""

    def __init__(
        self,
        *,
        topic: str,
        serial: str | None = None,
        node: Node | None = None,
        node_name: str = "camera_connector_ros",
    ) -> None:
        self._topic = topic
        self._serial = serial
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._rgb: np.ndarray | None = None
        self._stamp_sec: float = 0.0
        self._host_capture_monotonic_ns: int = 0
        self._owns_context = False
        self._owns_node = False

        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
        if node is None:
            self._node = Node(node_name)
            self._owns_node = True
        else:
            self._node = node
        self._sub = self._node.create_subscription(
            Image, self._topic, self._on_image, _BEST_EFFORT
        )
        print(f"[CameraROS] Subscribing to {self._topic}")

    @property
    def serial(self) -> str | None:
        return self._serial

    @property
    def img(self) -> np.ndarray | None:
        with self._lock:
            return None if self._rgb is None else np.array(self._rgb, copy=True)

    def snapshot(self) -> tuple[np.ndarray | None, float, int]:
        """Return an image and its timestamps from the same ROS callback."""
        with self._lock:
            image = None if self._rgb is None else np.array(self._rgb, copy=True)
            return image, float(self._stamp_sec), int(self._host_capture_monotonic_ns)

    @property
    def timestamp(self) -> float:
        with self._lock:
            return float(self._stamp_sec)

    @property
    def host_capture_monotonic_ns(self) -> int:
        with self._lock:
            return int(self._host_capture_monotonic_ns)

    def _on_image(self, msg: Image) -> None:
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            print(f"[CameraROS] cv_bridge failed on {self._topic}: {exc}")
            return
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self._lock:
            self._rgb = np.asarray(rgb, dtype=np.uint8)
            self._stamp_sec = stamp
            self._host_capture_monotonic_ns = time.monotonic_ns()

    def connect(self) -> RealSenseConnectorRos:
        return self

    def close(self) -> None:
        with self._lock:
            self._rgb = None
        if self._owns_node:
            with contextlib.suppress(Exception):
                self._node.destroy_node()
