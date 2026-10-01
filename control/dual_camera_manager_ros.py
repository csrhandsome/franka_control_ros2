"""Dual RealSense manager over ROS Image topics.

API aligned with DualRealsenseManager. Humble container only.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "DualRealsenseManagerRos requires Humble rclpy. "
        "Run it inside the franka_humble container."
    ) from exc

from control.camera_connector_ros import RealSenseConnectorRos
from control.util.img_util import center_crop_and_resize_rgb_uint8


@dataclass(frozen=True)
class CameraFrameTimestamps:
    camera_timestamp: float
    host_capture_monotonic_ns: int


class DualRealsenseManagerRos:
    """Subscribe to external + wrist color images."""

    def __init__(
        self,
        *,
        external_topic: str = "/external/color/image_raw",
        wrist_topic: str = "/wrist/color/image_raw",
        external_serial: str | None = None,
        wrist_serial: str | None = None,
        crop_scale: float = 0.9,
        out_hw: int | None = None,
        refresh_hz: float = 30.0,
        enabled: bool = True,
        node_name: str = "dual_camera_manager_ros",
    ) -> None:
        self._enabled = bool(enabled)
        self._crop_scale = float(crop_scale)
        self._out_hw = int(out_hw) if out_hw is not None else None
        self._refresh_hz = float(refresh_hz)
        if self._refresh_hz <= 0:
            raise ValueError("refresh_hz must be positive")
        self._image_lock = threading.Lock()
        self._external_img: np.ndarray | None = None
        self._wrist_img: np.ndarray | None = None
        self._external_timestamp: CameraFrameTimestamps | None = None
        self._wrist_timestamp: CameraFrameTimestamps | None = None
        self._owns_context = False
        self._executor = None
        self._spin_thread = None
        self._refresh_thread = None
        self._refresh_stop = threading.Event()
        self.external_camera = None
        self.wrist_camera = None

        if not self._enabled:
            self.external_camera = type("Cam", (), {"serial": external_serial})()
            self.wrist_camera = type("Cam", (), {"serial": wrist_serial})()
            print("[CameraROS] camera_backend=none, ROS image subscriptions disabled")
            return

        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
        self._node = Node(node_name)
        self.external_camera = RealSenseConnectorRos(
            topic=external_topic,
            serial=external_serial,
            node=self._node,
            node_name="external_camera_ros",
        )
        self.wrist_camera = RealSenseConnectorRos(
            topic=wrist_topic,
            serial=wrist_serial,
            node=self._node,
            node_name="wrist_camera_ros",
        )
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, name="camera-ros-spin", daemon=True
        )
        self._spin_thread.start()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, name="camera-ros-refresh", daemon=True
        )
        self._refresh_thread.start()

    def _refresh_loop(self) -> None:
        period = 1.0 / self._refresh_hz
        while not self._refresh_stop.is_set():
            started = time.monotonic()
            self._refresh_cache()
            self._refresh_stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _prepare_image(self, rgb: np.ndarray | None) -> np.ndarray | None:
        if rgb is None:
            return None
        rgb = np.asarray(rgb, dtype=np.uint8)
        if self._out_hw is None:
            return rgb
        return center_crop_and_resize_rgb_uint8(
            rgb, crop_scale=self._crop_scale, out_hw=self._out_hw
        )

    def _refresh_cache(self) -> None:
        if not self._enabled:
            return
        external_raw, external_stamp, external_ns = self.external_camera.snapshot()
        wrist_raw, wrist_stamp, wrist_ns = self.wrist_camera.snapshot()
        external_img = self._prepare_image(external_raw)
        wrist_img = self._prepare_image(wrist_raw)
        with self._image_lock:
            if external_img is not None:
                self._external_img = external_img
                self._external_timestamp = CameraFrameTimestamps(
                    camera_timestamp=external_stamp,
                    host_capture_monotonic_ns=external_ns,
                )
            if wrist_img is not None:
                self._wrist_img = wrist_img
                self._wrist_timestamp = CameraFrameTimestamps(
                    camera_timestamp=wrist_stamp,
                    host_capture_monotonic_ns=wrist_ns,
                )

    def connect(self) -> DualRealsenseManagerRos:
        return self

    def wait_for_frames(self, timeout_s: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
        if not self._enabled:
            size = int(self._out_hw or 224)
            blank = np.zeros((size, size, 3), dtype=np.uint8)
            return blank, blank
        start = time.time()
        while True:
            external_img, wrist_img = self.get_images()
            if external_img is not None and wrist_img is not None:
                return external_img, wrist_img
            if timeout_s > 0 and (time.time() - start) > timeout_s:
                raise RuntimeError(
                    f"Camera timeout: no ROS frames after {timeout_s:.1f}s "
                    f"({self.external_camera._topic}, {self.wrist_camera._topic})"
                )
            time.sleep(0.02)

    def get_images(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._image_lock:
            return self._external_img, self._wrist_img

    def get_frames(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        CameraFrameTimestamps | None,
        CameraFrameTimestamps | None,
    ]:
        with self._image_lock:
            return (
                self._external_img,
                self._wrist_img,
                self._external_timestamp,
                self._wrist_timestamp,
            )

    def close(self) -> None:
        print("[CameraROS] closing")
        self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=2.0)
        if self._executor is not None:
            with contextlib.suppress(Exception):
                self._executor.cancel()
        if self._enabled:
            with contextlib.suppress(Exception):
                self._node.destroy_node()
