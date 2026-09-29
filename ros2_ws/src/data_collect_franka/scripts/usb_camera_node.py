#!/usr/bin/env python3
"""Publish one V4L2/UVC camera as sensor_msgs/Image.

Humble overlay helper for the DH5 fingertip cameras. This is not a RealSense
driver and does not replace realsense2_camera.
"""

from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _parse_device(value: str):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


class UsbCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("usb_camera_node")
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("convert_bgr_to_rgb", True)

        device = _parse_device(str(self.get_parameter("device").value))
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = int(self.get_parameter("fps").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._convert_bgr_to_rgb = bool(self.get_parameter("convert_bgr_to_rgb").value)

        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("usb_camera_node requires OpenCV (cv2).") from exc

        self._cv2 = cv2
        capture = cv2.VideoCapture(device)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open USB camera device: {device}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture = capture
        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, "image_raw", _BEST_EFFORT)
        period = 1.0 / float(max(fps, 1))
        self._timer = self.create_timer(period, self._on_timer)
        self.get_logger().info(
            f"Publishing USB camera {device} at {width}x{height}@{fps} -> image_raw"
        )

    def _on_timer(self) -> None:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return
        if self._convert_bgr_to_rgb:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            encoding = "rgb8"
        else:
            encoding = "bgr8"
        msg = self._bridge.cv2_to_imgmsg(frame, encoding=encoding)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self._pub.publish(msg)

    def destroy_node(self) -> bool:
        try:
            if self._capture is not None:
                self._capture.release()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
