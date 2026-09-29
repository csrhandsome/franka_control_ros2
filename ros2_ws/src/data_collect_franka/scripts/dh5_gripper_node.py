#!/usr/bin/env python3
"""DH5 soft-gripper serial driver as a ROS 2 node.

Owns /dev/ttyUSB* (Modbus RTU). Collection scripts should command this node
instead of opening the serial port themselves.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Int32
from std_srvs.srv import Trigger


class ControlRoot:
    def __init__(self, com: str = "/dev/ttyUSB0", timeout: float = 0.2) -> None:
        import serial

        self.sc = serial.Serial(
            port=com,
            baudrate=115200,
            timeout=timeout,
            write_timeout=timeout,
        )
        import crcmod

        self.crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)

    def calCrc(self, array):
        bytes_ = b""
        for i in range(len(array)):
            bytes_ = bytes_ + array[i].to_bytes(1, byteorder="big", signed=True)
        crc = self.crc16(bytes_).to_bytes(2, byteorder="big", signed=False)
        crc_h = int.from_bytes(crc[0:1], byteorder="big", signed=False)
        crc_q = int.from_bytes(crc[1:2], byteorder="big", signed=False)
        return crc_q, crc_h

    def clear_input_buffer(self) -> None:
        try:
            self.sc.reset_input_buffer()
        except AttributeError:
            self.sc.read_all()

    def readSerial(self, expected_length=None, timeout_s: float | None = None):
        if expected_length is None:
            return self.sc.read_all()
        if timeout_s is None:
            timeout_s = self.sc.timeout
        timeout_s = 0.0 if timeout_s is None else max(float(timeout_s), 0.0)
        deadline = time.monotonic() + timeout_s
        received = bytearray()
        while len(received) < expected_length:
            remaining = expected_length - len(received)
            chunk = self.sc.read(remaining)
            if chunk:
                received.extend(chunk)
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.002)
        return bytes(received)

    def sendCmd(
        self,
        ModbusHighAddress,
        ModbusLowAddress,
        Value=0x01,
        isSet=True,
        isReadSerial=True,
    ):
        set_address = 0x06 if isSet else 0x03
        value = Value if Value >= 0 else Value - 1
        bytes_ = value.to_bytes(2, byteorder="big", signed=True)
        value_q = int.from_bytes(bytes_[0:1], byteorder="big", signed=True)
        value_h = int.from_bytes(bytes_[1:2], byteorder="big", signed=True)
        array = [
            0x01,
            set_address,
            ModbusHighAddress,
            ModbusLowAddress,
            value_q,
            value_h,
        ]
        crc_q, crc_h = self.calCrc(array)
        command = array + [crc_q, crc_h]
        for i in range(len(command)):
            command[i] = command[i] if command[i] >= 0 else command[i] + 256
        self.sc.write(bytes(command))
        if not isReadSerial:
            time.sleep(0.005)
            return None
        expected_length = 8 if isSet else 7
        back = self.readSerial(expected_length=expected_length)
        if len(back) < expected_length:
            raise RuntimeError(
                f"No or short response from gripper on {self.sc.port}: "
                f"expected {expected_length} bytes, got {len(back)} bytes ({back.hex(' ')})"
            )
        if back[1] & 0x80:
            raise RuntimeError(
                f"Gripper returned Modbus exception frame: {back.hex(' ')}"
            )
        if isSet:
            value = int.from_bytes(back[4:6], byteorder="big", signed=True)
        else:
            value = int.from_bytes(back[3:5], byteorder="big", signed=True)
        if value < 0:
            value = value + 1
        self.sc.flush()
        return value


def _in_range(value: int, min_: int, max_: int) -> None:
    if not min_ <= value <= max_:
        raise RuntimeError(f"Out of range: {value} not in [{min_}, {max_}]")


class Dh5GripperNode(Node):
    def __init__(self) -> None:
        super().__init__("dh5_gripper_node")
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("use_fake", False)
        self.declare_parameter("force", 50)
        self.declare_parameter("velocity", 100)
        self.declare_parameter("state_rate", 20.0)

        self._port = str(self.get_parameter("port").value)
        use_fake_param = self.get_parameter("use_fake").value
        if isinstance(use_fake_param, str):
            self._use_fake = use_fake_param.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            self._use_fake = bool(use_fake_param)
        self._position = 0
        self._force = int(self.get_parameter("force").value)
        self._velocity = int(self.get_parameter("velocity").value)
        self._hand: ControlRoot | None = None

        if self._use_fake:
            self.get_logger().warn(
                "[DH5] use_fake=true; serial port will not be opened"
            )
        else:
            self._hand = ControlRoot(com=self._port)
            self._initialize()
            self._set_register(0x01, 0x01, self._force)
            self._set_register(0x01, 0x04, self._velocity)
            self.get_logger().info(
                f"[DH5] Opened {self._port} force={self._force} velocity={self._velocity}"
            )

        self._state_pub = self.create_publisher(JointState, "state", 10)
        self._ratio_pub = self.create_publisher(Float32, "open_ratio", 10)
        self.create_subscription(Int32, "set_position", self._on_set_position, 10)
        self.create_subscription(Int32, "set_force", self._on_set_force, 10)
        self.create_subscription(Int32, "set_velocity", self._on_set_velocity, 10)
        self.create_service(Trigger, "initialize", self._on_initialize)
        rate = float(self.get_parameter("state_rate").value)
        self.create_timer(1.0 / max(rate, 1.0), self._publish_state)

    def _set_register(self, high: int, low: int, value: int, *, read: bool = True):
        if self._hand is None:
            return value
        return self._hand.sendCmd(
            ModbusHighAddress=high,
            ModbusLowAddress=low,
            Value=value,
            isReadSerial=read,
        )

    def _get_register(self, high: int, low: int) -> int:
        if self._hand is None:
            return 1
        return int(
            self._hand.sendCmd(
                ModbusHighAddress=high,
                ModbusLowAddress=low,
                isSet=False,
            )
        )

    def _initialize(self) -> None:
        if self._hand is None:
            return
        self._set_register(0x01, 0x00, 1)
        deadline = time.time() + 5.0
        status = self._get_register(0x02, 0x00)
        while status == 0:
            if time.time() >= deadline:
                raise TimeoutError("Gripper initialization timed out with status 0")
            self._set_register(0x01, 0x00, 1)
            time.sleep(0.1)
            status = self._get_register(0x02, 0x00)
            self.get_logger().info(f"[DH5] initialization_status={status}")
        while status == 2:
            if time.time() >= deadline:
                raise TimeoutError("Gripper initialization timed out with status 2")
            time.sleep(0.1)
            status = self._get_register(0x02, 0x00)
            self.get_logger().info(f"[DH5] initialization_status={status}")

    def _on_initialize(self, _request, response):
        try:
            self._initialize()
            response.success = True
            response.message = "ok"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _on_set_position(self, msg: Int32) -> None:
        value = int(msg.data)
        _in_range(value, 0, 1000)
        if self._hand is not None:
            self._hand.clear_input_buffer()
            self._set_register(0x01, 0x03, value, read=False)
        self._position = value

    def _on_set_force(self, msg: Int32) -> None:
        value = int(msg.data)
        _in_range(value, 20, 100)
        self._set_register(0x01, 0x01, value)
        self._force = value

    def _on_set_velocity(self, msg: Int32) -> None:
        value = int(msg.data)
        _in_range(value, 0, 1000)
        self._set_register(0x01, 0x04, value)
        self._velocity = value

    def _publish_state(self) -> None:
        state = JointState()
        state.header.stamp = self.get_clock().now().to_msg()
        state.name = ["dh5_gripper"]
        state.position = [float(self._position)]
        state.effort = [float(self._force)]
        state.velocity = [float(self._velocity)]
        self._state_pub.publish(state)
        ratio = Float32()
        ratio.data = float(1.0 - (self._position / 1000.0))
        self._ratio_pub.publish(ratio)

    def destroy_node(self) -> bool:
        try:
            if self._hand is not None and self._hand.sc.is_open:
                self._hand.sc.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = Dh5GripperNode()
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
