#!/usr/bin/env python3
"""ROS 2 Humble Franka adapter. Runs in the Humble Docker image only.

Uses ROS 2 topics, actions, and services for robot control.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from std_srvs.srv import Trigger
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RoboticArmControlerRos requires Humble rclpy. Run it inside the "
        "franka_humble container."
    ) from exc


DEFAULT_START_JOINTS = np.array(
    [0.0, -math.pi / 4, 0.0, -3.0 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4],
    dtype=np.float64,
)

_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
_RELIABLE = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _as_joint_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).flatten()
    if vector.shape != (7,):
        raise ValueError(f"{name} must be 7D, got shape {vector.shape}")
    return vector


def _quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    r21 = float(rotation[2, 0])
    pitch = math.asin(float(np.clip(-r21, -1.0, 1.0)))
    if abs(math.cos(pitch)) < 1e-6:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    else:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _pose_to_matrix(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix(quat_xyzw)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


@dataclass
class _GripperCommand:
    kind: Literal["move", "grasp"]
    params: dict[str, float]
    done: threading.Event = field(default_factory=threading.Event)
    result: bool | None = None
    error: BaseException | None = None


class RoboticArmControlerRos:
    """ROS 2 backend with the same high-level methods as RoboticArmControler."""

    def __init__(
        self,
        *,
        use_fake_hardware: bool = False,
        robot_type: str = "fr3",
        command_timeout_s: float = 2.0,
        node: Node | None = None,
        start_joint_position: Sequence[float] | None = None,
    ) -> None:
        self.use_fake_hardware = bool(use_fake_hardware)
        self.robot_ip = os.environ.get("FRANKA_ROBOT_IP")
        self._robot_type = str(robot_type or "fr3")
        self._command_timeout_s = float(command_timeout_s)
        self.auto_set_default_behavior = False
        self.auto_move_to_start = False
        self.joint_speed_factor = 0.21
        self._start_joint_position = (
            _as_joint_vector(start_joint_position, name="start_joint_position")
            if start_joint_position is not None
            else DEFAULT_START_JOINTS.copy()
        )
        self._shutdown = False
        self._owns_context = False
        self._owns_node = False
        self._ee_streaming = False
        self._active_arm_controller: str | None = None
        self._last_ee_pos = np.array([0.3, 0.0, 0.5], dtype=np.float64)
        self._last_ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._joint_positions = DEFAULT_START_JOINTS.copy()
        self._joint_velocities = np.zeros(7, dtype=np.float64)
        self._gripper_position = 0.07
        self._have_joint_state = False
        self._have_ee_pose = False
        self._state_lock = threading.Lock()

        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True

        if node is None:
            self._node = Node("franka_ros2_control")
            self._owns_node = True
        else:
            self._node = node

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin, name="franka-ros2-spin", daemon=True
        )
        self._spin_thread.start()

        self._pose_pub = self._node.create_publisher(
            PoseStamped, "/cartesian_pose_target_controller/target_pose", _BEST_EFFORT
        )
        self._joint_pub = self._node.create_publisher(
            Float64MultiArray, "/joint_position_target_controller/target_joints", 10
        )
        self._node.create_subscription(
            JointState, "/joint_states", self._on_joint_states, _RELIABLE
        )
        self._node.create_subscription(
            PoseStamped,
            "/franka_robot_state_broadcaster/current_pose",
            self._on_current_pose,
            _BEST_EFFORT,
        )
        self._node.create_subscription(
            JointState,
            "/franka_gripper/joint_states",
            self._on_gripper_states,
            _RELIABLE,
        )

        self._switch_client = self._node.create_client(
            self._switch_srv_type(), "/controller_manager/switch_controller"
        )
        self._list_client = self._node.create_client(
            self._list_srv_type(), "/controller_manager/list_controllers"
        )
        self._load_client = self._node.create_client(
            self._load_srv_type(), "/controller_manager/load_controller"
        )

        self._gripper_move = None
        self._gripper_grasp = None
        self._gripper_stop = None
        self._init_gripper_clients()

        self._gripper_lock = threading.Lock()
        self._gripper_command: _GripperCommand | None = None
        self._gripper_active_command: _GripperCommand | None = None
        self._gripper_request = threading.Event()
        self._gripper_stop_event = threading.Event()
        self._gripper_thread = threading.Thread(
            target=self._gripper_worker, name="franka-ros2-gripper", daemon=True
        )
        self._gripper_thread.start()

        if not self._wait_for_joint_states(timeout_s=max(self._command_timeout_s, 5.0)):
            message = "[FrankaROS2] No /joint_states yet."
            if self.use_fake_hardware:
                self._node.get_logger().warn(message + " Is fake bringup running?")
            else:
                self._node.get_logger().error(message)
                if not self.use_fake_hardware:
                    print(message + " Pass --use-fake-hardware if this is expected.")

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception:
            if not self._shutdown:
                raise

    @staticmethod
    def _switch_srv_type():
        from controller_manager_msgs.srv import SwitchController

        return SwitchController

    @staticmethod
    def _list_srv_type():
        from controller_manager_msgs.srv import ListControllers

        return ListControllers

    @staticmethod
    def _load_srv_type():
        from controller_manager_msgs.srv import LoadController

        return LoadController

    def _init_gripper_clients(self) -> None:
        try:
            from franka_msgs.action import Grasp, Move
        except ImportError:
            self._node.get_logger().warn(
                "[FrankaROS2] franka_msgs actions not available."
            )
            return
        self._gripper_move = ActionClient(self._node, Move, "/franka_gripper/move")
        self._gripper_grasp = ActionClient(self._node, Grasp, "/franka_gripper/grasp")
        self._gripper_stop = self._node.create_client(Trigger, "/franka_gripper/stop")

    def _on_joint_states(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        positions = []
        velocities = []
        for index in range(1, 8):
            name = f"{self._robot_type}_joint{index}"
            positions.append(
                float(name_to_pos.get(name, self._joint_positions[index - 1]))
            )
            velocities.append(float(name_to_vel.get(name, 0.0)))
        finger_names = [
            f"{self._robot_type}_finger_joint1",
            f"{self._robot_type}_finger_joint2",
        ]
        fingers = [name_to_pos[name] for name in finger_names if name in name_to_pos]
        with self._state_lock:
            self._joint_positions = np.asarray(positions, dtype=np.float64)
            self._joint_velocities = np.asarray(velocities, dtype=np.float64)
            if fingers:
                self._gripper_position = float(sum(fingers))
            self._have_joint_state = True

    def _on_current_pose(self, msg: PoseStamped) -> None:
        with self._state_lock:
            self._last_ee_pos = np.array(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                dtype=np.float64,
            )
            self._last_ee_quat = np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=np.float64,
            )
            self._have_ee_pose = True

    def _on_gripper_states(self, msg: JointState) -> None:
        if not msg.position:
            return
        width = float(msg.position[0])
        if len(msg.position) > 1:
            width += float(msg.position[1])
        with self._state_lock:
            self._gripper_position = width

    def _wait_for_joint_states(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline and not self._shutdown:
            with self._state_lock:
                if self._have_joint_state:
                    return True
            time.sleep(0.05)
        return self._have_joint_state

    def _call_service(self, client, request, timeout_s: float | None = None):
        timeout_s = self._command_timeout_s if timeout_s is None else timeout_s
        if not client.wait_for_service(timeout_sec=timeout_s):
            return None
        future = client.call_async(request)
        deadline = time.time() + timeout_s
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        if not future.done():
            return None
        return future.result()

    def list_controllers(self) -> list[tuple[str, str]]:
        from controller_manager_msgs.srv import ListControllers

        result = self._call_service(self._list_client, ListControllers.Request())
        if result is None:
            return []
        return [(item.name, item.state) for item in result.controller]

    def _ensure_loaded(self, name: str) -> None:
        from controller_manager_msgs.srv import LoadController

        known = {item[0] for item in self.list_controllers()}
        if name in known:
            return
        request = LoadController.Request()
        request.name = name
        result = self._call_service(self._load_client, request)
        if result is None or not getattr(result, "ok", True):
            raise RuntimeError(f"[FrankaROS2] Failed to load controller {name}")

    def _switch_controllers(
        self,
        *,
        activate: Sequence[str] = (),
        deactivate: Sequence[str] = (),
    ) -> bool:
        from controller_manager_msgs.srv import SwitchController

        activate_list = [name for name in activate if name]
        deactivate_list = [name for name in deactivate if name]
        for name in activate_list:
            self._ensure_loaded(name)

        request = SwitchController.Request()
        if hasattr(request, "activate_controllers"):
            request.activate_controllers = list(activate_list)
            request.deactivate_controllers = list(deactivate_list)
        else:
            request.start_controllers = list(activate_list)
            request.stop_controllers = list(deactivate_list)
        if hasattr(request, "strictness"):
            best_effort = getattr(SwitchController.Request, "BEST_EFFORT", 1)
            request.strictness = best_effort
        if hasattr(request, "activate_asap"):
            request.activate_asap = True
        if hasattr(request, "timeout"):
            request.timeout.sec = int(self._command_timeout_s)
            request.timeout.nanosec = int((self._command_timeout_s % 1.0) * 1e9)

        result = self._call_service(self._switch_client, request, timeout_s=5.0)
        return bool(result is not None and getattr(result, "ok", True))

    def _activate_arm_controller(self, name: str) -> bool:
        deactivate = []
        if self._active_arm_controller and self._active_arm_controller != name:
            deactivate.append(self._active_arm_controller)
        ok = self._switch_controllers(activate=[name], deactivate=deactivate)
        if ok:
            self._active_arm_controller = name
            print(f"[FrankaROS2] Active controller: {name}")
        else:
            print(f"[FrankaROS2] Failed to activate {name}")
        return ok

    def start_ee_streaming(self, settle_s: float = 0.0) -> None:
        if self._activate_arm_controller("cartesian_pose_target_controller"):
            self._ee_streaming = True
            pos, quat = self._current_ee_pose()
            self.set_ee_control(pos, quat, self._current_qpos())
            if settle_s > 0.0:
                time.sleep(settle_s)
            return
        print(
            "[FrankaROS2] Cartesian controller unavailable; "
            "falling back to joint_position_target_controller."
        )
        if not self._activate_arm_controller("joint_position_target_controller"):
            raise RuntimeError(
                "[FrankaROS2] No arm command controller could be activated"
            )
        self._ee_streaming = False
        self._publish_joint_target(self._current_qpos())
        if settle_s > 0.0:
            time.sleep(settle_s)

    def start_joint_streaming(self, settle_s: float = 0.0) -> None:
        if not self._activate_arm_controller("joint_position_target_controller"):
            raise RuntimeError(
                "[FrankaROS2] joint_position_target_controller did not activate"
            )
        self._ee_streaming = False
        self._publish_joint_target(self._current_qpos())
        if settle_s > 0.0:
            time.sleep(settle_s)

    def set_joint_control(self, qpos: Sequence[float]) -> None:
        """Publish one joint target after joint streaming has been activated."""
        if self._active_arm_controller != "joint_position_target_controller":
            raise RuntimeError("[FrankaROS2] Joint streaming is not active")
        self._publish_joint_target(qpos)

    @property
    def joint_state_ready(self) -> bool:
        with self._state_lock:
            return self._have_joint_state

    @property
    def ee_pose_ready(self) -> bool:
        with self._state_lock:
            return self._have_ee_pose

    @property
    def ee_streaming_active(self) -> bool:
        return self._ee_streaming and self._active_arm_controller == "cartesian_pose_target_controller"

    def set_ee_control(
        self,
        pos: Sequence[float],
        quat_xyzw: Sequence[float],
        qpos: Sequence[float] | None = None,
    ) -> None:
        position = np.asarray(pos, dtype=np.float64).reshape(3)
        quaternion = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        # Keep the measured pose from current_pose separate from issued targets.
        with self._state_lock:
            if not self._have_ee_pose:
                self._last_ee_pos = position
                self._last_ee_quat = quaternion
        if self._active_arm_controller == "joint_position_target_controller":
            if qpos is not None:
                self._publish_joint_target(qpos)
            return
        msg = PoseStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = f"{self._robot_type}_link0"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])
        self._pose_pub.publish(msg)

    def stop_ee_streaming(self) -> None:
        pos, quat = self._current_ee_pose()
        self.set_ee_control(pos, quat, self._current_qpos())
        self._ee_streaming = False

    def _publish_joint_target(self, qpos: Sequence[float]) -> None:
        joints = _as_joint_vector(qpos, name="qpos")
        msg = Float64MultiArray()
        msg.data = [float(value) for value in joints.tolist()]
        self._joint_pub.publish(msg)

    def move_to_joint_position(self, qpos: Sequence[float], *_args, **_kwargs) -> None:
        joints = _as_joint_vector(qpos, name="qpos")
        if not self._activate_arm_controller("joint_position_target_controller"):
            raise RuntimeError(
                "[FrankaROS2] joint_position_target_controller did not activate"
            )
        self._ee_streaming = False
        deadline = time.time() + max(self._command_timeout_s, 8.0)
        while time.time() < deadline:
            self._publish_joint_target(joints)
            if float(np.max(np.abs(self._current_qpos() - joints))) < 5e-3:
                break
            time.sleep(0.05)

    def move_to_start(self) -> None:
        print("[FrankaROS2] Moving to start joint position")
        self.move_to_joint_position(self._start_joint_position)

    def wait_until_stopped(
        self,
        *,
        timeout_s: float = 2.0,
        velocity_tol: float = 5e-3,
        poll_s: float = 0.01,
    ) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if float(np.max(np.abs(self._current_dq()))) <= velocity_tol:
                return True
            time.sleep(poll_s)
        return False

    def _current_qpos(self) -> np.ndarray:
        with self._state_lock:
            return self._joint_positions.copy()

    def _current_dq(self) -> np.ndarray:
        with self._state_lock:
            return self._joint_velocities.copy()

    def _current_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        with self._state_lock:
            return self._last_ee_pos.copy(), self._last_ee_quat.copy()

    def _submit_gripper_command(
        self,
        kind: Literal["move", "grasp"],
        params: dict[str, float],
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        command = _GripperCommand(kind=kind, params=dict(params))
        with self._gripper_lock:
            self._gripper_command = command
        self._gripper_request.set()
        if not wait:
            return True
        if not command.done.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for gripper command")
        if command.error is not None:
            if self.use_fake_hardware:
                print(
                    f"[FrankaROS2] Gripper {kind} skipped on fake hardware: {command.error}"
                )
                return True
            raise RuntimeError("Gripper command failed") from command.error
        return bool(command.result)

    def _send_gripper_goal(self, client, goal) -> bool:
        if client is None:
            raise RuntimeError("gripper action client is not available")
        if not client.wait_for_server(timeout_sec=self._command_timeout_s):
            raise RuntimeError("gripper action server is not available")
        future = client.send_goal_async(goal)
        deadline = time.time() + self._command_timeout_s
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise TimeoutError("gripper goal send timed out")
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("gripper goal was rejected")
        result_future = handle.get_result_async()
        deadline = time.time() + max(self._command_timeout_s, 5.0)
        while not result_future.done() and time.time() < deadline:
            time.sleep(0.02)
        return bool(result_future.done())

    def _gripper_worker(self) -> None:
        while not self._gripper_stop_event.is_set():
            self._gripper_request.wait(timeout=0.2)
            self._gripper_request.clear()
            with self._gripper_lock:
                command = self._gripper_command
                self._gripper_command = None
                self._gripper_active_command = command
            if command is None:
                continue
            try:
                if command.kind == "move":
                    from franka_msgs.action import Move

                    goal = Move.Goal()
                    goal.width = float(command.params["width"])
                    goal.speed = float(command.params["speed"])
                    command.result = self._send_gripper_goal(self._gripper_move, goal)
                else:
                    from franka_msgs.action import Grasp

                    goal = Grasp.Goal()
                    goal.width = float(command.params["width"])
                    goal.speed = float(command.params["speed"])
                    goal.force = float(command.params["force"])
                    goal.epsilon.inner = float(command.params["epsilon_inner"])
                    goal.epsilon.outer = float(command.params["epsilon_outer"])
                    command.result = self._send_gripper_goal(self._gripper_grasp, goal)
            except BaseException as exc:
                command.error = exc
                command.result = False
            with self._gripper_lock:
                self._gripper_active_command = None
            command.done.set()

    @property
    def gripper_busy(self) -> bool:
        with self._gripper_lock:
            command = self._gripper_command
            active = self._gripper_active_command
        return (command is not None and not command.done.is_set()) or (
            active is not None and not active.done.is_set()
        )

    def wait_gripper(self, timeout: float | None = None) -> None:
        with self._gripper_lock:
            command = self._gripper_command or self._gripper_active_command
        if command is None:
            return
        if not command.done.wait(timeout=timeout):
            raise TimeoutError("Timed out waiting for gripper")

    def gripper_open(
        self,
        width: float = 0.05,
        speed: float = 0.2,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        return self._submit_gripper_command(
            "move", {"width": width, "speed": speed}, wait=wait, timeout=timeout
        )

    def gripper_close(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 60.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        return self._submit_gripper_command(
            "grasp",
            {
                "width": width,
                "speed": speed,
                "force": force,
                "epsilon_inner": epsilon_inner,
                "epsilon_outer": epsilon_outer,
            },
            wait=wait,
            timeout=timeout,
        )

    @property
    def state(self) -> dict:
        with self._state_lock:
            qpos = self._joint_positions.copy()
            dq = self._joint_velocities.copy()
            pose = _pose_to_matrix(self._last_ee_pos, self._last_ee_quat)
            gripper = float(self._gripper_position)
        return {
            "joint_positions": qpos,
            "joint_velocities": dq,
            "end_effector_pose": pose,
            "cartesian_position": pose[:3, 3],
            "gripper_position": gripper,
        }

    @property
    def ee_pose_matrix(self) -> np.ndarray:
        return self.state["end_effector_pose"].astype(np.float32)

    @property
    def pose(self) -> np.ndarray:
        matrix = self.state["end_effector_pose"]
        return np.concatenate([matrix[:3, 3], _matrix_to_rpy(matrix[:3, :3])])

    def hold_current_pose(self) -> None:
        if self._active_arm_controller != "cartesian_pose_target_controller":
            if not self._activate_arm_controller("cartesian_pose_target_controller"):
                self._activate_arm_controller("joint_position_target_controller")
                self._publish_joint_target(self._current_qpos())
                return
        pos, quat = self._current_ee_pose()
        self.set_ee_control(pos, quat, self._current_qpos())

    @contextmanager
    def control_loop(self, frequency: float):
        period = 1.0 / max(float(frequency), 1e-3)
        last = time.perf_counter()
        running = True

        class _Ctx:
            def ok(_self) -> bool:
                nonlocal last, running
                if not running or self._shutdown or not rclpy.ok():
                    return False
                remaining = period - (time.perf_counter() - last)
                if remaining > 0.0:
                    time.sleep(remaining)
                last = time.perf_counter()
                return True

        try:
            yield _Ctx()
        finally:
            running = False

    def status_text(self) -> str:
        qpos = self._current_qpos()
        pos, quat = self._current_ee_pose()
        controllers = self.list_controllers()
        lines = [
            f"robot_type={self._robot_type} fake={self.use_fake_hardware}",
            f"joint_positions={np.array2string(qpos, precision=5, separator=', ')}",
        ]
        if self._have_ee_pose:
            lines.append(
                f"ee_position={np.array2string(pos, precision=5, separator=', ')}"
            )
            lines.append(
                f"ee_orientation_xyzw={np.array2string(quat, precision=5, separator=', ')}"
            )
        else:
            lines.append("ee_pose: unavailable (normal on fake hardware)")
        lines.append(f"gripper_position={self.state['gripper_position']:.4f}")
        if controllers:
            lines.append("controllers:")
            for name, state in controllers:
                lines.append(f"  {name}: {state}")
        else:
            lines.append("controllers: (controller_manager not reachable)")
        return "\n".join(lines)

    def cleanup(self) -> None:
        self._shutdown = True
        with suppress(Exception):
            self.wait_gripper(timeout=1.0)
        with suppress(Exception):
            self._gripper_stop_event.set()
            self._gripper_request.set()
        with suppress(Exception):
            if self._gripper_thread.is_alive():
                self._gripper_thread.join(timeout=1.0)
        with suppress(Exception):
            self._executor.cancel()
        with suppress(Exception):
            if self._owns_node:
                self._node.destroy_node()
        with suppress(Exception):
            if self._owns_context and rclpy.ok():
                rclpy.shutdown()
