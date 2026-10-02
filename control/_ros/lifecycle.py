"""Private ROS resources and control-mode transitions."""

from __future__ import annotations
import math
import os
import threading
import time
from functools import partial
from collections.abc import Sequence
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control.motion_config import load_motion_config
from . import lifecycle, state, motion, gripper


def _connect_resources(self, node):
    if node is not None and node.executor is not None:
        raise ValueError("node is already assigned to an executor; pass an unassigned node")
    if node is None:
        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True
            self._owned_context = rclpy.get_default_context()
        self._node = Node("franka_ros2_control")
        self._owns_node = True
    else:
        if not node.context.ok():
            raise ValueError("The external node's ROS context is not running")
        self._node = node
    self._executor = MultiThreadedExecutor(num_threads=2, context=self._node.context)
    self._executor.add_node(self._node)
    self._spin_thread = threading.Thread(
        target=partial(lifecycle._spin, self), name="franka-ros2-spin", daemon=True
    )
    best_effort = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
    )
    reliable = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1
    )
    self._pose_pub = _track_entity(
        self,
        "publisher",
        self._node.create_publisher(
            PoseStamped, self._config["ee"]["streaming"]["target_topic"], best_effort
        ),
    )
    self._joint_pub = _track_entity(
        self,
        "publisher",
        self._node.create_publisher(
            Float64MultiArray, self._config["joint"]["streaming"]["target_topic"], 10
        ),
    )
    _track_entity(
        self,
        "subscription",
        self._node.create_subscription(
            JointState,
            self._config["ros"]["state"]["joint_topic"],
            partial(state._on_joint_states, self),
            reliable,
        ),
    )
    _track_entity(
        self,
        "subscription",
        self._node.create_subscription(
            PoseStamped,
            self._config["ros"]["state"]["ee_pose_topic"],
            partial(state._on_current_pose, self),
            best_effort,
        ),
    )
    if self._robot_type == "panda":
        from franka_msgs.msg import FrankaState

        _track_entity(
            self,
            "subscription",
            self._node.create_subscription(
                FrankaState,
                self._config["ros"]["state"]["robot_state_topic"],
                partial(state._on_panda_state, self),
                best_effort,
            ),
        )
    _track_entity(
        self,
        "subscription",
        self._node.create_subscription(
            JointState,
            self._config["gripper"]["state_topic"],
            partial(state._on_gripper_states, self),
            reliable,
        ),
    )
    self._switch_client = _track_entity(
        self,
        "client",
        self._node.create_client(
            lifecycle._switch_srv_type(),
            self._config["ros"]["controller_manager"] + "/switch_controller",
        ),
    )
    self._list_client = _track_entity(
        self,
        "client",
        self._node.create_client(
            lifecycle._list_srv_type(),
            self._config["ros"]["controller_manager"] + "/list_controllers",
        ),
    )
    self._load_client = _track_entity(
        self,
        "client",
        self._node.create_client(
            lifecycle._load_srv_type(),
            self._config["ros"]["controller_manager"] + "/load_controller",
        ),
    )
    self._configure_client = _track_entity(
        self,
        "client",
        self._node.create_client(
            lifecycle._configure_srv_type(),
            self._config["ros"]["controller_manager"] + "/configure_controller",
        ),
    )
    self._gripper_move = None
    self._gripper_grasp = None
    self._gripper_stop = None
    gripper._init_gripper_clients(self)
    self._gripper_lock = threading.Lock()
    self._gripper_command: gripper._GripperCommand | None = None
    self._gripper_active_command: gripper._GripperCommand | None = None
    self._gripper_request = threading.Event()
    self._gripper_stop_event = threading.Event()
    self._gripper_thread = threading.Thread(
        target=partial(gripper._gripper_worker, self), name="franka-ros2-gripper", daemon=True
    )
    self._gripper_thread.start()
    self._spin_thread.start()


def _initialize(
    self,
    *,
    config_path=None,
    node=None,
    use_fake_hardware=None,
    robot_type=None,
    command_timeout_s=None,
    motion_config=None,
):
    self._config = load_motion_config(
        config_path, robot_type=robot_type or "panda", overrides=motion_config
    )
    if robot_type is not None and robot_type != self._config["robot"]["robot_type"]:
        raise ValueError("robot_type conflicts with the motion configuration")
    self._use_fake_hardware = (
        self._config["robot"]["use_fake_hardware"]
        if use_fake_hardware is None
        else bool(use_fake_hardware)
    )
    self._robot_ip = os.environ.get("FRANKA_ROBOT_IP")
    self._robot_type = self._config["robot"]["robot_type"]
    self._command_timeout_s = float(
        self._config["execution"]["service_timeout_s"]
        if command_timeout_s is None
        else command_timeout_s
    )
    if not math.isfinite(self._command_timeout_s) or self._command_timeout_s <= 0:
        raise ValueError("command_timeout_s must be positive and finite")
    self._shutdown = False
    self._owns_context = False
    self._owned_context = None
    self._owns_node = False
    self._ee_streaming = False
    self._active_arm_controller: str | None = None
    self._last_ee_pos = np.array([0.3, 0.0, 0.5], dtype=np.float64)
    self._last_ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    self._joint_positions = np.zeros(7, dtype=np.float64)
    self._joint_velocities = np.zeros(7, dtype=np.float64)
    self._gripper_position = 0.07
    self._have_joint_state = False
    self._have_joint_velocity = False
    self._have_ee_pose = False
    self._have_gripper_state = False
    self._robot_fault = None
    self._state_received = {"joints": None, "ee": None, "gripper": None}
    self._state_stamps = {"joints": None, "ee": None, "gripper": None}
    self._stream_space: str | None = None
    self._stream_origin = None
    self._last_command = None
    self._last_command_time = None
    self._last_command_velocity = None
    self._trajectory_running = False
    self._cancel_requested = threading.Event()
    self._trajectory_goal = None
    self._state_lock = threading.Lock()
    self._connected = False
    self._external_node = node
    self._command_lock = threading.RLock()
    self._motion_run = None
    self._closing_thread = None
    self._closed_event = threading.Event()
    self._close_error = None
    self._ros_entities = []


def _spin(self):
    try:
        self._executor.spin()
    except Exception:
        if not self._shutdown:
            raise


def _switch_srv_type():
    from controller_manager_msgs.srv import SwitchController

    return SwitchController


def _list_srv_type():
    from controller_manager_msgs.srv import ListControllers

    return ListControllers


def _load_srv_type():
    from controller_manager_msgs.srv import LoadController

    return LoadController


def _configure_srv_type():
    from controller_manager_msgs.srv import ConfigureController

    return ConfigureController


def _call_service(self, client, request, timeout_s=None):
    timeout = _timeout(self._command_timeout_s if timeout_s is None else timeout_s)
    deadline = time.monotonic() + timeout
    if not client.wait_for_service(timeout_sec=timeout):
        return None
    future = client.call_async(request)
    while not future.done() and (not self._shutdown):
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.01)
    return future.result() if future.done() else None


def _ensure_loaded(self, name: str):
    from controller_manager_msgs.srv import ConfigureController, LoadController

    status = dict(state._list_controllers(self)).get(name)
    if status in ("inactive", "active"):
        return
    if status is None:
        request = LoadController.Request()
        request.name = name
        result = lifecycle._call_service(self, self._load_client, request)
        if result is None:
            raise TimeoutError(f"Timed out loading controller {name}")
        if not result.ok:
            raise RuntimeError(f"Failed to load controller {name}; check controller-manager logs")
        status = "unconfigured"
    if status != "unconfigured":
        raise RuntimeError(f"Controller {name} has unexpected lifecycle state {status}")
    request = ConfigureController.Request()
    request.name = name
    result = lifecycle._call_service(self, self._configure_client, request)
    if result is None:
        raise TimeoutError(f"Timed out configuring controller {name}")
    if not result.ok:
        raise RuntimeError(f"Failed to configure controller {name}; check controller-manager logs")


def _switch_controllers(self, *, activate: Sequence[str] = (), deactivate: Sequence[str] = ()):
    from controller_manager_msgs.srv import SwitchController

    activate_list = [name for name in activate if name]
    deactivate_list = [name for name in deactivate if name]
    for name in activate_list:
        lifecycle._ensure_loaded(self, name)
    request = SwitchController.Request()
    if hasattr(request, "activate_controllers"):
        request.activate_controllers = list(activate_list)
        request.deactivate_controllers = list(deactivate_list)
    else:
        request.start_controllers = list(activate_list)
        request.stop_controllers = list(deactivate_list)
    if hasattr(request, "strictness"):
        request.strictness = getattr(SwitchController.Request, "STRICT", 2)
    if hasattr(request, "activate_asap"):
        request.activate_asap = True
    if hasattr(request, "timeout"):
        request.timeout.sec = int(self._command_timeout_s)
        request.timeout.nanosec = int(self._command_timeout_s % 1.0 * 1000000000.0)
    result = lifecycle._call_service(self, self._switch_client, request, timeout_s=5.0)
    return bool(result is not None and getattr(result, "ok", True))


def _activate_arm_controller(self, name):
    controllers = dict(state._list_controllers(self))
    if controllers.get(name) == "active":
        self._active_arm_controller = name
        return True
    configured = {
        self._config["joint"]["streaming"]["controller"],
        self._config["joint"]["trajectory"]["controller"],
        self._config["ee"]["streaming"]["controller"],
    }
    if self._active_arm_controller:
        configured.add(self._active_arm_controller)
    deactivate = [
        controller
        for controller in configured
        if controller != name and controllers.get(controller) == "active"
    ]
    ok = _switch_controllers(self, activate=[name], deactivate=deactivate)
    if ok:
        self._active_arm_controller = name
        self._node.get_logger().info(f"Active controller: {name}")
    return ok


def _start_stream(self, space):
    with self._command_lock:
        _require_connected(self)
        motion._require_no_motion(self)
        if self._stream_space is not None:
            raise RuntimeError("A stream is active; call stop_stream() before starting another")
        controller = motion._check_available(self, space)
        if not _activate_arm_controller(self, controller):
            raise RuntimeError(f"Could not activate {controller}")
        self._stream_space = space
        self._stream_origin = (
            state._current_qpos(self) if space == "joint" else state._current_ee_pose(self)
        )
        self._last_command = self._last_command_time = self._last_command_velocity = None
        try:
            if space == "joint":
                motion._send_joint_target(self, state._current_qpos(self))
            else:
                motion._send_ee_target(self, *state._current_ee_pose(self))
        except BaseException:
            _deactivate_controller(self)
            raise


def _deactivate_controller(self):
    name = self._active_arm_controller
    if name and (not lifecycle._switch_controllers(self, deactivate=[name])):
        raise RuntimeError(f"Could not deactivate {name}")
    self._active_arm_controller = None
    self._stream_space = None
    self._ee_streaming = False


def _close(self):
    with self._command_lock:
        if self._shutdown:
            if self._close_error is not None:
                raise self._close_error
            return
        if self._closing_thread is not None:
            if self._closing_thread == threading.get_ident():
                return
            wait_for_close = True
        else:
            self._closing_thread = threading.get_ident()
            wait_for_close = False
    if wait_for_close:
        if not self._closed_event.wait(self._command_timeout_s):
            raise TimeoutError("Robot client close was not completed")
        if self._close_error is not None:
            raise self._close_error
        return
    error = None
    try:
        if self._connected:
            motion._cancel_motion(self, timeout_s=self._command_timeout_s)
            if self._stream_space is not None:
                _stop_stream(self)
            elif self._active_arm_controller is not None:
                _deactivate_controller(self)
            if gripper._busy(self):
                gripper._stop_gripper(self)
    except Exception as exc:
        error = exc
    finally:
        self._shutdown = True
        self._connected = False
        try:
            _release_resources(self)
        except Exception as exc:
            if error is None:
                error = exc
        finally:
            self._close_error = error
            self._closed_event.set()
    if error is not None:
        raise error


def _release_resources(self):
    if hasattr(self, "_gripper_thread"):
        self._gripper_stop_event.set()
        self._gripper_request.set()
        self._gripper_thread.join(timeout=self._command_timeout_s)
        if self._gripper_thread.is_alive():
            raise TimeoutError("Gripper worker did not stop")
    if hasattr(self, "_executor"):
        stopped = self._executor.shutdown(timeout_sec=self._command_timeout_s)
        self._spin_thread.join(timeout=self._command_timeout_s)
        if self._spin_thread.is_alive():
            raise TimeoutError("ROS spin thread did not stop")
        # Humble's executor shutdown does not shut down its ThreadPoolExecutor.
        # Stop submissions first, drain/join workers before destroying any ROS entity.
        pool = self._executor._executor
        pool.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + self._command_timeout_s
        for thread in list(pool._threads):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in pool._threads):
            raise TimeoutError("ROS executor workers did not stop")
        pool.shutdown(wait=True)
        # Cancel suspended or unsubmitted rclpy Tasks too; Task.cancel() closes
        # their coroutine handlers, preventing leaks and unawaited warnings.
        for task in list(self._executor._pending_tasks):
            if not task.done():
                task.cancel()
        self._executor._pending_tasks.clear()
        self._executor._ready_tasks.clear()
        self._executor._futures.clear()
        if not stopped:
            raise TimeoutError("ROS executor callbacks did not finish before shutdown")
        self._executor.remove_node(self._node)
        self._node.executor = None
    for client in (getattr(self, "_gripper_move", None), getattr(self, "_gripper_grasp", None)):
        if client is not None:
            client.destroy()
    for kind, entity in reversed(self._ros_entities):
        getattr(self._node, "destroy_" + kind)(entity)
    self._ros_entities.clear()
    if self._owns_node:
        self._node.destroy_node()
    if self._owns_context:
        if rclpy.ok():
            rclpy.shutdown()
        # Shutdown alone retains the underlying context until Python GC runs.
        # Finalize it explicitly after all nodes/executors have been released.
        self._owned_context.destroy()


def _timeout(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_s must be positive and finite")
    return value


def _require_connected(self):
    if self._shutdown:
        raise RuntimeError("Robot client is closed")
    if self._closing_thread is not None and self._closing_thread != threading.get_ident():
        raise RuntimeError("Robot client is closing")
    if not self._connected:
        raise RuntimeError("Robot client is not connected; call connect() first")


def _connect(self, *, timeout_s=None):
    with self._command_lock:
        if self._shutdown:
            raise RuntimeError("Robot client is closed")
        if not self._connected:
            try:
                _connect_resources(self, self._external_node)
                self._connected = True
                state._wait_ready(self, required=("joints",), timeout_s=timeout_s)
            except BaseException:
                _close(self)
                raise
        else:
            state._wait_ready(self, required=("joints",), timeout_s=timeout_s)
        return state._get_status(self)


def _stop_stream(self, *, timeout_s=None):
    with self._command_lock:
        _require_connected(self)
        motion._require_no_motion(self)
        if self._stream_space is None:
            return
        timeout = self._command_timeout_s if timeout_s is None else _timeout(timeout_s)
        motion._hold(self)
        state._wait_until_stopped(self, timeout_s=timeout)
        _deactivate_controller(self)


def _track_entity(self, kind, entity):
    self._ros_entities.append((kind, entity))
    return entity
