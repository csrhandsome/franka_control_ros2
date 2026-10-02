"""Private gripper action clients and worker; no public operation handles."""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Literal
from rclpy.action import ActionClient
from std_srvs.srv import Trigger
from . import lifecycle, motion, gripper


@dataclass
class _GripperCommand:
    kind: Literal["move", "grasp"]
    params: dict[str, float]
    done: threading.Event = field(default_factory=threading.Event)
    result: bool | None = None
    error: BaseException | None = None


def _init_gripper_clients(self):
    if not self._config["gripper"]["enabled"]:
        return
    try:
        from franka_msgs.action import Grasp, Move
    except ImportError:
        self._node.get_logger().warn("[FrankaROS2] franka_msgs actions not available.")
        return
    self._gripper_move = ActionClient(self._node, Move, self._config["gripper"]["move_action"])
    self._gripper_grasp = ActionClient(self._node, Grasp, self._config["gripper"]["grasp_action"])
    self._gripper_stop = lifecycle._track_entity(
        self, "client", self._node.create_client(Trigger, self._config["gripper"]["stop_service"])
    )


def _stop_gripper(self, *, timeout_s=None):
    lifecycle._require_connected(self)
    if self._gripper_stop is None:
        raise RuntimeError("Gripper is disabled or unavailable")
    result = lifecycle._call_service(
        self, self._gripper_stop, Trigger.Request(), timeout_s=timeout_s
    )
    if result is None or not result.success:
        raise RuntimeError("Gripper stop failed")


def _gripper_worker(self):
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
                command.result = gripper._send_gripper_goal(self, self._gripper_move, goal)
            else:
                from franka_msgs.action import Grasp

                goal = Grasp.Goal()
                goal.width = float(command.params["width"])
                goal.speed = float(command.params["speed"])
                goal.force = float(command.params["force"])
                goal.epsilon.inner = float(command.params["epsilon_inner"])
                goal.epsilon.outer = float(command.params["epsilon_outer"])
                command.result = gripper._send_gripper_goal(self, self._gripper_grasp, goal)
        except BaseException as exc:
            command.error = exc
            command.result = False
        with self._gripper_lock:
            self._gripper_active_command = None
        command.done.set()


def _submit_gripper_command(
    self,
    kind: Literal["move", "grasp"],
    params: dict[str, float],
    *,
    wait: bool = True,
    timeout: float | None = None,
):
    lifecycle._require_connected(self)
    command = _GripperCommand(kind=kind, params=dict(params))
    with self._gripper_lock:
        self._gripper_command = command
    self._gripper_request.set()
    if not wait:
        return True
    if not command.done.wait(timeout=timeout):
        raise TimeoutError("Timed out waiting for gripper command")
    if command.error is not None:
        if self._use_fake_hardware:
            print(f"[FrankaROS2] Gripper {kind} skipped on fake hardware: {command.error}")
            return True
        raise RuntimeError("Gripper command failed") from command.error
    return bool(command.result)


def _send_gripper_goal(self, client, goal):
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
    deadline = time.time() + self._config["gripper"]["command_timeout_s"]
    while not result_future.done() and time.time() < deadline:
        time.sleep(0.02)
    if not result_future.done():
        motion._wait_future(self, handle.cancel_goal_async(), self._command_timeout_s)
        raise TimeoutError("Gripper action timed out and cancellation was requested")
    from action_msgs.msg import GoalStatus

    result = result_future.result()
    return bool(result.status == GoalStatus.STATUS_SUCCEEDED and result.result.success)


def _gripper_open(
    self,
    width: float = 0.05,
    speed: float = 0.2,
    *,
    wait: bool = True,
    timeout: float | None = None,
):
    return gripper._submit_gripper_command(
        self, "move", {"width": width, "speed": speed}, wait=wait, timeout=timeout
    )


def _gripper_close(
    self,
    width: float = 0.0,
    speed: float = 0.2,
    force: float = 60.0,
    epsilon_inner: float = 0.04,
    epsilon_outer: float = 0.04,
    *,
    wait: bool = True,
    timeout: float | None = None,
):
    return gripper._submit_gripper_command(
        self,
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


def _busy(self):
    with self._gripper_lock:
        command = self._gripper_command
        active = self._gripper_active_command
    return (
        command is not None
        and (not command.done.is_set())
        or (active is not None and (not active.done.is_set()))
    )


def _wait_gripper(self, timeout: float | None = None):
    lifecycle._require_connected(self)
    with self._gripper_lock:
        command = self._gripper_command or self._gripper_active_command
    if command is None:
        return
    if not command.done.wait(timeout=timeout):
        raise TimeoutError("Timed out waiting for gripper")
