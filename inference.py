#!/usr/bin/env python3
"""Run a Franka Cartesian policy and robot command loop at 30 Hz.

The policy server must serve a pi0_base_ee_ros_30hz checkpoint. Its actions
are 16 future absolute [xyz, roll/pitch/yaw, logical gripper] targets at 30 Hz.
Run this script in the ROS Humble environment; run the GPU server separately.
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from control.robot_config import config_path, load_mapping, section
from control.robot_state import EEPose, quat_angle_xyzw

ACTION_FPS = 30.0
ACTION_HORIZON = 16
ACTION_DIM = 7
IMAGE_HW = 224
PERIOD_NS = round(1e9 / ACTION_FPS)
GRIPPER_OPEN_WIDTH_M = 0.05
GRIPPER_OPEN_THRESHOLD_M = GRIPPER_OPEN_WIDTH_M / 2


@dataclass(frozen=True)
class ActionPlan:
    observation_ns: int
    actions: np.ndarray


def validate_policy_metadata(metadata: dict[str, Any]) -> None:
    """Reject a server running a checkpoint with different action timing."""
    if not isinstance(metadata, dict):
        raise ValueError("Policy server did not provide metadata")
    try:
        fps = float(metadata["action_fps"])
        horizon = int(metadata["action_horizon"])
        action_space = str(metadata["action_space"])
        action_dim = int(metadata["action_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Policy server must serve pi0_base_ee_ros_30hz with "
            "action_fps=30, action_horizon=16, action_space=ee, action_dim=7"
        ) from exc
    if (
        not np.isclose(fps, ACTION_FPS)
        or horizon != ACTION_HORIZON
        or action_space != "ee"
        or action_dim != ACTION_DIM
    ):
        raise ValueError(f"Incompatible policy metadata: {metadata}")


def build_observation(
    front: np.ndarray,
    wrist: np.ndarray,
    ee_pose: np.ndarray,
    gripper_open: float,
    prompt: str,
    frame_index: int,
) -> dict[str, Any]:
    """Send the keys read by the server's FrankaEEInputs transform directly."""
    front = np.asarray(front)
    wrist = np.asarray(wrist)
    pose = EEPose.from_vector(ee_pose)
    if front.shape != (IMAGE_HW, IMAGE_HW, 3) or front.dtype != np.uint8:
        raise ValueError("Front image must be 224x224x3 uint8 RGB")
    if wrist.shape != (IMAGE_HW, IMAGE_HW, 3) or wrist.dtype != np.uint8:
        raise ValueError("Wrist image must be 224x224x3 uint8 RGB")
    if not np.isfinite(gripper_open) or not 0.0 <= gripper_open <= 1.0:
        raise ValueError("Logical gripper observation must be in 0..1")
    return {
        "observation/exterior_image_1_left": front.copy(),
        "observation/wrist_image_left": wrist.copy(),
        "observation/ee_pose": pose.vector,
        "observation/gripper_position": np.array([gripper_open], dtype=np.float32),
        "prompt": prompt,
    }


def parse_action_plan(response: dict[str, Any], observation_ns: int) -> ActionPlan:
    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError("Policy response is missing actions")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite (16, 7) actions, got {actions.shape}")
    return ActionPlan(observation_ns=observation_ns, actions=actions)


def action_at(plan: ActionPlan, now_ns: int) -> tuple[np.ndarray | None, int]:
    """The first predicted target is for observation time + 1/30 second."""
    index = (now_ns - plan.observation_ns) // PERIOD_NS - 1
    if index < 0:
        return None, int(index)
    if index >= len(plan.actions):
        return None, int(index)
    return plan.actions[index], int(index)


def checked_ee_target(
    target: np.ndarray,
    measured: EEPose,
    max_translation_m: float,
    max_rotation_rad: float,
) -> EEPose:
    desired = EEPose.from_vector(np.asarray(target)[:6])
    translation = float(np.linalg.norm(desired.xyz_m - measured.xyz_m))
    rotation = quat_angle_xyzw(desired.quaternion_xyzw(), measured.quaternion_xyzw())
    if translation > max_translation_m or rotation > max_rotation_rad:
        raise ValueError(
            "Policy EE target differs from measured pose by "
            f"{translation:.3f} m and {rotation:.3f} rad"
        )
    return desired


class PolicyWorker:
    """Keep WebSocket inference off the fixed-rate robot command loop."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self._requests: queue.Queue[tuple[dict[str, Any], int] | None] = queue.Queue(
            maxsize=1
        )
        self._lock = threading.Lock()
        self._pending = False
        self._result: ActionPlan | None = None
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="policy-inference", daemon=True
        )
        self._thread.start()

    def submit(self, observation: dict[str, Any], observation_ns: int) -> bool:
        with self._lock:
            if self._pending or self._stop.is_set():
                return False
            self._pending = True
        self._requests.put_nowait((observation, observation_ns))
        return True

    def take_result(self) -> ActionPlan | None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Policy inference failed") from self._error
            result, self._result = self._result, None
            return result

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                request = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return
            observation, observation_ns = request
            try:
                plan = parse_action_plan(
                    self._policy.infer(observation), observation_ns
                )
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                    self._pending = False
                return
            with self._lock:
                self._result = plan
                self._pending = False

    def close(self) -> None:
        self._stop.set()
        self._policy.close()
        self._thread.join(timeout=2.0)


def _command_gripper(
    arm: Any, target_open: float, last_open: float | None
) -> float | None:
    if not np.isfinite(target_open) or not 0.0 <= target_open <= 1.0:
        raise ValueError(f"Policy logical gripper target outside 0..1: {target_open:.3f}")
    if arm.gripper_busy:
        return last_open
    command = float(target_open >= 0.5)
    if last_open == command:
        return last_open
    if command:
        arm.gripper_open(width=GRIPPER_OPEN_WIDTH_M, wait=False)
    else:
        arm.gripper_close(wait=False)
    return command


def run(
    config: dict[str, Any],
    *,
    host: str | None,
    port: int | None,
    prompt: str | None,
    max_steps: int | None,
    move_to_start: bool,
) -> None:
    # ROS imports stay here so --help and pure validation work outside Humble.
    from control.dual_camera_manager_ros import DualRealsenseManagerRos
    from control.hitl.policy import PolicyClient
    from control.recording_writer import FreshCameraPair
    from control.robotic_arm_controller_ros import RoboticArmControlerRos

    policy_cfg = section(config, "policy_server")
    inference_cfg = section(config, "inference")
    camera_cfg = section(config, "camera")
    robot_cfg = section(config, "robot")
    gripper_cfg = section(config, "gripper")
    if str(camera_cfg.camera_backend) != "ros":
        raise ValueError("Inference requires camera.camera_backend: ros")
    if int(camera_cfg.image_hw) != IMAGE_HW or not np.isclose(
        float(camera_cfg.camera_fps), ACTION_FPS
    ):
        raise ValueError("Inference requires 224x224 images and 30 Hz cameras")
    if str(gripper_cfg.gripper_type) != "franka":
        raise ValueError("This 8D joint policy requires gripper.gripper_type: franka")
    if not np.isclose(float(inference_cfg.action_fps), ACTION_FPS):
        raise ValueError("Inference action_fps must be 30")
    replan_steps = int(inference_cfg.replan_every_steps)
    if not 1 <= replan_steps < ACTION_HORIZON:
        raise ValueError("replan_every_steps must be between 1 and 15")
    translation_limit = float(inference_cfg.max_ee_translation_step_m)
    rotation_limit = float(inference_cfg.max_ee_rotation_step_rad)
    camera_age_ns = round(float(inference_cfg.max_camera_age_s) * 1e9)
    if translation_limit <= 0 or rotation_limit <= 0 or camera_age_ns <= 0:
        raise ValueError("EE step and camera age limits must be positive")
    step_limit = int(inference_cfg.max_steps if max_steps is None else max_steps)
    if step_limit < 0:
        raise ValueError("max_steps must be nonnegative")
    task = str(inference_cfg.prompt if prompt is None else prompt)
    if not task.strip():
        raise ValueError("A task prompt is required")

    policy = PolicyClient(
        host=str(policy_cfg.host if host is None else host),
        port=int(policy_cfg.port if port is None else port),
    )
    cameras = None
    arm = None
    worker = None
    try:
        validate_policy_metadata(policy.server_metadata)
        cameras = DualRealsenseManagerRos(
            external_topic=str(camera_cfg.external_image_topic),
            wrist_topic=str(camera_cfg.wrist_image_topic),
            external_serial=str(camera_cfg.external_camera_serial),
            wrist_serial=str(camera_cfg.wrist_camera_serial),
            crop_scale=float(camera_cfg.crop_scale),
            out_hw=IMAGE_HW,
            refresh_hz=ACTION_FPS,
            enabled=True,
        )
        cameras.wait_for_frames(timeout_s=float(camera_cfg.camera_startup_timeout_s))
        arm = RoboticArmControlerRos(
            use_fake_hardware=bool(robot_cfg.use_fake_hardware),
            robot_type=str(robot_cfg.robot_type),
            start_joint_position=list(robot_cfg.start_joint_position),
        )
        if not arm.joint_state_ready or not arm.ee_pose_ready:
            raise RuntimeError("No measured joint or EE state; refusing to run inference")
        if move_to_start or bool(inference_cfg.move_to_start):
            arm.move_to_start()
        arm.start_ee_streaming()
        if not arm.ee_streaming_active:
            raise RuntimeError("Cartesian controller is unavailable for EE inference")
        worker = PolicyWorker(policy)
        fresh_pair = FreshCameraPair()
        plan: ActionPlan | None = None
        last_gripper_open: float | None = None
        tick_ns = time.monotonic_ns()
        steps = 0
        logging.info(
            "Inference started: prompt=%r, 30 Hz, max_steps=%s", task, step_limit
        )

        while step_limit <= 0 or steps < step_limit:
            now_ns = time.monotonic_ns()
            if now_ns < tick_ns:
                time.sleep((tick_ns - now_ns) / 1e9)
            front, wrist, front_ts, wrist_ts = cameras.get_frames()
            now_ns = time.monotonic_ns()
            if front is None or wrist is None or front_ts is None or wrist_ts is None:
                raise RuntimeError("Camera frames unavailable during inference")
            front_ns = int(front_ts.host_capture_monotonic_ns)
            wrist_ns = int(wrist_ts.host_capture_monotonic_ns)
            if max(now_ns - front_ns, now_ns - wrist_ns) > camera_age_ns:
                raise RuntimeError("Camera frame stale during inference")
            if front_ns > now_ns or wrist_ns > now_ns:
                raise RuntimeError("Camera frame timestamp is in the future")

            measured = arm.state
            current_joints = np.asarray(measured["joint_positions"], dtype=np.float64)
            current_ee = EEPose.from_matrix(measured["end_effector_pose"])
            new_plan = worker.take_result()
            if new_plan is not None:
                plan = new_plan
            action, index = action_at(plan, now_ns) if plan is not None else (None, 0)
            if action is None:
                arm.set_ee_control(current_ee.xyz_m, current_ee.quaternion_xyzw(), current_joints)
            else:
                desired_ee = checked_ee_target(
                    action, current_ee, translation_limit, rotation_limit
                )
                arm.set_ee_control(desired_ee.xyz_m, desired_ee.quaternion_xyzw(), current_joints)
                last_gripper_open = _command_gripper(
                    arm, float(action[6]), last_gripper_open
                )

            if (plan is None or index >= replan_steps) and fresh_pair.accept(
                front_ns, wrist_ns
            ):
                observation = build_observation(
                    front,
                    wrist,
                    current_ee.vector,
                    float(
                        float(measured["gripper_position"])
                        >= GRIPPER_OPEN_THRESHOLD_M
                    ),
                    task,
                    steps,
                )
                worker.submit(observation, now_ns)
            steps += 1
            tick_ns += PERIOD_NS
            if tick_ns < time.monotonic_ns():
                tick_ns = time.monotonic_ns() + PERIOD_NS
    finally:
        if arm is not None:
            try:
                current_state = arm.state
                hold = EEPose.from_matrix(current_state["end_effector_pose"])
                arm.set_ee_control(
                    hold.xyz_m, hold.quaternion_xyzw(), current_state["joint_positions"]
                )
            except Exception:
                logging.exception("Failed to hold current joint position")
        try:
            if worker is not None:
                worker.close()
            else:
                policy.close()
        finally:
            try:
                if arm is not None:
                    arm.cleanup()
            finally:
                if cameras is not None:
                    cameras.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="30 Hz Franka inference without recording"
    )
    parser.add_argument(
        "--robot", default="franka", help="Config name under config/inference/"
    )
    parser.add_argument("--config", type=Path, help="Override inference YAML config")
    parser.add_argument("--host", help="Override policy server hostname or ws:// URI")
    parser.add_argument("--port", type=int, help="Override policy server port")
    parser.add_argument("--prompt", help="Override task instruction")
    parser.add_argument(
        "--max-steps", type=int, help="30 Hz control ticks; 0 runs until Ctrl+C"
    )
    parser.add_argument(
        "--move-to-start",
        action="store_true",
        help="Move to configured start joints first",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_mapping(
        config_path(stage="inference", robot=args.robot, explicit=args.config)
    )
    try:
        run(
            config,
            host=args.host,
            port=args.port,
            prompt=args.prompt,
            max_steps=args.max_steps,
            move_to_start=args.move_to_start,
        )
    except KeyboardInterrupt:
        logging.info("Inference stopped by Ctrl+C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
