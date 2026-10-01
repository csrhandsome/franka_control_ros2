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
from control.robot_state import EEPose
from control.util.pose import quat_angle_xyzw

logger = logging.getLogger(__name__)

ACTION_FPS = 30.0
ACTION_HORIZON = 16
ACTION_DIM = 7
IMAGE_HW = 224
WARMUP_CHUNKS = 5
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
        raise TypeError("Policy server did not provide metadata")
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


def warmup_policy(policy: Any) -> None:
    """Request and discard five chunks before starting the robot inference loop."""
    blank_image = np.zeros((IMAGE_HW, IMAGE_HW, 3), dtype=np.uint8)
    blank_observation = build_observation(
        blank_image,
        blank_image,
        np.zeros(6, dtype=np.float32),
        0.0,
        "",
        0,
    )
    for request_index in range(WARMUP_CHUNKS):
        try:
            parse_action_plan(policy.infer(blank_observation), observation_ns=0)
        except Exception as exc:
            raise RuntimeError(
                f"Policy warmup failed on request {request_index + 1}/{WARMUP_CHUNKS}"
            ) from exc
    logger.info("Policy warmup completed: %d action chunks", WARMUP_CHUNKS)


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

    def __init__(self, policy: Any, *, parse_response=parse_action_plan) -> None:
        self._policy = policy
        self._parse_response = parse_response
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
                plan = self._parse_response(
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
        raise ValueError(
            f"Policy logical gripper target outside 0..1: {target_open:.3f}"
        )
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
        if policy.server_metadata.get("rlt_actor_protocol"):
            run_force_rlt(
                config,
                policy=policy,
                prompt=task,
                max_steps=step_limit,
                move_to_start=move_to_start,
            )
            return
        validate_policy_metadata(policy.server_metadata)
        warmup_policy(policy)
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
            raise RuntimeError(
                "No measured joint or EE state; refusing to run inference"
            )
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
        logger.info(
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
                arm.set_ee_control(
                    current_ee.xyz_m, current_ee.quaternion_xyzw(), current_joints
                )
            else:
                desired_ee = checked_ee_target(
                    action, current_ee, translation_limit, rotation_limit
                )
                arm.set_ee_control(
                    desired_ee.xyz_m, desired_ee.quaternion_xyzw(), current_joints
                )
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
                        float(measured["gripper_position"]) >= GRIPPER_OPEN_THRESHOLD_M
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
                logger.exception("Failed to hold current joint position")
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


@dataclass(frozen=True)
class ForceActionPlan:
    observation_ns: int
    actions: np.ndarray
    ref_q: np.ndarray


def validate_force_metadata(metadata: dict[str, Any]) -> tuple[str, int]:
    """Check that the single server loaded a matching VLA, readout, and actor."""
    if (
        not isinstance(metadata, dict)
        or metadata.get("rlt_protocol") != "pytorch-rl-token-v1"
        or metadata.get("rlt_actor_protocol") != "pytorch-force-rlt-actor-v1"
        or metadata.get("action_space") != "joint"
        or not np.isclose(float(metadata.get("action_fps", 0)), ACTION_FPS)
        or int(metadata.get("action_horizon", 0)) != ACTION_HORIZON
        or int(metadata.get("prediction_horizon", 0)) != ACTION_HORIZON
        or int(metadata.get("action_dim", 0)) != 7
        or int(metadata.get("physical_joint_dim", 0)) != 7
        or metadata.get("group") not in ("B", "C")
        or not metadata.get("actor_sha256")
    ):
        raise ValueError("Incompatible force RLT server metadata")
    execution_horizon = int(metadata.get("execution_horizon", 0))
    if not 0 < execution_horizon <= ACTION_HORIZON:
        raise ValueError("Invalid force RLT execution horizon")
    return str(metadata["group"]), execution_horizon


def build_force_observation(
    front: np.ndarray,
    wrist: np.ndarray,
    joints: np.ndarray,
    tcp: np.ndarray,
    gripper_open: float,
    prompt: str,
    observation_ns: int,
    step_id: int,
) -> dict[str, Any]:
    """Send the joint-policy inputs plus the readout's timestamp and TCP anchor."""
    front = np.asarray(front)
    wrist = np.asarray(wrist)
    joints = np.asarray(joints, dtype=np.float32)
    tcp = np.asarray(tcp, dtype=np.float32)
    if any(
        image.shape != (IMAGE_HW, IMAGE_HW, 3) or image.dtype != np.uint8
        for image in (front, wrist)
    ):
        raise ValueError("Force RLT cameras must provide 224x224x3 uint8 RGB")
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("Measured joints must be finite with shape (7,)")
    if (
        tcp.shape != (4, 4)
        or not np.isfinite(tcp).all()
        or not np.allclose(tcp[3], [0, 0, 0, 1])
    ):
        raise ValueError("Measured O_T_TCP must be a finite homogeneous 4x4 matrix")
    if not np.isfinite(gripper_open) or not 0 <= gripper_open <= 1:
        raise ValueError("Logical gripper observation must be in 0..1")
    return {
        "observation/exterior_image_1_left": front.copy(),
        "observation/wrist_image_left": wrist.copy(),
        "observation/joint_position": joints.copy(),
        "observation/gripper_position": np.array([gripper_open], dtype=np.float32),
        "prompt": prompt,
        "monotonic_ts": np.array(observation_ns / 1e9, dtype=np.float64),
        "O_T_TCP": tcp.copy(),
        "step_id": step_id,
    }


def parse_force_action_plan(
    response: dict[str, Any], observation_ns: int, metadata: dict[str, Any]
) -> ForceActionPlan:
    """Use actor proposals only when they belong to this server and observation."""
    group, _ = validate_force_metadata(metadata)
    if not isinstance(response, dict) or response.get("group") != group:
        raise ValueError("Force RLT response group mismatch")
    if response.get("actor_sha256") != metadata["actor_sha256"]:
        raise ValueError("Force RLT actor hash mismatch")
    timestamp = np.asarray(response.get("ref_obs_ts", np.nan), dtype=np.float64)
    if (
        timestamp.shape != ()
        or not np.isfinite(timestamp)
        or abs(float(timestamp) - observation_ns / 1e9) > 1e-6
    ):
        raise ValueError("Force RLT reference observation timestamp mismatch")
    anchor = np.asarray(response.get("ref_anchor_TCP", []), dtype=np.float64)
    if anchor.shape != (4, 4) or not np.isfinite(anchor).all():
        raise ValueError("Force RLT reference TCP anchor is missing")
    actions = np.asarray(response.get("proposed_q", []), dtype=np.float32)
    ref_q = np.asarray(response.get("ref_q", []), dtype=np.float32)
    for name, value in (("proposed_q", actions), ("ref_q", ref_q)):
        if value.shape != (ACTION_HORIZON, 7) or not np.isfinite(value).all():
            raise ValueError(f"Force RLT {name} must be finite with shape (16, 7)")
    return ForceActionPlan(observation_ns, actions, ref_q)


def checked_force_target(
    target: np.ndarray,
    reference: np.ndarray,
    measured: np.ndarray,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    max_step: np.ndarray,
    reference_radius: np.ndarray,
) -> np.ndarray:
    """Bound a joint proposal before a fake-hardware command."""
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if any(
        value.shape != (7,) or not np.isfinite(value).all()
        for value in (target, reference, measured)
    ):
        raise ValueError("Force RLT joint vectors must be finite with shape (7,)")
    if (
        np.any(target < joint_lower)
        or np.any(target > joint_upper)
        or np.any(np.abs(target - measured) > max_step)
        or np.any(np.abs(target - reference) > reference_radius)
    ):
        raise ValueError("Force RLT proposal exceeds configured joint bounds")
    return target


def run_force_rlt(
    config: dict[str, Any],
    *,
    policy: Any,
    prompt: str,
    max_steps: int,
    move_to_start: bool,
) -> None:
    """Query the merged server; hold the robot unless fake-hardware execution is enabled."""
    from control.dual_camera_manager_ros import DualRealsenseManagerRos
    from control.recording_writer import FreshCameraPair
    from control.robotic_arm_controller_ros import RoboticArmControlerRos

    group, execution_horizon = validate_force_metadata(policy.server_metadata)
    if group == "C":
        raise RuntimeError(
            "C requires synchronized finger-history features; this inference client has no finger camera pipeline"
        )
    inference_cfg = section(config, "inference")
    force_cfg = section(config, "force_rlt")
    camera_cfg = section(config, "camera")
    robot_cfg = section(config, "robot")
    execute = bool(getattr(force_cfg, "execute_joint_targets", False))
    if execute and not bool(robot_cfg.use_fake_hardware):
        raise RuntimeError(
            "Real-hardware force RLT execution requires a calibrated TCP/IK safety projector"
        )
    limits = None
    if execute:
        names = (
            "joint_lower",
            "joint_upper",
            "max_joint_step_rad",
            "reference_radius_rad",
        )
        limits = {}
        for name in names:
            value = np.asarray(getattr(force_cfg, name), dtype=np.float64)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"force_rlt.{name} must contain seven finite values")
            limits[name] = value
        if np.any(limits["joint_lower"] >= limits["joint_upper"]) or any(
            np.any(limits[name] <= 0)
            for name in ("max_joint_step_rad", "reference_radius_rad")
        ):
            raise ValueError("Invalid force RLT joint limits")
    if str(camera_cfg.camera_backend) != "ros" or int(camera_cfg.image_hw) != IMAGE_HW:
        raise ValueError("Force RLT requires ROS 224x224 cameras")
    if not np.isclose(float(camera_cfg.camera_fps), ACTION_FPS):
        raise ValueError("Force RLT requires 30 Hz cameras")
    replan_steps = min(int(inference_cfg.replan_every_steps), execution_horizon)
    if replan_steps <= 0:
        raise ValueError("Force RLT replanning interval must be positive")
    max_age_ns = round(float(getattr(force_cfg, "max_ref_age_s", 0.2)) * 1e9)
    camera_age_ns = round(float(inference_cfg.max_camera_age_s) * 1e9)
    if max_age_ns <= 0 or camera_age_ns <= 0:
        raise ValueError("Force RLT reference and camera age limits must be positive")
    cameras = None
    arm = None
    worker = None
    try:
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
            raise RuntimeError("Measured joint/TCP state is unavailable")
        if move_to_start or bool(inference_cfg.move_to_start):
            arm.move_to_start()
        arm.start_joint_streaming()
        worker = PolicyWorker(
            policy,
            parse_response=lambda response, stamp: parse_force_action_plan(
                response, stamp, policy.server_metadata
            ),
        )
        fresh_pair = FreshCameraPair()
        plan: ForceActionPlan | None = None
        tick_ns = time.monotonic_ns()
        steps = 0
        logger.info("Force RLT %s started: execute_joint_targets=%s", group, execute)
        while max_steps <= 0 or steps < max_steps:
            now_ns = time.monotonic_ns()
            if now_ns < tick_ns:
                time.sleep((tick_ns - now_ns) / 1e9)
            front, wrist, front_ts, wrist_ts = cameras.get_frames()
            now_ns = time.monotonic_ns()
            if front is None or wrist is None or front_ts is None or wrist_ts is None:
                raise RuntimeError(
                    "Camera frames unavailable during force RLT inference"
                )
            front_ns = int(front_ts.host_capture_monotonic_ns)
            wrist_ns = int(wrist_ts.host_capture_monotonic_ns)
            if any(
                stamp > now_ns or now_ns - stamp > camera_age_ns
                for stamp in (front_ns, wrist_ns)
            ):
                raise RuntimeError("Camera frame stale during force RLT inference")
            measured = arm.state
            q = np.asarray(measured["joint_positions"], dtype=np.float64)
            tcp = np.asarray(measured["end_effector_pose"], dtype=np.float32)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise RuntimeError("Measured joints are invalid")
            if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
                raise RuntimeError("Measured TCP pose is invalid")
            new_plan = worker.take_result()
            if new_plan is not None:
                plan = new_plan
            index = (
                -1 if plan is None else (now_ns - plan.observation_ns) // PERIOD_NS - 1
            )
            target = q
            if (
                plan is not None
                and 0 <= index < execution_horizon
                and now_ns - plan.observation_ns <= max_age_ns
                and execute
            ):
                target = checked_force_target(
                    plan.actions[index],
                    plan.ref_q[index],
                    q,
                    joint_lower=limits["joint_lower"],
                    joint_upper=limits["joint_upper"],
                    max_step=limits["max_joint_step_rad"],
                    reference_radius=limits["reference_radius_rad"],
                )
            arm.set_joint_control(target)
            if (plan is None or index >= replan_steps) and fresh_pair.accept(
                front_ns, wrist_ns
            ):
                observation = build_force_observation(
                    front,
                    wrist,
                    q,
                    tcp,
                    float(
                        float(measured["gripper_position"]) >= GRIPPER_OPEN_THRESHOLD_M
                    ),
                    prompt,
                    now_ns,
                    steps,
                )
                worker.submit(observation, now_ns)
            steps += 1
            tick_ns = max(tick_ns + PERIOD_NS, time.monotonic_ns() + PERIOD_NS)
    finally:
        if arm is not None:
            try:
                arm.set_joint_control(arm.state["joint_positions"])
            except Exception:
                logger.exception("Failed to hold measured joints")
        if worker is not None:
            worker.close()
        if arm is not None:
            arm.cleanup()
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
        logger.info("Inference stopped by Ctrl+C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
