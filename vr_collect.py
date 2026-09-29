#!/usr/bin/env python3
"""ROS Humble collection script for VR demonstrations.

Reads ``config/collect/<robot>.yaml``; runs in the Humble container::

    python vr_collect.py

- Cameras: DualRealsenseManagerRos (or camera_backend: none)
- Control: latest VR PoseStamped -> CartesianPoseTargetController at 100 Hz
- EE pose and joint angles are stored in every LeRobot frame
- No microphone / VAD path
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import websockets
import websockets.sync.client

from control.dual_camera_manager_ros import DualRealsenseManagerRos
from control.robot_state import EEPose, JointPose
from control.recording_writer import AsyncDatasetFrames, FreshCameraPair
from control.robot_config import (
    config_path,
    flatten_config,
    parse_action_space,
    parse_control_mode,
    parse_gripper_type,
)
from control.robotic_arm_controller_ros import RoboticArmControlerRos
from control.soft_gripper_control_ros import DH5GripperRos
from control.vr_input_mapper import VREEPoseMapper
from control.vr_input_ros import VRInputRos

_lerobot_import_error = None
try:
    from control.util.lerobot_util import (
        _discard_unsaved_episode,
        _load_or_create_dataset,
        _load_or_create_dataset_force,
        _prepare_episode_for_save,
        _prepare_episode_for_save_force,
    )
except Exception as exc:  # pragma: no cover
    _lerobot_import_error = exc
    _discard_unsaved_episode = None
    _load_or_create_dataset = None
    _load_or_create_dataset_force = None
    _prepare_episode_for_save = None
    _prepare_episode_for_save_force = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka ROS 2 VR collection")
    parser.add_argument(
        "--robot",
        default="franka",
        help="Robot yaml name under config/collect/, default franka",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override config path. Default: config/collect/<robot>.yaml",
    )
    return parser.parse_args()


def _load_config(config_file: Path) -> SimpleNamespace:
    return flatten_config(config_file)


class ReactiveDeskVlaClient:
    """Minimal Reactive Desk websocket client compatible with the openpi sender."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/ws/VlaIngest",
        *,
        enabled: bool = True,
    ) -> None:
        self._uri = self._build_ws_uri(host, port, path)
        self._enabled = enabled
        self._ws: websockets.sync.client.ClientConnection | None = None

    def connect(self) -> None:
        if not self._enabled or self._ws is not None:
            return

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
        )
        self._ws.recv()

    def send_predictions(
        self,
        xyz: np.ndarray,
        probabilities: np.ndarray,
        *,
        prompt: str = "",
        is_executing: bool = True,
    ) -> bool:
        if not self._enabled:
            return False

        xyz = np.asarray(xyz, dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        predictions = [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if point.shape[0] > 2 else 0.0,
                "probability": float(probabilities[rank])
                if rank < probabilities.shape[0]
                else 1.0,
                "rank": int(rank),
            }
            for rank, point in enumerate(xyz)
        ]
        payload = {
            "type": "vla_predictions",
            "predictions": predictions,
            "is_executing": is_executing,
            "current_prompt": prompt,
        }

        try:
            self.connect()
            if self._ws is None:
                return False
            self._ws.send(json.dumps(payload))
            self._ws.recv()
            return True
        except websockets.ConnectionClosed:
            self._ws = None
            return False
        except OSError:
            self._ws = None
            return False

    def close(self) -> None:
        if self._ws is None:
            return
        self._ws.close()
        self._ws = None

    @staticmethod
    def _build_ws_uri(host: str, port: int, path: str) -> str:
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        if path:
            path = path if path.startswith("/") else f"/{path}"
            if not uri.endswith(path):
                uri = f"{uri.rstrip('/')}{path}"
        return uri


def main() -> None:
    args = _parse_args()
    config_file = config_path(stage="collect", robot=args.robot, explicit=args.config)
    config = _load_config(config_file)
    sys.setswitchinterval(0.0005)

    enable_logging = config.enable_logging
    if enable_logging and _load_or_create_dataset is None:
        raise RuntimeError(
            "Recording is enabled, but LeRobot cannot be imported in this "
            "Python environment. Run with the rebuilt franka_humble uv image."
        ) from _lerobot_import_error
    control_mode = parse_control_mode(getattr(config, "control_mode", "ee"))
    action_space = parse_action_space(getattr(config, "action_space", "ee"))
    gripper_type = parse_gripper_type(getattr(config, "gripper_type", "franka"))
    enable_soft_gripper = gripper_type == "dh5"
    use_force_dataset = bool(enable_logging and enable_soft_gripper)

    label = str(config.label)

    print("=" * 70)
    print(f"Franka LeRobot data collection (ROS 2 VR teleop)  config={config_file}")
    print("=" * 70)
    print(f"control_mode: {control_mode}  action_space: {action_space}  gripper_type: {gripper_type}")
    trans_limit_str = (
        "off"
        if config.max_ee_translation <= 0
        else f"±{config.max_ee_translation:.3f}m"
    )
    rot_limit_str = (
        "off" if config.max_ee_rotation <= 0 else f"±{config.max_ee_rotation:.3f}rad"
    )
    print(f"Control frequency: {config.control_frequency} Hz")
    print(f"Sensitivity: {config.sensitivity}")
    print(
        "EE control tuning: "
        f"step_xyz={config.max_ee_translation_step:.3f}m, "
        f"step_rot={config.max_ee_rotation_step:.3f}rad, "
        f"limit_xyz={trans_limit_str}, "
        f"limit_rot={rot_limit_str}, "
        f"vr_rot={'on' if config.vr_enable_rotation else 'off'}, "
        f"filter={config.ee_filter_coeff:.2f}, "
        f"nullspace={config.ee_nullspace_stiffness:.2f}"
    )
    action_label = (
        "6D EE pose + gripper (7D)"
        if action_space == "ee"
        else "7D joint position + gripper (8D)"
    )
    print(f"Action logging: next measured {action_label}")
    print(
        f"Camera stream: {config.camera_width}x{config.camera_height}@{config.camera_fps} "
        f"(depth={'off' if config.color_only else 'on'})"
    )
    print(f"VR pose topic: {config.vr_pose_topic}")
    if gripper_type == "dh5":
        print(
            "Gripper: DH5 ROS "
            f"left={getattr(config, 'gripper_left_image_topic', '/gripper_left/image_raw')} "
            f"right={getattr(config, 'gripper_right_image_topic', '/gripper_right/image_raw')} "
            f"force={getattr(config, 'soft_gripper_force', 50)} "
            f"velocity={getattr(config, 'soft_gripper_velocity', 100)}"
        )
    elif gripper_type == "franka":
        print("Gripper: Franka parallel gripper")
    else:
        print("Gripper: none")
    print(
        "Reactive Desk: "
        + (
            f"{config.reactive_desk_host}:{config.reactive_desk_port}{config.reactive_desk_path}"
            if config.reactive_desk_enabled
            else "disabled"
        )
    )
    print(f"External camera serial: {config.external_camera_serial}")
    print(f"Wrist camera serial: {config.wrist_camera_serial}")
    if enable_logging:
        config.repo_id = f"{config.repo_id}_{config.date}"
        print(f"Logging: enabled (max {config.max_duration_s} s)")
        print(f"Instruction: {config.instruction}")
        print(f"Label: {label}")
        print(f"LeRobot repo_id: {config.repo_id}")
    else:
        print("Logging: disabled")
    print("=" * 70)

    print("Starting VR ROS pose reader...")
    vr_reader = VRInputRos(
        pose_topic=getattr(config, "vr_pose_topic", "/xr/controller_right/pose"),
        left_joy_topic=getattr(config, "vr_left_joy_topic", "/xr/controller_left/joy"),
        right_joy_topic=getattr(
            config, "vr_right_joy_topic", "/xr/controller_right/joy"
        ),
        long_press_s=config.vr_long_press_s,
        assume_arm_enabled=bool(getattr(config, "vr_assume_arm_enabled", False)),
    )
    vr_reader.start()

    max_ee_translation = float(config.max_ee_translation)
    max_ee_rotation = float(config.max_ee_rotation)
    max_ee_translation_step = float(config.max_ee_translation_step)
    max_ee_rotation_step = float(config.max_ee_rotation_step)

    vr_mapper = VREEPoseMapper(
        translation_scale=config.vr_translation_scale,
        rotation_scale=config.vr_rotation_scale,
        max_translation_step=max_ee_translation_step,
        max_rotation_step=max_ee_rotation_step,
        translation_limit=max_ee_translation,
        rotation_limit=max_ee_rotation,
        sensitivity=config.sensitivity,
        enable_rotation=config.vr_enable_rotation,
    )

    print("Initializing Franka ROS 2 arm adapter...")
    arm = RoboticArmControlerRos(
        use_fake_hardware=bool(getattr(config, "use_fake_hardware", True)),
        robot_type=str(getattr(config, "robot_type", "fr3")),
        start_joint_position=config.start_joint_position,
    )

    camera_backend = str(getattr(config, "camera_backend", "ros")).lower()
    print(f"Initializing ROS cameras (backend={camera_backend})...")
    camera_manager = DualRealsenseManagerRos(
        external_topic=str(
            getattr(config, "external_image_topic", "/external/color/image_raw")
        ),
        wrist_topic=str(getattr(config, "wrist_image_topic", "/wrist/color/image_raw")),
        external_serial=config.external_camera_serial,
        wrist_serial=config.wrist_camera_serial,
        crop_scale=float(config.crop_scale),
        out_hw=int(config.image_hw),
        refresh_hz=float(config.camera_fps),
        enabled=camera_backend != "none",
    )
    camera_manager.connect()

    soft_gripper = None
    if enable_soft_gripper:
        print("Initializing DH5 ROS gripper client...")
        soft_gripper = DH5GripperRos(
            left_image_topic=str(
                getattr(config, "gripper_left_image_topic", "/gripper_left/image_raw")
            ),
            right_image_topic=str(
                getattr(config, "gripper_right_image_topic", "/gripper_right/image_raw")
            ),
            enable_cameras=bool(getattr(config, "enable_soft_gripper_cameras", True)),
            crop_scale=float(config.crop_scale),
            image_hw=int(config.image_hw),
            vr_axis_threshold=float(
                getattr(config, "soft_gripper_axis_threshold", 0.55)
            ),
            vr_step_interval_s=float(
                getattr(config, "soft_gripper_step_interval", 0.08)
            ),
            use_fake=bool(getattr(config, "soft_gripper_use_fake", False)),
        )
        soft_gripper.set_force(int(getattr(config, "soft_gripper_force", 50)))
        soft_gripper.set_velocity(int(getattr(config, "soft_gripper_velocity", 100)))

    dataset = None
    dataset_root: Path | None = None
    if enable_logging:
        dataset_root = Path(__file__).resolve().parent / "data" / config.repo_id
        resume_existing = dataset_root.exists()
        dataset_loader = (
            _load_or_create_dataset_force
            if use_force_dataset
            else _load_or_create_dataset
        )
        if use_force_dataset and _load_or_create_dataset_force is None:
            raise RuntimeError("Force dataset loader is unavailable")
        dataset = dataset_loader(
            config.repo_id,
            fps=config.camera_fps,
            image_hw=config.image_hw,
            root=dataset_root,
            action_space=action_space,
        )
        if resume_existing:
            print(f"LeRobot dataset exists, resuming: {dataset_root}")
        else:
            print(f"LeRobot dataset path: {dataset_root}")
        if use_force_dataset:
            print(
                "[Recording] Using DH5 force dataset features (gripper left/right images)"
            )

    frame_writer = AsyncDatasetFrames(dataset) if dataset is not None else None

    if camera_backend != "none":
        print("[CameraROS] Waiting for first frames...")
        camera_manager.wait_for_frames(timeout_s=float(config.camera_startup_timeout_s))
        print("[CameraROS] First frames acquired, ready to record!")
    else:
        print("[CameraROS] camera_backend=none, skipping wait_for_frames")

    if soft_gripper is not None and bool(
        getattr(config, "enable_soft_gripper_cameras", True)
    ):
        print("[DH5ROS] Waiting for gripper camera frames...")
        soft_gripper.wait_for_frames(timeout_s=float(config.camera_startup_timeout_s))
        print("[DH5ROS] Gripper cameras ready")

    def _current_qpos() -> np.ndarray:
        return np.asarray(arm.state["joint_positions"], dtype=np.float64)

    def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
        return np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )

    def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
        return np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )

    def _quat_angle_xyzw(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a / max(np.linalg.norm(a), 1e-12)
        b = b / max(np.linalg.norm(b), 1e-12)
        dot = abs(float(np.dot(a, b)))
        return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    def _current_ee_pose() -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(arm.ee_pose_matrix, dtype=np.float64)
        pos = matrix[:3, 3]
        rot = matrix[:3, :3]
        quat_xyzw = _matrix_to_quat_xyzw(rot)
        return pos.astype(np.float64), quat_xyzw

    def _matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
        rotation = np.asarray(rotation, dtype=np.float64)
        t = float(np.trace(rotation))
        if t > 0.0:
            r = np.sqrt(1.0 + t)
            w = 0.5 * r
            s = 0.5 / r
            x = (rotation[2, 1] - rotation[1, 2]) * s
            y = (rotation[0, 2] - rotation[2, 0]) * s
            z = (rotation[1, 0] - rotation[0, 1]) * s
        else:
            i = int(np.argmax([rotation[0, 0], rotation[1, 1], rotation[2, 2]]))
            j, k = (i + 1) % 3, (i + 2) % 3
            r = np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k])
            vals = [0.0, 0.0, 0.0]
            vals[i] = 0.5 * r
            s = 0.5 / r
            w = (rotation[k, j] - rotation[j, k]) * s
            vals[j] = (rotation[j, i] + rotation[i, j]) * s
            vals[k] = (rotation[k, i] + rotation[i, k]) * s
            x, y, z = vals
        return np.array([x, y, z, w], dtype=np.float64)

    def _hold_current_ee_pose() -> None:
        ee_pos, ee_quat_xyzw = _current_ee_pose()
        qpos = _current_qpos()
        arm.set_ee_control(ee_pos, ee_quat_xyzw, qpos)

    def _start_ee_controller(settle_s: float = 0.0):
        if control_mode == "ee":
            arm.start_ee_streaming(settle_s=settle_s)
        else:
            arm.start_joint_streaming(settle_s=settle_s)
        _hold_current_ee_pose()

    def _format_joint_position(qpos: np.ndarray) -> str:
        return np.array2string(
            np.asarray(qpos, dtype=np.float64),
            precision=5,
            separator=", ",
            suppress_small=False,
        )

    def _print_current_joint_position(qpos: np.ndarray) -> None:
        line = f"[Joint] current q = {_format_joint_position(qpos)}"
        print(f"{line:<140}", end="\r", flush=True)

    def _move_robot_to_joint_pose(
        target_qpos: np.ndarray | list[float] | tuple[float, ...],
    ) -> None:
        qpos = np.asarray(target_qpos, dtype=np.float64).flatten()
        if qpos.shape != (7,):
            raise ValueError(f"target_qpos must be 7D, got shape {qpos.shape}")
        print(f"[Control] Moving to custom joint pose: {_format_joint_position(qpos)}")
        arm.move_to_joint_position(qpos)

    def _move_robot_to_start_pose() -> None:
        move_name = "move_to_start"
        if config.start_joint_position is None:
            arm.move_to_start()
        else:
            move_name = "custom start joint pose"
            _move_robot_to_joint_pose(config.start_joint_position)
        if not arm.wait_until_stopped():
            max_vel = float(np.max(np.abs(np.asarray(arm.state["joint_velocities"]))))
            print(
                f"[Warning] Robot did not fully stop after {move_name}: "
                f"max_vel={max_vel:.4f} rad/s"
            )

    print("Opening gripper...")
    if soft_gripper is not None:
        soft_gripper.set_gripper_level(0, wait=True)
    elif gripper_type == "franka":
        arm.gripper_open()
    else:
        print("[Gripper] gripper_type=none, skip open")
    print("Moving to start position...")
    _move_robot_to_start_pose()
    _start_ee_controller(settle_s=0.5)
    hold_ee_pos_target, hold_ee_quat_xyzw_target = _current_ee_pose()
    hold_qpos_target = _current_qpos().copy()

    def _refresh_hold_target_from_current() -> None:
        nonlocal hold_ee_pos_target, hold_ee_quat_xyzw_target, hold_qpos_target
        hold_ee_pos_target, hold_ee_quat_xyzw_target = _current_ee_pose()
        hold_qpos_target = _current_qpos().copy()

    def _set_hold_control() -> None:
        arm.set_ee_control(
            hold_ee_pos_target,
            hold_ee_quat_xyzw_target,
            hold_qpos_target,
        )

    active_instruction = config.instruction
    reactive_desk_client = ReactiveDeskVlaClient(
        host=config.reactive_desk_host,
        port=config.reactive_desk_port,
        path=config.reactive_desk_path,
        enabled=config.reactive_desk_enabled,
    )
    reactive_stop = threading.Event()
    reactive_lock = threading.Lock()
    reactive_latest_xy: np.ndarray | None = None

    def _reactive_worker() -> None:
        while not reactive_stop.is_set():
            with reactive_lock:
                point = reactive_latest_xy
            if point is not None:
                reactive_desk_client.send_predictions(
                    point.reshape(1, 2),
                    np.ones(1, dtype=np.float32),
                    prompt=active_instruction,
                    is_executing=True,
                )
            reactive_stop.wait(1.0 / 30.0)

    reactive_thread = None
    if config.reactive_desk_enabled:
        reactive_thread = threading.Thread(
            target=_reactive_worker, name="reactive-desk", daemon=True
        )
        reactive_thread.start()
    gripper_state = 1.0
    last_gripper_cmd = 1.0
    if soft_gripper is not None:
        gripper_state = float(soft_gripper.gripper_open_ratio.reshape(-1)[0])
        last_gripper_cmd = gripper_state

    recording_started = False
    recording_started_at: float | None = None
    episode_start_monotonic_ns: int | None = None
    current_episode_index: int | None = None
    current_episode_success: bool | None = None
    frame_records: list[dict] = []
    pending_frame: dict | None = None
    pending_action: dict | None = None
    action_trace_file = None
    action_trace_tmp_path: Path | None = None
    action_count = 0
    first_action_sample_ns = 0
    last_action_sample_ns = 0
    last_recorded_vr_seq = 0
    unique_vr_samples = 0
    camera_pair_gate = FreshCameraPair()
    frame_count = 0
    motion_start_threshold = max(
        float(config.action_epsilon),
        float(getattr(config, "motion_start_threshold", 0.0002)),
    )
    last_gripper_switch_time = 0.0
    gripper_busy = False
    gripper_switch_cooldown_s = 0.12
    reflex_error_occurred = False
    prev_y_pressed = False
    prev_x_pressed = False
    prev_arm_enabled = False
    control_tick = 0

    def _reset_episode_state() -> None:
        nonlocal frame_count
        nonlocal recording_started
        nonlocal recording_started_at
        nonlocal episode_start_monotonic_ns
        nonlocal current_episode_index
        nonlocal current_episode_success
        nonlocal frame_records
        nonlocal pending_frame
        nonlocal pending_action, action_trace_file, action_trace_tmp_path
        nonlocal action_count
        nonlocal first_action_sample_ns, last_action_sample_ns
        nonlocal last_recorded_vr_seq, unique_vr_samples

        frame_count = 0
        recording_started = False
        recording_started_at = None
        episode_start_monotonic_ns = None
        current_episode_index = None
        current_episode_success = None
        frame_records = []
        pending_frame = None
        pending_action = None
        action_trace_file = None
        action_trace_tmp_path = None
        action_count = 0
        first_action_sample_ns = 0
        last_action_sample_ns = 0
        last_recorded_vr_seq = 0
        unique_vr_samples = 0
        camera_pair_gate.reset()

    def _append_pending_action(
        next_qpos: np.ndarray, next_ee_pose: np.ndarray, next_gripper: float
    ) -> None:
        nonlocal pending_action, action_count
        nonlocal first_action_sample_ns, last_action_sample_ns
        nonlocal last_recorded_vr_seq, unique_vr_samples
        if pending_action is None or action_trace_file is None:
            return
        pending_action["action_joint_position"] = JointPose(next_qpos).vector.tolist()
        pending_action["action_ee_pose"] = EEPose.from_vector(next_ee_pose).vector.tolist()
        pending_action["action_gripper_position"] = float(next_gripper)
        action_trace_file.write(
            json.dumps(pending_action, separators=(",", ":")) + "\n"
        )
        sample_ns = int(pending_action["host_sample_monotonic_ns"])
        if first_action_sample_ns == 0:
            first_action_sample_ns = sample_ns
        last_action_sample_ns = sample_ns
        vr_seq = int(pending_action["vr_pose_seq"])
        if vr_seq > 0 and vr_seq != last_recorded_vr_seq:
            unique_vr_samples += 1
            last_recorded_vr_seq = vr_seq
        action_count += 1
        pending_action = None

    def _append_pending_frame(
        action_qpos: np.ndarray, action_ee_pose: np.ndarray, action_gripper_state: float
    ) -> None:
        nonlocal frame_count, pending_frame

        if pending_frame is None or dataset is None:
            return

        action_gripper_state = float(action_gripper_state)
        joint = JointPose(action_qpos)
        ee = EEPose.from_vector(action_ee_pose)
        actions = (
            ee.action(action_gripper_state)
            if action_space == "ee"
            else joint.action(action_gripper_state)
        )

        frame_record = pending_frame["frame_record"]
        frame_record["action_joint_position"] = joint.vector.tolist()
        frame_record["action_ee_pose"] = ee.vector.tolist()
        frame_record["action_gripper_position"] = action_gripper_state

        frame_payload = {
            "exterior_image_1_left": pending_frame["external_img"],
            "exterior_image_2_left": pending_frame["blank"],
            "wrist_image_left": pending_frame["wrist_img"],
            "joint_position": pending_frame["joint_pos"],
            "ee_pose": pending_frame["ee_pose"],
            "gripper_position": pending_frame["gripper_pos"],
            "actions": actions,
            "task": active_instruction,
        }
        if use_force_dataset:
            frame_payload["gripper_image_left"] = pending_frame["gripper_left_img"]
            frame_payload["gripper_image_right"] = pending_frame["gripper_right_img"]
            frame_payload["external_camera_timestamp_ms"] = np.asarray(
                [frame_record["external_camera_timestamp"] * 1000.0], dtype=np.float32
            )
            frame_payload["wrist_camera_timestamp_ms"] = np.asarray(
                [frame_record["wrist_camera_timestamp"] * 1000.0], dtype=np.float32
            )
            frame_ns = int(frame_record["host_frame_monotonic_ns"])
            frame_payload["external_camera_frame_age_s"] = np.asarray(
                [
                    max(
                        0, frame_ns - frame_record["external_host_capture_monotonic_ns"]
                    )
                    * 1e-9
                ],
                dtype=np.float32,
            )
            frame_payload["wrist_camera_frame_age_s"] = np.asarray(
                [
                    max(0, frame_ns - frame_record["wrist_host_capture_monotonic_ns"])
                    * 1e-9
                ],
                dtype=np.float32,
            )
        if frame_writer is None:
            raise RuntimeError("LeRobot frame writer is unavailable")
        frame_writer.submit(frame_payload)
        frame_records.append(frame_record)
        frame_count += 1
        pending_frame = None
        if frame_count % 50 == 0:
            print(f"[Recording] {frame_count} frames", end="\r")

    def _cleanup_episode_files(*paths: Path | None) -> None:
        for path in paths:
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _write_episode_sync_json() -> Path:
        if dataset_root is None or current_episode_index is None:
            raise RuntimeError(
                "Episode sync JSON cannot be written without logging state"
            )

        sync_path = dataset_root / f"episode_{current_episode_index:06d}.sync.json"
        payload = {
            "episode_index": current_episode_index,
            "task": active_instruction,
            "label": label,
            "success": current_episode_success,
            "divergence_time": None,
            "control_frequency": float(config.control_frequency),
            "camera_fps": float(config.camera_fps),
            "action_space": action_space,
            "action_target_offset_frames": 1,
            "action_frequency": float(config.control_frequency),
            "action_records": action_count,
            "action_effective_hz": (
                (action_count - 1)
                * 1e9
                / (last_action_sample_ns - first_action_sample_ns)
                if action_count > 1 and last_action_sample_ns > first_action_sample_ns
                else 0.0
            ),
            "vr_unique_pose_samples": unique_vr_samples,
            "vr_reused_pose_samples": action_count - unique_vr_samples,
            "action_trace": f"episode_{current_episode_index:06d}.actions.jsonl",
            "episode_start_monotonic_ns": episode_start_monotonic_ns,
            "video_frames": frame_count,
            "camera_pair_effective_hz": (
                (frame_count - 1)
                * 1e9
                / (
                    frame_records[-1]["host_frame_monotonic_ns"]
                    - frame_records[0]["host_frame_monotonic_ns"]
                )
                if frame_count > 1
                and frame_records[-1]["host_frame_monotonic_ns"]
                > frame_records[0]["host_frame_monotonic_ns"]
                else 0.0
            ),
            "frame_records": frame_records,
            "camera_serials": {
                "external": camera_manager.external_camera.serial,
                "wrist": camera_manager.wrist_camera.serial,
            },
        }
        sync_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return sync_path

    def _finalize_episode_data(*, save_episode: bool) -> None:
        nonlocal frame_count

        try:
            if save_episode and (pending_frame is not None or pending_action is not None):
                final_qpos = _current_qpos()
                final_pos, final_quat = _current_ee_pose()
                final_ee_pose = EEPose.from_position_quat(final_pos, final_quat).vector
                if pending_frame is not None:
                    _append_pending_frame(final_qpos, final_ee_pose, gripper_state)
                if pending_action is not None:
                    _append_pending_action(final_qpos, final_ee_pose, gripper_state)
            if action_trace_file is not None:
                action_trace_file.close()
            if frame_writer is not None:
                frame_writer.drain()
        except Exception:
            if action_trace_file is not None and not action_trace_file.closed:
                action_trace_file.close()
            _cleanup_episode_files(action_trace_tmp_path)
            if dataset is not None:
                _discard_unsaved_episode(dataset)
            _reset_episode_state()
            raise

        should_persist = (
            bool(save_episode)
            and frame_count > 0
            and action_count > 0
            and dataset is not None
            and current_episode_index is not None
        )

        sync_path: Path | None = None
        action_trace_path: Path | None = None
        if not should_persist:
            _cleanup_episode_files(action_trace_tmp_path)
            if dataset is not None:
                try:
                    _discard_unsaved_episode(dataset)
                except Exception:
                    pass
            _reset_episode_state()
            return

        try:
            if dataset_root is None or action_trace_tmp_path is None:
                raise RuntimeError("Action trace was not initialized")
            action_trace_path = (
                dataset_root / f"episode_{current_episode_index:06d}.actions.jsonl"
            )
            action_trace_tmp_path.replace(action_trace_path)
            sync_path = _write_episode_sync_json()
            _prepare_episode_for_save_force(
                dataset
            ) if use_force_dataset else _prepare_episode_for_save(dataset)
            dataset.save_episode()
            print(
                f"[Recording] Saved episode {current_episode_index:06d}: "
                f"{action_count} action samples, {frame_count} camera frames"
            )
            print(f"[Recording] Saved sync metadata: {sync_path}")
        except Exception as exc:
            print(f"[Error] Failed to save episode: {exc}")
            _cleanup_episode_files(sync_path, action_trace_tmp_path, action_trace_path)
            try:
                _discard_unsaved_episode(dataset)
            except Exception:
                pass
            raise
        finally:
            _reset_episode_state()

    def _reset_robot_to_start() -> None:
        nonlocal gripper_state, last_gripper_cmd

        _set_hold_control()
        arm.stop_ee_streaming()
        if soft_gripper is not None:
            soft_gripper.set_gripper_level(0, wait=True)
            gripper_state = float(soft_gripper.gripper_open_ratio.reshape(-1)[0])
        elif gripper_type == "franka":
            arm.gripper_open()
            gripper_state = 1.0
        else:
            gripper_state = 1.0
        last_gripper_cmd = 1.0
        print("[Control] Moving to start position...")
        _move_robot_to_start_pose()
        _start_ee_controller(settle_s=0.0)
        _refresh_hold_target_from_current()
        _set_hold_control()
        vr_mapper.reset()

    def _finish_episode(
        *, save_episode: bool, message: str, success: bool | None = None
    ) -> None:
        nonlocal current_episode_success
        print(message)
        current_episode_success = success
        _finalize_episode_data(save_episode=save_episode)
        _reset_robot_to_start()

    def _send_current_ee_xy(ee_pos: np.ndarray) -> None:
        nonlocal reactive_latest_xy
        if not config.reactive_desk_enabled:
            return
        with reactive_lock:
            reactive_latest_xy = np.asarray(ee_pos[:2], dtype=np.float32).copy()

    print("\nControl mapping (VR):")
    print("  Hold both triggers (long press): enable arm movement")
    print("  Right controller pose: EE pose target -> Cartesian impedance control")
    print(f"  Action: next measured {action_label}")
    print("  Release / re-hold triggers: re-anchor the VR neutral pose")
    print("  A (right): gripper close | B (right): gripper open")
    if soft_gripper is not None:
        print("  Right thumbstick Y: DH5 level step (up=close, down=open)")
    print("  Recording: starts automatically when motion begins")
    print("  Y (left): save current episode and return to start")
    print("  X (left): save current episode labelled failed and return to start")

    try:
        with arm.control_loop(frequency=config.control_frequency) as ctx:
            while ctx.ok():
                if (
                    enable_logging
                    and recording_started
                    and recording_started_at is not None
                    and (time.time() - recording_started_at) > config.max_duration_s
                ):
                    if gripper_busy:
                        continue
                    _finish_episode(
                        save_episode=frame_count > 0 or pending_frame is not None,
                        message="\n[Recording] Max duration reached, saving episode and returning to start...",
                    )
                    continue

                vr = vr_reader.latest

                y_pressed = bool(getattr(vr, "y_pressed", False))
                y_edge = y_pressed and not prev_y_pressed
                prev_y_pressed = y_pressed

                x_pressed = bool(getattr(vr, "x_pressed", False))
                x_edge = x_pressed and not prev_x_pressed
                prev_x_pressed = x_pressed

                if y_edge:
                    if not enable_logging:
                        print(
                            "\n[Recording] Ignored Y press because logging is disabled."
                        )
                    elif gripper_busy:
                        print("\n[Recording] Ignored Y press because gripper is busy.")
                    else:
                        _finish_episode(
                            save_episode=frame_count > 0 or pending_frame is not None,
                            message=(
                                "\n[Control] Y pressed, saving episode and returning to start..."
                                if frame_count > 0 or pending_frame is not None
                                else "\n[Control] Y pressed, resetting with no captured frames."
                            ),
                            success=True,
                        )
                    continue

                if x_edge:
                    if not enable_logging:
                        print(
                            "\n[Recording] Ignored X press because logging is disabled."
                        )
                    elif gripper_busy:
                        print("\n[Recording] Ignored X press because gripper is busy.")
                    else:
                        _finish_episode(
                            save_episode=frame_count > 0 or pending_frame is not None,
                            message=(
                                "\n[Control] X pressed, saving episode as failed and returning to start..."
                                if frame_count > 0 or pending_frame is not None
                                else "\n[Control] X pressed, resetting with no captured frames."
                            ),
                            success=False,
                        )
                    continue

                qpos = _current_qpos()
                sample_ns = time.monotonic_ns()
                robot_state = arm.state
                control_tick += 1
                if control_tick % 10 == 0:
                    _print_current_joint_position(qpos)
                ee_pos, ee_quat_xyzw = _current_ee_pose()
                ee_pose6 = EEPose.from_position_quat(ee_pos, ee_quat_xyzw).vector
                if enable_logging and recording_started and pending_action is not None:
                    _append_pending_action(qpos, ee_pose6, gripper_state)
                _send_current_ee_xy(ee_pos)

                arm_enabled = bool(vr.arm_enabled)
                if not arm_enabled and prev_arm_enabled:
                    _refresh_hold_target_from_current()
                    vr_mapper.reset()
                elif not arm_enabled:
                    vr_mapper.reset()
                prev_arm_enabled = arm_enabled

                if arm_enabled:
                    hold_ee_quat_wxyz = _quat_xyzw_to_wxyz(hold_ee_quat_xyzw_target)
                    target_ee_pos, target_ee_quat_wxyz = vr_mapper.map(
                        vr,
                        hold_ee_pos_target,
                        hold_ee_quat_wxyz,
                    )
                    target_ee_quat_xyzw = _quat_wxyz_to_xyzw(target_ee_quat_wxyz)
                else:
                    target_ee_pos = hold_ee_pos_target.copy()
                    target_ee_quat_xyzw = hold_ee_quat_xyzw_target.copy()

                command_translation = target_ee_pos - hold_ee_pos_target
                rotation_error = _quat_angle_xyzw(
                    target_ee_quat_xyzw, hold_ee_quat_xyzw_target
                )
                motion_norm = float(np.linalg.norm(command_translation))
                rotation_motion_threshold = max(float(config.action_epsilon), 1e-3)
                has_ee_motion_cmd = bool(
                    arm_enabled
                    and not gripper_busy
                    and (
                        motion_norm >= motion_start_threshold
                        or rotation_error >= rotation_motion_threshold
                    )
                )

                if not gripper_busy:
                    if has_ee_motion_cmd:
                        arm.set_ee_control(target_ee_pos, target_ee_quat_xyzw, qpos)
                        hold_ee_pos_target = target_ee_pos.copy()
                        hold_ee_quat_xyzw_target = target_ee_quat_xyzw.copy()
                        hold_qpos_target = qpos.copy()
                    else:
                        _set_hold_control()

                gripper_changed = False
                if soft_gripper is not None:
                    if vr.gripper_close:
                        gripper_changed = soft_gripper.set_gripper_level(
                            soft_gripper.max_gripper_level,
                            wait=False,
                        )
                    elif vr.gripper_open:
                        gripper_changed = soft_gripper.set_gripper_level(0, wait=False)
                    else:
                        gripper_changed = soft_gripper.update_from_vr(vr, wait=False)
                    gripper_state = float(
                        soft_gripper.gripper_open_ratio.reshape(-1)[0]
                    )
                    last_gripper_cmd = gripper_state
                    gripper_busy = False
                elif gripper_type == "franka":
                    gripper_cmd = last_gripper_cmd
                    if vr.gripper_close:
                        gripper_cmd = 0.0
                    elif vr.gripper_open:
                        gripper_cmd = 1.0

                    now = time.time()
                    if now - last_gripper_switch_time < gripper_switch_cooldown_s:
                        gripper_cmd = last_gripper_cmd

                    if gripper_cmd != last_gripper_cmd and not gripper_busy:
                        gripper_changed = True
                        last_gripper_switch_time = now
                        gripper_state = 1.0 if gripper_cmd > 0.5 else 0.0
                        last_gripper_cmd = gripper_cmd
                        gripper_busy = True

                        def _do_gripper(cmd):
                            nonlocal gripper_busy
                            try:
                                _set_hold_control()
                                if cmd > 0.5:
                                    arm.gripper_open()
                                else:
                                    arm.gripper_close()
                                _refresh_hold_target_from_current()
                                _set_hold_control()
                                vr_mapper.reset()
                            finally:
                                gripper_busy = False

                        threading.Thread(
                            target=_do_gripper, args=(gripper_cmd,), daemon=True
                        ).start()
                else:
                    gripper_busy = False

                if not enable_logging:
                    continue

                has_action = has_ee_motion_cmd or gripper_changed
                if has_action and not recording_started:
                    if dataset is None or dataset_root is None:
                        raise RuntimeError("Logging state is not initialized")
                    current_episode_index = int(dataset.episode_buffer["episode_index"])
                    episode_start_monotonic_ns = time.monotonic_ns()
                    recording_started = True
                    recording_started_at = time.time()
                    frame_records = []
                    action_trace_tmp_path = (
                        dataset_root
                        / f"episode_{current_episode_index:06d}.actions.jsonl.tmp"
                    )
                    action_trace_file = action_trace_tmp_path.open(
                        "w", encoding="utf-8"
                    )
                    print(
                        "\n[Recording] First motion detected, start logging "
                        f"for episode {current_episode_index:06d} with prompt: {active_instruction}"
                    )

                if not recording_started:
                    continue

                pending_action = {
                    "sample_index": action_count,
                    "host_sample_monotonic_ns": sample_ns,
                    "joint_position": np.asarray(qpos, dtype=np.float32).tolist(),
                    "gripper_position": float(gripper_state),
                    "ee_position": np.asarray(ee_pos, dtype=np.float32).tolist(),
                    "ee_orientation_xyzw": np.asarray(
                        ee_quat_xyzw, dtype=np.float32
                    ).tolist(),
                    "ee_pose": np.asarray(ee_pose6, dtype=np.float32).tolist(),
                    "target_ee_position": np.asarray(
                        hold_ee_pos_target, dtype=np.float32
                    ).tolist(),
                    "target_ee_orientation_xyzw": np.asarray(
                        hold_ee_quat_xyzw_target, dtype=np.float32
                    ).tolist(),
                    "action_gripper_position": float(gripper_state),
                    "vr_pose_seq": int(vr.pose_seq),
                    "vr_pose_monotonic_ns": int(vr.pose_monotonic_ns),
                    "vr_position": [float(vr.pos_x), float(vr.pos_y), float(vr.pos_z)],
                    "vr_orientation_xyzw": [
                        float(vr.quat_x),
                        float(vr.quat_y),
                        float(vr.quat_z),
                        float(vr.quat_w),
                    ],
                    "vr_arm_enabled": bool(vr.arm_enabled),
                }

                external_img, wrist_img, external_ts, wrist_ts = (
                    camera_manager.get_frames()
                )
                if (
                    external_img is None
                    or wrist_img is None
                    or external_ts is None
                    or wrist_ts is None
                ):
                    continue

                if external_img.shape != (config.image_hw, config.image_hw, 3):
                    continue
                if wrist_img.shape != (config.image_hw, config.image_hw, 3):
                    continue
                if not camera_pair_gate.accept(
                    external_ts.host_capture_monotonic_ns,
                    wrist_ts.host_capture_monotonic_ns,
                ):
                    continue

                gripper_left_img = None
                gripper_right_img = None
                if use_force_dataset:
                    if soft_gripper is None:
                        continue
                    if bool(getattr(config, "enable_soft_gripper_cameras", True)):
                        right_img, left_img, _, _ = (
                            soft_gripper.dual_camera_manager.get_frames()
                        )
                        gripper_left_img = left_img
                        gripper_right_img = right_img
                        if gripper_left_img is None or gripper_right_img is None:
                            continue
                        if gripper_left_img.shape != (
                            config.image_hw,
                            config.image_hw,
                            3,
                        ):
                            continue
                        if gripper_right_img.shape != (
                            config.image_hw,
                            config.image_hw,
                            3,
                        ):
                            continue
                    else:
                        blank_gripper = np.zeros(
                            (int(config.image_hw), int(config.image_hw), 3),
                            dtype=np.uint8,
                        )
                        gripper_left_img = blank_gripper
                        gripper_right_img = blank_gripper

                if pending_frame is not None:
                    _append_pending_frame(qpos, ee_pose6, gripper_state)

                if recording_started:
                    joint_pos = np.asarray(
                        robot_state["joint_positions"], dtype=np.float32
                    )
                    gripper_pos = np.asarray(
                        [np.float32(gripper_state)], dtype=np.float32
                    )
                    blank = np.zeros_like(external_img)

                    frame_record = {
                        "frame_index": frame_count,
                        "host_frame_monotonic_ns": time.monotonic_ns(),
                        "external_camera_timestamp": float(
                            external_ts.camera_timestamp
                        ),
                        "wrist_camera_timestamp": float(wrist_ts.camera_timestamp),
                        "external_host_capture_monotonic_ns": int(
                            external_ts.host_capture_monotonic_ns
                        ),
                        "wrist_host_capture_monotonic_ns": int(
                            wrist_ts.host_capture_monotonic_ns
                        ),
                        "joint_position": joint_pos.tolist(),
                        "gripper_position": float(gripper_pos[0]),
                        "ee_position": np.asarray(ee_pos, dtype=np.float32).tolist(),
                        "ee_orientation_xyzw": np.asarray(
                            ee_quat_xyzw, dtype=np.float32
                        ).tolist(),
                        "ee_pose": np.asarray(ee_pose6, dtype=np.float32).tolist(),
                    }

                    pending_frame = {
                        "external_img": external_img,
                        "wrist_img": wrist_img,
                        "blank": blank,
                        "joint_pos": joint_pos,
                        "ee_pose": np.asarray(ee_pose6, dtype=np.float32),
                        "gripper_pos": gripper_pos,
                        "gripper_left_img": gripper_left_img,
                        "gripper_right_img": gripper_right_img,
                        "frame_record": frame_record,
                    }

    except RuntimeError as exc:
        msg = str(exc)
        if "motion aborted by reflex" in msg or "joint_velocity_violation" in msg:
            reflex_error_occurred = True
            print("\n[Error] Franka reflex triggered; aborting teleop safely.")
            print(f"[Error] {msg}")
            print(
                "[Hint] Try smaller control.sensitivity, "
                "control.max_ee_translation_step, or "
                "control.max_ee_rotation_step in the YAML config."
            )
        else:
            raise
    except KeyboardInterrupt:
        print("\n[Recording] Ctrl+C detected, stopping...")
    finally:
        try:
            _set_hold_control()
            arm.stop_ee_streaming()
        except Exception:
            pass

        if enable_logging and not reflex_error_occurred:
            try:
                _finalize_episode_data(
                    save_episode=frame_count > 0 or pending_frame is not None
                )
            except Exception as exc:
                print(f"\n[Error] Failed to finalize current episode: {exc}")
        elif enable_logging and reflex_error_occurred:
            try:
                _finalize_episode_data(save_episode=False)
            except Exception:
                pass
        try:
            reactive_stop.set()
            if reactive_thread is not None:
                reactive_thread.join(timeout=2.0)
            reactive_desk_client.close()
        except Exception:
            pass
        try:
            camera_manager.close()
        except Exception:
            pass
        if soft_gripper is not None:
            try:
                soft_gripper.close()
            except Exception:
                pass

        try:
            vr_reader.stop()
        except Exception:
            pass

        try:
            arm.cleanup()
        except Exception:
            pass

        if enable_logging and dataset is not None:
            if frame_writer is not None:
                try:
                    frame_writer.close()
                except Exception as exc:
                    print(f"[Error] Failed to close LeRobot frame writer: {exc}")
            try:
                dataset.stop_image_writer()
            except Exception:
                pass


if __name__ == "__main__":
    main()
