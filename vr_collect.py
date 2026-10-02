"""ROS 2 collection script for VR demonstrations.

Reads ``config/collect/<robot>.yaml``. Select the dataset format with
``dataset.lerobot_format`` or ``--lerobot-format``::

    python vr_collect.py --lerobot-format v2

Use ``v2`` in the Humble image and ``v3`` in the Jazzy image.

- Cameras: DualRealsenseManagerRos (or camera_backend: none)
- Control: latest VR PoseStamped -> CartesianPoseTargetController at 100 Hz
- EE pose and joint angles are stored in every LeRobot frame
- No microphone / VAD path
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import rclpy

from control.collection_recording import (
    ActionSample,
    CollectionEpisodeRecorder,
    FrameSample,
)
from control.collection_robot import CollectionRobot
from control.dual_camera_manager_ros import DualRealsenseManagerRos
from control.recording_writer import AsyncDatasetFrames
from control.robot_config import (
    config_path,
    flatten_config,
    parse_action_space,
    parse_control_mode,
    parse_gripper_type,
)
from control.robot_state import EEPose
from control.robotic_arm_controller_ros import RoboticArmControlerRos
from control.motion_config import workflow_motion_config
from control.soft_gripper_control_ros import DH5GripperRos
from control.util.lerobot_recording import (
    open_recording_dataset,
    parse_lerobot_format,
)
from control.util.pose import (
    quat_angle_xyzw,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from control.util.timing import control_loop
from control.util.robot import (
    current_ee_pose,
    current_joint_position,
    format_joint_position,
    move_robot_to_start_pose,
    start_control_streaming,
)
from control.vr_input import VREEPoseMapper, VRInputRos


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
    parser.add_argument(
        "--lerobot-format",
        choices=("v2", "v3"),
        default=None,
        help="Dataset format; overrides dataset.lerobot_format in the YAML config",
    )
    return parser.parse_args()


def main(*, lerobot_format: str | None = None) -> None:
    args = _parse_args()
    config_file = config_path(stage="collect", robot=args.robot, explicit=args.config)
    config = flatten_config(config_file)
    sys.setswitchinterval(0.0005)
    if config.reactive_desk_enabled:
        raise RuntimeError("Reactive Desk publishing is unavailable")

    enable_logging = config.enable_logging
    selected_format = parse_lerobot_format(
        args.lerobot_format or lerobot_format or getattr(config, "lerobot_format", "v2")
    )
    control_mode = parse_control_mode(getattr(config, "control_mode", "ee"))
    action_space = parse_action_space(getattr(config, "action_space", "ee"))
    gripper_type = parse_gripper_type(getattr(config, "gripper_type", "franka"))
    motion_config = workflow_motion_config(
        {
            "robot": {
                "robot_type": config.robot_type,
                "use_fake_hardware": config.use_fake_hardware,
            },
            "gripper": {"gripper_type": gripper_type},
            "motion": config.motion,
        },
        control_mode=control_mode,
    )
    enable_soft_gripper = gripper_type == "dh5"
    use_force_dataset = bool(enable_logging and enable_soft_gripper)

    label = str(config.label)

    print("=" * 70)
    print(f"Franka LeRobot data collection (ROS 2 VR teleop)  config={config_file}")
    print(f"LeRobot dataset format: {selected_format}")
    print("=" * 70)
    print(
        f"control_mode: {control_mode}  action_space: {action_space}  gripper_type: {gripper_type}"
    )
    trans_limit_str = (
        "off" if config.max_ee_translation <= 0 else f"±{config.max_ee_translation:.3f}m"
    )
    rot_limit_str = "off" if config.max_ee_rotation <= 0 else f"±{config.max_ee_rotation:.3f}rad"
    print(f"Control frequency: {config.control_frequency} Hz")
    print(f"Sensitivity: {config.sensitivity}")
    print(
        "EE control tuning: "
        f"step_xyz={config.max_ee_translation_step:.3f}m, "
        f"step_rot={config.max_ee_rotation_step:.3f}rad, "
        f"limit_xyz={trans_limit_str}, "
        f"limit_rot={rot_limit_str}, "
        f"vr_rot={'on' if config.vr_enable_rotation else 'off'}"
    )
    action_label = (
        "6D EE pose + gripper (7D)" if action_space == "ee" else "7D joint position + gripper (8D)"
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
        right_joy_topic=getattr(config, "vr_right_joy_topic", "/xr/controller_right/joy"),
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
        robot_type=str(getattr(config, "robot_type", "panda")),
        motion_config=motion_config,
    )

    arm.connect()

    camera_backend = str(getattr(config, "camera_backend", "ros")).lower()
    print(f"Initializing ROS cameras (backend={camera_backend})...")
    camera_manager = DualRealsenseManagerRos(
        external_topic=str(getattr(config, "external_image_topic", "/external/color/image_raw")),
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
            vr_axis_threshold=float(getattr(config, "soft_gripper_axis_threshold", 0.55)),
            vr_step_interval_s=float(getattr(config, "soft_gripper_step_interval", 0.08)),
            use_fake=bool(getattr(config, "soft_gripper_use_fake", False)),
        )
        soft_gripper.set_force(int(getattr(config, "soft_gripper_force", 50)))
        soft_gripper.set_velocity(int(getattr(config, "soft_gripper_velocity", 100)))

    dataset = None
    dataset_root: Path | None = None
    if enable_logging:
        dataset_root = Path(__file__).resolve().parent / "data" / config.repo_id
        resume_existing = dataset_root.exists()
        dataset = open_recording_dataset(
            selected_format,
            config.repo_id,
            fps=config.camera_fps,
            image_hw=config.image_hw,
            root=dataset_root,
            action_space=action_space,
            force=use_force_dataset,
        )
        if resume_existing:
            print(f"LeRobot dataset exists, resuming: {dataset_root}")
        else:
            print(f"LeRobot dataset path: {dataset_root}")
        if use_force_dataset:
            print("[Recording] Using DH5 force dataset features (gripper left/right images)")

    frame_writer = AsyncDatasetFrames(dataset) if dataset is not None else None

    if camera_backend != "none":
        print("[CameraROS] Waiting for first frames...")
        camera_manager.wait_for_frames(timeout_s=float(config.camera_startup_timeout_s))
        print("[CameraROS] First frames acquired, ready to record!")
    else:
        print("[CameraROS] camera_backend=none, skipping wait_for_frames")

    if soft_gripper is not None and bool(getattr(config, "enable_soft_gripper_cameras", True)):
        print("[DH5ROS] Waiting for gripper camera frames...")
        soft_gripper.wait_for_frames(timeout_s=float(config.camera_startup_timeout_s))
        print("[DH5ROS] Gripper cameras ready")

    print("Opening gripper...")
    if soft_gripper is not None:
        soft_gripper.set_gripper_level(0, wait=True)
    elif gripper_type == "franka":
        arm.gripper_open()
    else:
        print("[Gripper] gripper_type=none, skip open")
    print("Moving to start position...")
    move_robot_to_start_pose(arm, config.start_joint_position, motion_config=motion_config)
    start_control_streaming(arm, control_mode, settle_s=0.5)
    collection_robot = CollectionRobot(
        arm=arm,
        vr_mapper=vr_mapper,
        soft_gripper=soft_gripper,
        gripper_type=gripper_type,
        control_mode=control_mode,
        start_joint_position=config.start_joint_position,
        motion_config=motion_config,
    )

    active_instruction = config.instruction

    recorder = CollectionEpisodeRecorder(
        dataset=dataset,
        frame_writer=frame_writer,
        root=dataset_root,
        action_space=action_space,
        force=use_force_dataset,
        task=active_instruction,
        label=label,
        control_frequency=float(config.control_frequency),
        camera_fps=float(config.camera_fps),
        camera_serials={
            "external": camera_manager.external_camera.serial,
            "wrist": camera_manager.wrist_camera.serial,
        },
    )
    motion_start_threshold = max(
        float(config.action_epsilon),
        float(getattr(config, "motion_start_threshold", 0.0002)),
    )
    reflex_error_occurred = False
    prev_y_pressed = False
    prev_x_pressed = False
    prev_arm_enabled = False
    control_tick = 0

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
        with control_loop(frequency=config.control_frequency, keep_running=rclpy.ok) as ctx:
            while ctx.ok():
                if (
                    enable_logging
                    and recorder.is_recording
                    and recorder.recording_started_at is not None
                    and (time.time() - recorder.recording_started_at) > config.max_duration_s
                ):
                    if collection_robot.gripper_busy:
                        continue
                    print(
                        "\n[Recording] Max duration reached, saving episode and returning to start..."
                    )
                    recorder.save(
                        save_episode=recorder.has_samples,
                        success=None,
                        arm=arm,
                        final_gripper=collection_robot.gripper_state,
                    )
                    collection_robot.reset_to_start()
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
                        print("\n[Recording] Ignored Y press because logging is disabled.")
                    elif collection_robot.gripper_busy:
                        print("\n[Recording] Ignored Y press because gripper is busy.")
                    else:
                        print(
                            "\n[Control] Y pressed, saving episode and returning to start..."
                            if recorder.has_samples
                            else "\n[Control] Y pressed, resetting with no captured frames."
                        )
                        recorder.save(
                            save_episode=recorder.has_samples,
                            success=True,
                            arm=arm,
                            final_gripper=collection_robot.gripper_state,
                        )
                        collection_robot.reset_to_start()
                    continue

                if x_edge:
                    if not enable_logging:
                        print("\n[Recording] Ignored X press because logging is disabled.")
                    elif collection_robot.gripper_busy:
                        print("\n[Recording] Ignored X press because gripper is busy.")
                    else:
                        print(
                            "\n[Control] X pressed, saving episode as failed and returning to start..."
                            if recorder.has_samples
                            else "\n[Control] X pressed, resetting with no captured frames."
                        )
                        recorder.save(
                            save_episode=recorder.has_samples,
                            success=False,
                            arm=arm,
                            final_gripper=collection_robot.gripper_state,
                        )
                        collection_robot.reset_to_start()
                    continue

                qpos = current_joint_position(arm)
                sample_ns = time.monotonic_ns()
                robot_state = arm.get_state()
                control_tick += 1
                if control_tick % 10 == 0:
                    line = f"[Joint] current q = {format_joint_position(qpos)}"
                    print(f"{line:<140}", end="\r", flush=True)
                ee_pos, ee_quat_xyzw = current_ee_pose(arm)
                ee_pose6 = EEPose.from_position_quat(ee_pos, ee_quat_xyzw).vector
                if enable_logging and recorder.is_recording:
                    recorder.complete_action(qpos, ee_pose6, collection_robot.gripper_state)

                arm_enabled = bool(vr.arm_enabled)
                if not arm_enabled and prev_arm_enabled:
                    collection_robot.hold.refresh(arm)
                    vr_mapper.reset()
                elif not arm_enabled:
                    vr_mapper.reset()
                prev_arm_enabled = arm_enabled

                if arm_enabled:
                    hold_ee_quat_wxyz = quat_xyzw_to_wxyz(collection_robot.hold.quaternion_xyzw)
                    target_ee_pos, target_ee_quat_wxyz = vr_mapper.map(
                        vr,
                        collection_robot.hold.position,
                        hold_ee_quat_wxyz,
                    )
                    target_ee_quat_xyzw = quat_wxyz_to_xyzw(target_ee_quat_wxyz)
                else:
                    target_ee_pos = collection_robot.hold.position.copy()
                    target_ee_quat_xyzw = collection_robot.hold.quaternion_xyzw.copy()

                command_translation = target_ee_pos - collection_robot.hold.position
                rotation_error = quat_angle_xyzw(
                    target_ee_quat_xyzw, collection_robot.hold.quaternion_xyzw
                )
                motion_norm = float(np.linalg.norm(command_translation))
                rotation_motion_threshold = max(float(config.action_epsilon), 1e-3)
                has_ee_motion_cmd = bool(
                    arm_enabled
                    and not collection_robot.gripper_busy
                    and (
                        motion_norm >= motion_start_threshold
                        or rotation_error >= rotation_motion_threshold
                    )
                )

                if not collection_robot.gripper_busy:
                    if has_ee_motion_cmd:
                        if control_mode == "joint":
                            arm.send_joint_target(qpos)
                        else:
                            arm.send_ee_target(target_ee_pos, target_ee_quat_xyzw)
                        collection_robot.hold.position = target_ee_pos.copy()
                        collection_robot.hold.quaternion_xyzw = target_ee_quat_xyzw.copy()
                        collection_robot.hold.joints = qpos.copy()
                    else:
                        collection_robot.hold.apply(arm)

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
                    collection_robot.gripper_state = float(
                        soft_gripper.gripper_open_ratio.reshape(-1)[0]
                    )
                    collection_robot.last_gripper_cmd = collection_robot.gripper_state
                    collection_robot.gripper_busy = False
                elif gripper_type == "franka":
                    gripper_cmd = collection_robot.last_gripper_cmd
                    if vr.gripper_close:
                        gripper_cmd = 0.0
                    elif vr.gripper_open:
                        gripper_cmd = 1.0
                    gripper_changed = collection_robot.request_franka_gripper(gripper_cmd)
                else:
                    collection_robot.gripper_busy = False

                if not enable_logging:
                    continue

                has_action = has_ee_motion_cmd or gripper_changed
                if has_action and not recorder.is_recording:
                    episode_index = recorder.start()
                    print(
                        "\n[Recording] First motion detected, start logging "
                        f"for episode {episode_index:06d} with prompt: {active_instruction}"
                    )

                if not recorder.is_recording:
                    continue

                recorder.queue_action(
                    ActionSample(
                        sample_index=recorder.action_count,
                        host_sample_monotonic_ns=sample_ns,
                        joint_position=qpos.copy(),
                        gripper_position=collection_robot.gripper_state,
                        ee_position=ee_pos.copy(),
                        ee_orientation_xyzw=ee_quat_xyzw.copy(),
                        ee_pose=ee_pose6.copy(),
                        target_ee_position=collection_robot.hold.position.copy(),
                        target_ee_orientation_xyzw=collection_robot.hold.quaternion_xyzw.copy(),
                        vr_pose_seq=int(vr.pose_seq),
                        vr_pose_monotonic_ns=int(vr.pose_monotonic_ns),
                        vr_position=(float(vr.pos_x), float(vr.pos_y), float(vr.pos_z)),
                        vr_orientation_xyzw=(
                            float(vr.quat_x),
                            float(vr.quat_y),
                            float(vr.quat_z),
                            float(vr.quat_w),
                        ),
                        vr_arm_enabled=bool(vr.arm_enabled),
                    )
                )

                external_img, wrist_img, external_ts, wrist_ts = camera_manager.get_frames()
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
                if not recorder.camera_pair_gate.accept(
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
                        right_img, left_img, _, _ = soft_gripper.dual_camera_manager.get_frames()
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

                recorder.complete_frame(qpos, ee_pose6, collection_robot.gripper_state)
                recorder.queue_frame(
                    FrameSample(
                        frame_index=recorder.frame_count,
                        host_frame_monotonic_ns=time.monotonic_ns(),
                        external_camera_timestamp=float(external_ts.camera_timestamp),
                        wrist_camera_timestamp=float(wrist_ts.camera_timestamp),
                        external_host_capture_monotonic_ns=int(
                            external_ts.host_capture_monotonic_ns
                        ),
                        wrist_host_capture_monotonic_ns=int(wrist_ts.host_capture_monotonic_ns),
                        joint_position=np.asarray(robot_state["joint_positions"], dtype=np.float32),
                        gripper_position=float(collection_robot.gripper_state),
                        ee_position=ee_pos.copy(),
                        ee_orientation_xyzw=ee_quat_xyzw.copy(),
                        ee_pose=ee_pose6.copy(),
                        external_img=external_img,
                        wrist_img=wrist_img,
                        gripper_left_img=gripper_left_img,
                        gripper_right_img=gripper_right_img,
                    )
                )

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
        with contextlib.suppress(Exception):
            collection_robot.hold.apply(arm)
            arm.stop_stream()

        if enable_logging:
            try:
                recorder.save(
                    save_episode=recorder.has_samples and not reflex_error_occurred,
                    success=None,
                    arm=arm,
                    final_gripper=collection_robot.gripper_state,
                )
            except Exception as exc:
                if not reflex_error_occurred:
                    print(f"\n[Error] Failed to finalize current episode: {exc}")
        with contextlib.suppress(Exception):
            camera_manager.close()
        if soft_gripper is not None:
            with contextlib.suppress(Exception):
                soft_gripper.close()

        with contextlib.suppress(Exception):
            vr_reader.stop()

        with contextlib.suppress(Exception):
            arm.close()

        if enable_logging and dataset is not None:
            if frame_writer is not None:
                try:
                    frame_writer.close()
                except Exception as exc:
                    print(f"[Error] Failed to close LeRobot frame writer: {exc}")
            active_error = sys.exc_info()[0] is not None
            try:
                dataset.close()
            except Exception as exc:
                if not active_error:
                    raise
                print(f"[Error] Failed to finalize LeRobot dataset: {exc}")


if __name__ == "__main__":
    main()
