"""Robot environments used by Human in the Loop collection."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np

from control.hitl.types import FRONT_IMAGE_KEY, STATE_KEY, WRIST_IMAGE_KEY
from control.robot_config import parse_control_mode, parse_gripper_type
from control.robot_config import section as _section


def blank_image(size: int) -> np.ndarray:
    return np.zeros((int(size), int(size), 3), dtype=np.uint8)


def state_vector(qpos: np.ndarray, gripper: float, ee_xyz: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(qpos, dtype=np.float32).reshape(7),
            [np.float32(gripper)],
            np.asarray(ee_xyz, dtype=np.float32).reshape(3),
        ]
    ).astype(np.float32)


class RobotEnv(Protocol):
    def reset(self) -> None: ...

    def observation(self) -> dict[str, Any]: ...

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]: ...

    def apply(
        self,
        target_xyz: np.ndarray,
        gripper_close: bool,
        target_quat_xyzw: np.ndarray | None = None,
    ) -> None: ...

    def close(self) -> None: ...


class DummyRobotEnv:
    def __init__(self, start_joints: list[float], image_size: int) -> None:
        self._start_joints = np.asarray(start_joints, dtype=np.float64).reshape(7)
        self._image_size = int(image_size)
        self.qpos = self._start_joints.copy()
        self.ee_pos = np.array([0.40, 0.00, 0.40], dtype=np.float64)
        self.ee_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.gripper = 0.08

    def reset(self) -> None:
        self.qpos = self._start_joints.copy()
        self.ee_pos = np.array([0.40, 0.00, 0.40], dtype=np.float64)
        self.ee_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.gripper = 0.08

    def observation(self) -> dict[str, Any]:
        return {
            FRONT_IMAGE_KEY: blank_image(self._image_size),
            WRIST_IMAGE_KEY: blank_image(self._image_size),
            STATE_KEY: state_vector(self.qpos, self.gripper, self.ee_pos),
        }

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ee_pos.copy(), self.ee_quat_xyzw.copy()

    def apply(
        self,
        target_xyz: np.ndarray,
        gripper_close: bool,
        target_quat_xyzw: np.ndarray | None = None,
    ) -> None:
        self.ee_pos = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        if target_quat_xyzw is not None:
            self.ee_quat_xyzw = np.asarray(target_quat_xyzw, dtype=np.float64).reshape(
                4
            )
        self.gripper = 0.0 if gripper_close else 0.08

    def close(self) -> None:
        return


class RosRobotEnv:
    def __init__(
        self,
        robot_cfg: SimpleNamespace,
        camera_cfg: SimpleNamespace,
        *,
        control_mode: str,
        gripper_type: str,
        gripper_cfg: SimpleNamespace,
    ) -> None:
        from control.dual_camera_manager_ros import DualRealsenseManagerRos
        from control.robotic_arm_controller_ros import RoboticArmControlerRos

        self._control_mode = control_mode
        self._gripper_type = gripper_type
        self._image_size = int(getattr(camera_cfg, "image_hw", 128))
        self.arm = RoboticArmControlerRos(
            use_fake_hardware=bool(robot_cfg.use_fake_hardware),
            robot_type=str(robot_cfg.robot_type),
            start_joint_position=list(robot_cfg.start_joint_position),
        )
        backend = str(getattr(camera_cfg, "camera_backend", "none"))
        self.cameras = DualRealsenseManagerRos(
            external_topic=str(camera_cfg.external_image_topic),
            wrist_topic=str(camera_cfg.wrist_image_topic),
            external_serial=str(camera_cfg.external_camera_serial),
            wrist_serial=str(camera_cfg.wrist_camera_serial),
            crop_scale=float(getattr(camera_cfg, "crop_scale", 0.9)),
            out_hw=self._image_size,
            refresh_hz=float(getattr(camera_cfg, "camera_fps", 30.0)),
            enabled=backend != "none",
        )
        self.cameras.connect()
        timeout = float(getattr(camera_cfg, "camera_startup_timeout_s", 10.0))
        self.cameras.wait_for_frames(timeout_s=timeout)
        self._gripper_closed = False
        self.soft_gripper = None
        if gripper_type == "dh5":
            from control.soft_gripper_control_ros import DH5GripperRos

            self.soft_gripper = DH5GripperRos(
                left_image_topic=str(
                    getattr(
                        gripper_cfg,
                        "gripper_left_image_topic",
                        "/gripper_left/image_raw",
                    )
                ),
                right_image_topic=str(
                    getattr(
                        gripper_cfg,
                        "gripper_right_image_topic",
                        "/gripper_right/image_raw",
                    )
                ),
                enable_cameras=bool(
                    getattr(gripper_cfg, "enable_soft_gripper_cameras", False)
                ),
                use_fake=bool(getattr(gripper_cfg, "soft_gripper_use_fake", False)),
            )
            self.soft_gripper.set_force(
                int(getattr(gripper_cfg, "soft_gripper_force", 50))
            )
            self.soft_gripper.set_velocity(
                int(getattr(gripper_cfg, "soft_gripper_velocity", 100))
            )

    def reset(self) -> None:
        self.arm.move_to_start()
        self._open_gripper()
        self._gripper_closed = False
        if self._control_mode == "ee":
            self.arm.start_ee_streaming()
        else:
            self.arm.start_joint_streaming()

    def _open_gripper(self) -> None:
        if self.soft_gripper is not None:
            self.soft_gripper.set_gripper_level(0, wait=False)
        elif self._gripper_type == "franka":
            self.arm.gripper_open(wait=False)

    def _set_gripper(self, gripper_close: bool) -> None:
        if gripper_close and not self._gripper_closed:
            if self.soft_gripper is not None:
                self.soft_gripper.set_gripper_level(
                    self.soft_gripper.max_gripper_level, wait=False
                )
            elif self._gripper_type == "franka":
                self.arm.gripper_close(wait=False)
            self._gripper_closed = True
        elif not gripper_close and self._gripper_closed:
            self._open_gripper()
            self._gripper_closed = False

    def _ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.arm._current_ee_pose()

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self._ee_pose()
        return np.asarray(pos, dtype=np.float64).reshape(3), np.asarray(
            quat, dtype=np.float64
        ).reshape(4)

    def observation(self) -> dict[str, Any]:
        front, wrist, _, _ = self.cameras.get_frames()
        if front is None:
            front = blank_image(self._image_size)
        if wrist is None:
            wrist = blank_image(self._image_size)
        state = self.arm.state
        pos, _quat = self._ee_pose()
        return {
            FRONT_IMAGE_KEY: np.asarray(front, dtype=np.uint8),
            WRIST_IMAGE_KEY: np.asarray(wrist, dtype=np.uint8),
            STATE_KEY: state_vector(
                state["joint_positions"],
                float(state["gripper_position"]),
                pos,
            ),
        }

    def apply(
        self,
        target_xyz: np.ndarray,
        gripper_close: bool,
        target_quat_xyzw: np.ndarray | None = None,
    ) -> None:
        _pos, quat = self._ee_pose()
        if target_quat_xyzw is not None:
            quat = np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
        qpos = self.arm.state["joint_positions"]
        if self._control_mode == "ee":
            self.arm.set_ee_control(target_xyz, quat, qpos)
        else:
            self.arm.set_ee_control(_pos, quat, qpos)
        self._set_gripper(gripper_close)

    def close(self) -> None:
        try:
            self.arm.cleanup()
        except Exception:
            pass
        try:
            self.cameras.close()
        except Exception:
            pass
        if self.soft_gripper is not None:
            try:
                self.soft_gripper.close()
            except Exception:
                pass


def make_env(config: dict, *, dry_run: bool) -> RobotEnv:
    robot_cfg = _section(config, "robot")
    camera_cfg = _section(config, "camera")
    control_cfg = _section(config, "control")
    gripper_cfg = _section(config, "gripper")
    control_mode = parse_control_mode(getattr(control_cfg, "control_mode", "ee"))
    gripper_type = parse_gripper_type(getattr(gripper_cfg, "gripper_type", "franka"))
    logging.info(
        "[HITL] control_mode=%s gripper_type=%s dry_run=%s",
        control_mode,
        gripper_type,
        dry_run,
    )
    if dry_run:
        return DummyRobotEnv(
            start_joints=list(robot_cfg.start_joint_position),
            image_size=int(getattr(camera_cfg, "image_hw", 128)),
        )
    return RosRobotEnv(
        robot_cfg,
        camera_cfg,
        control_mode=control_mode,
        gripper_type=gripper_type,
        gripper_cfg=gripper_cfg,
    )
