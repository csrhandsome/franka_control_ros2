#!/usr/bin/env python3
"""Run vr_collect end to end with synthetic inputs and the real LeRobot writer."""

from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import vr_collect
from control.dual_camera_manager_ros import CameraFrameTimestamps
from control.vr_input import VRInput


class FakeArm:
    active: FakeArm | None = None

    def __init__(self, *, start_joint_position, **_kwargs) -> None:
        self.tick = 0
        self.joints = np.asarray(start_joint_position, dtype=np.float64)
        self.position = np.array([0.3, 0.0, 0.5], dtype=np.float64)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        FakeArm.active = self

    @property
    def state(self) -> dict:
        return {
            "joint_positions": self.joints.copy(),
            "joint_velocities": np.zeros(7, dtype=np.float64),
        }

    @property
    def ee_pose_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = self.position
        return matrix

    def move_to_joint_position(self, target) -> None:
        self.joints = np.asarray(target, dtype=np.float64)

    def wait_until_stopped(self) -> bool:
        return True

    def start_ee_streaming(self, **_kwargs) -> None:
        pass

    def start_joint_streaming(self, **_kwargs) -> None:
        pass

    def stop_ee_streaming(self) -> None:
        pass

    def set_ee_control(self, position, quaternion, _joints) -> None:
        self.position = np.asarray(position, dtype=np.float64)
        self.quaternion = np.asarray(quaternion, dtype=np.float64)

    @contextmanager
    def control_loop(self, *, frequency):
        assert frequency > 0
        arm = self

        class Context:
            def ok(self) -> bool:
                arm.tick += 1
                return arm.tick <= 32

        yield Context()

    def cleanup(self) -> None:
        pass


class FakeVR:
    def __init__(self, **_kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def latest(self) -> VRInput:
        tick = FakeArm.active.tick
        return VRInput(
            arm_enabled=tick < 24,
            pos_x=min(tick, 20) * 0.01,
            pose_seq=tick,
            y_pressed=tick == 24,
        )


class FakeCameras:
    def __init__(self, *, out_hw, **_kwargs) -> None:
        self.size = out_hw
        self.external_camera = SimpleNamespace(serial="synthetic-external")
        self.wrist_camera = SimpleNamespace(serial="synthetic-wrist")

    def connect(self) -> None:
        pass

    def wait_for_frames(self, **_kwargs) -> None:
        pass

    def get_frames(self):
        tick = FakeArm.active.tick
        image = np.full((self.size, self.size, 3), tick, dtype=np.uint8)
        stamp = CameraFrameTimestamps(
            camera_timestamp=float(tick) / 30.0,
            host_capture_monotonic_ns=tick * 33_333_333,
        )
        return image, image, stamp, stamp

    def close(self) -> None:
        pass


def run_case(action_space: str) -> None:
    with tempfile.TemporaryDirectory(prefix="franka-collect-smoke-") as tmp:
        root = Path(tmp)
        config = yaml.safe_load(
            (
                Path(__file__).resolve().parents[1] / "config/collect/franka.yaml"
            ).read_text()
        )
        config["dataset"].update(
            repo_id="smoke/franka", date="test", enable_logging=True,
            action_space=action_space,
        )
        config["camera"].update(camera_backend="ros", image_hw=64)
        config["robot"]["use_fake_hardware"] = True
        config["gripper"]["gripper_type"] = "none"
        config_path = root / "collect.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        with (
            patch.object(vr_collect, "__file__", str(root / "vr_collect.py")),
            patch.object(vr_collect, "RoboticArmControlerRos", FakeArm),
            patch.object(vr_collect, "VRInputRos", FakeVR),
            patch.object(vr_collect, "DualRealsenseManagerRos", FakeCameras),
            patch.object(sys, "argv", ["vr_collect.py", "--config", str(config_path)]),
        ):
            vr_collect.main()

        dataset_root = root / "data/smoke/franka_test"
        info = json.loads((dataset_root / "meta/info.json").read_text())
        sync = json.loads((dataset_root / "episode_000000.sync.json").read_text())
        actions = (
            (dataset_root / "episode_000000.actions.jsonl").read_text().splitlines()
        )
        replay = LeRobotDataset(repo_id="smoke/franka_test", root=dataset_root)
        assert info["total_episodes"] == 1, info
        assert len(replay) == sync["video_frames"] > 0
        assert len(actions) == sync["action_records"] > 0
        assert sync["success"] is True
        assert sync["action_space"] == action_space
        assert sync["action_target_offset_frames"] == 1
        assert tuple(replay[0]["ee_pose"].shape) == (6,)
        assert tuple(replay[0]["joint_position"].shape) == (7,)
        assert tuple(replay[0]["actions"].shape) == ((7,) if action_space == "ee" else (8,))
        assert info["features"]["actions"]["shape"] == ([7] if action_space == "ee" else [8])
        next_frame = replay[1]
        expected_arm = next_frame["ee_pose"] if action_space == "ee" else next_frame["joint_position"]
        np.testing.assert_allclose(replay[0]["actions"][:-1], expected_arm, atol=1e-6)
        np.testing.assert_allclose(
            replay[0]["actions"][-1],
            np.asarray(next_frame["gripper_position"]).reshape(-1)[0],
            atol=1e-6,
        )
        assert tuple(replay[0]["wrist_image_left"].shape) == (3, 64, 64)
        print(f"PASS: {action_space} actions match the next synthetic Docker frame")


def main() -> None:
    for action_space in ("ee", "joint"):
        run_case(action_space)


if __name__ == "__main__":
    main()
