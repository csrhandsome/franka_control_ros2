#!/usr/bin/env python3
"""Exercise the production LeRobot recording path with synthetic frames."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rclpy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from control.recording_writer import AsyncDatasetFrames
from control.util.lerobot_util import (
    _load_or_create_dataset,
    _prepare_episode_for_save,
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="franka-lerobot-smoke-") as tmp:
        root = Path(tmp) / "dataset"
        repo_id = "smoke/franka"
        dataset = _load_or_create_dataset(repo_id, fps=30, image_hw=64, root=root)
        writer = AsyncDatasetFrames(dataset)
        try:
            for index in range(3):
                image = np.full((64, 64, 3), index * 40, dtype=np.uint8)
                writer.submit(
                    {
                        "exterior_image_1_left": image,
                        "exterior_image_2_left": image,
                        "wrist_image_left": image,
                        "joint_position": np.full(7, index, dtype=np.float32),
                        "ee_pose": np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
                        "gripper_position": np.array([0.5], dtype=np.float32),
                        "actions": np.full(8, index + 1, dtype=np.float32),
                        "task": "synthetic container recording test",
                    }
                )
            writer.drain()
            _prepare_episode_for_save(dataset)
            dataset.save_episode()
        finally:
            writer.close()
            dataset.stop_image_writer()

        info = json.loads((root / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 1, info
        assert info["total_frames"] == 3, info
        assert list(root.glob("data/**/*.parquet")), "No episode Parquet file"
        assert info["total_videos"] == 0, info

        replay = LeRobotDataset(repo_id=repo_id, root=root)
        assert len(replay) == 3, len(replay)
        sample = replay[1]
        assert tuple(sample["joint_position"].shape) == (7,)
        assert tuple(sample["ee_pose"].shape) == (6,)
        assert tuple(sample["actions"].shape) == (8,)
        assert tuple(sample["exterior_image_1_left"].shape) == (3, 64, 64)
        assert rclpy.__file__.startswith("/opt/ros/humble/"), rclpy.__file__
        print("PASS: Humble container wrote and read a 3-frame LeRobot episode")


if __name__ == "__main__":
    main()
