"""Exercise the production LeRobot recording path with synthetic frames."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rclpy

from control.recording_writer import AsyncDatasetFrames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("humble", "jazzy"), default="humble")
    backend = parser.parse_args().backend
    if backend == "jazzy":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from control.util import lerobot_dataset as recorder
    else:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        from control.util import lerobot_util as recorder

    with tempfile.TemporaryDirectory(prefix="franka-lerobot-smoke-") as tmp:
        root = Path(tmp) / "dataset"
        repo_id = "smoke/franka"
        dataset = recorder._load_or_create_dataset(
            repo_id, fps=30, image_hw=64, root=root
        )
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
                        "ee_pose": np.array(
                            [0.3, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32
                        ),
                        "gripper_position": np.array([0.5], dtype=np.float32),
                        "actions": np.full(8, index + 1, dtype=np.float32),
                        "task": "synthetic container recording test",
                    }
                )
            writer.drain()
            recorder._prepare_episode_for_save(dataset)
            dataset.save_episode()
        finally:
            writer.close()
            recorder._close_dataset(dataset)

        info = json.loads((root / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 1, info
        assert info["total_frames"] == 3, info
        assert list(root.glob("data/**/*.parquet")), "No episode Parquet file"
        if backend == "humble":
            assert info["total_videos"] == 0, info
        else:
            assert info["codebase_version"] == "v3.0", info

        read_options = {"video_backend": "pyav"} if backend == "jazzy" else {}
        replay = LeRobotDataset(repo_id=repo_id, root=root, **read_options)
        assert len(replay) == 3, len(replay)
        sample = replay[1]
        assert tuple(sample["joint_position"].shape) == (7,)
        assert tuple(sample["ee_pose"].shape) == (6,)
        assert tuple(sample["actions"].shape) == (8,)
        assert tuple(sample["exterior_image_1_left"].shape) == (3, 64, 64)
        assert rclpy.__file__.startswith(f"/opt/ros/{backend}/"), rclpy.__file__

        if backend == "jazzy":
            resumed = recorder._load_or_create_dataset(
                repo_id, fps=30, image_hw=64, root=root
            )
            try:
                assert recorder._next_episode_index(resumed) == 1
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                frame = {
                    "exterior_image_1_left": image,
                    "exterior_image_2_left": image,
                    "wrist_image_left": image,
                    "joint_position": np.zeros(7, dtype=np.float32),
                    "ee_pose": np.zeros(6, dtype=np.float32),
                    "gripper_position": np.array([0.5], dtype=np.float32),
                    "actions": np.zeros(8, dtype=np.float32),
                    "task": "discarded episode",
                }
                resumed.add_frame(frame.copy())
                recorder._discard_unsaved_episode(resumed)
                assert recorder._next_episode_index(resumed) == 1
                frame["task"] = "resumed episode"
                resumed.add_frame(frame)
                resumed.save_episode()
            finally:
                recorder._close_dataset(resumed)
            assert len(LeRobotDataset(repo_id=repo_id, root=root, **read_options)) == 4

            force_root = Path(tmp) / "force_dataset"
            force = recorder._load_or_create_dataset_force(
                "smoke/franka_force", fps=30, image_hw=64, root=force_root
            )
            try:
                force_frame = frame.copy()
                force_frame["task"] = "synthetic force episode"
                force_frame["gripper_image_left"] = image
                force_frame["gripper_image_right"] = image
                for name in (
                    "external_camera_timestamp_ms",
                    "wrist_camera_timestamp_ms",
                    "external_camera_frame_age_s",
                    "wrist_camera_frame_age_s",
                ):
                    force_frame[name] = np.array([1.0], dtype=np.float32)
                force.add_frame(force_frame)
                force.save_episode()
            finally:
                recorder._close_dataset(force)
            force_replay = LeRobotDataset(
                repo_id="smoke/franka_force", root=force_root, **read_options
            )
            assert tuple(force_replay[0]["gripper_image_left"].shape) == (3, 64, 64)

        print(f"PASS: {backend} production LeRobot recording and readback")


if __name__ == "__main__":
    main()
