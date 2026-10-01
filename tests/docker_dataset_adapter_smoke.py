"""Verify the shared recording API against the pinned v2 or v3 Docker image."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from control.recording_writer import AsyncDatasetFrames
from control.util.lerobot_recording import RecordingDataset, open_recording_dataset


def _frame(index: int, *, action_space: str, force: bool) -> dict:
    image = np.full((64, 64, 3), index * 30, dtype=np.uint8)
    frame = {
        "exterior_image_1_left": image,
        "exterior_image_2_left": image,
        "wrist_image_left": image,
        "joint_position": np.full(7, index, dtype=np.float32),
        "ee_pose": np.full(6, index, dtype=np.float32),
        "gripper_position": np.array([0.5], dtype=np.float32),
        "actions": np.full(
            7 if action_space == "ee" else 8, index + 1, dtype=np.float32
        ),
        "task": "synthetic adapter test",
    }
    if force:
        frame["gripper_image_left"] = image
        frame["gripper_image_right"] = image
        for name in (
            "external_camera_timestamp_ms",
            "wrist_camera_timestamp_ms",
            "external_camera_frame_age_s",
            "wrist_camera_frame_age_s",
        ):
            frame[name] = np.array([float(index)], dtype=np.float32)
    return frame


def _run_case(dataset_format: str, *, force: bool) -> None:
    if dataset_format == "v3":
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        read_options = {"video_backend": "pyav"}
    else:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        read_options = {}

    action_space = "ee" if force else "joint"
    with tempfile.TemporaryDirectory(
        prefix=f"lerobot-{dataset_format}-adapter-"
    ) as tmp:
        root = Path(tmp) / "dataset"
        repo_id = f"smoke/adapter_{dataset_format}_{action_space}"
        options = {
            "fps": 30,
            "image_hw": 64,
            "root": root,
            "action_space": action_space,
            "force": force,
        }
        dataset = open_recording_dataset(dataset_format, repo_id, **options)
        assert isinstance(dataset, RecordingDataset)
        assert dataset.next_episode_index == 0
        writer = AsyncDatasetFrames(dataset)
        try:
            writer.submit(_frame(0, action_space=action_space, force=force))
            writer.submit(_frame(1, action_space=action_space, force=force))
            writer.drain()
            dataset.save_episode()
            assert dataset.next_episode_index == 1
        finally:
            writer.close()
            dataset.close()

        resumed = open_recording_dataset(dataset_format, repo_id, **options)
        try:
            assert resumed.next_episode_index == 1
            resumed.add_frame(_frame(2, action_space=action_space, force=force))
            resumed.discard_episode()
            assert resumed.next_episode_index == 1
            resumed.add_frame(_frame(3, action_space=action_space, force=force))
            resumed.save_episode()
        finally:
            resumed.close()

        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        assert info["total_episodes"] == 2, info
        assert info["total_frames"] == 3, info
        replay = LeRobotDataset(repo_id=repo_id, root=root, **read_options)
        assert len(replay) == 3
        assert int(replay[2]["episode_index"]) == 1
        np.testing.assert_allclose(replay[2]["joint_position"], np.full(7, 3))
        assert tuple(replay[2]["actions"].shape) == ((7,) if force else (8,))
        assert tuple(replay[2]["wrist_image_left"].shape) == (3, 64, 64)
        if force:
            assert tuple(replay[2]["gripper_image_left"].shape) == (3, 64, 64)
            np.testing.assert_allclose(replay[2]["external_camera_timestamp_ms"], [3.0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("v2", "v3"), required=True)
    dataset_format = parser.parse_args().format
    _run_case(dataset_format, force=False)
    _run_case(dataset_format, force=True)
    print(
        f"PASS: shared recording API saved, discarded, resumed and read {dataset_format}"
    )


if __name__ == "__main__":
    main()
