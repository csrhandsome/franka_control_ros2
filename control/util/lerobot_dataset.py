"""Recording adapters for the Humble (v2) and Jazzy (v3) LeRobot images."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from control.util.lerobot_recording import RecordingDataset
from control.util.lerobot_schema import build_features

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotV2Dataset(RecordingDataset):
    """Legacy LeRobot writer used in the Humble image."""

    def _open_dataset(
        self,
        repo_id: str,
        *,
        fps: float,
        image_hw: int,
        root: Path,
        action_space: str,
        force: bool,
    ) -> Any:
        legacy = import_module("control.util.lerobot_util")
        loader = (
            legacy._load_or_create_dataset_force
            if force
            else legacy._load_or_create_dataset
        )
        return loader(
            repo_id, fps=fps, image_hw=image_hw, root=root, action_space=action_space
        )

    def _prepare_episode_for_save(self) -> None:
        legacy = import_module("control.util.lerobot_util")
        prepare = (
            legacy._prepare_episode_for_save_force
            if self.force
            else legacy._prepare_episode_for_save
        )
        prepare(self._dataset)

    def discard_episode(self) -> None:
        import_module("control.util.lerobot_util")._discard_unsaved_episode(
            self._dataset
        )

    @property
    def next_episode_index(self) -> int:
        return import_module("control.util.lerobot_util")._next_episode_index(
            self._dataset
        )

    def close(self) -> None:
        import_module("control.util.lerobot_util")._close_dataset(self._dataset)


_DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _validate_existing(root: Path, fps: int, features: dict[str, dict]) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Existing directory is not a LeRobot dataset: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(
            f"Existing dataset is {info.get('codebase_version')}, not LeRobot v3. "
            "Use a new dataset repo_id/date."
        )
    if not np.isclose(float(info["fps"]), fps, rtol=0, atol=1e-6):
        raise RuntimeError(
            f"Existing dataset uses {info['fps']} fps, but this run needs {fps} fps. "
            "Use a new dataset repo_id/date."
        )
    existing = info["features"]
    if set(existing) - _DEFAULT_FEATURES != set(features):
        raise RuntimeError(
            "Existing dataset has a different feature set. Use a new repo_id/date."
        )
    for name, expected in features.items():
        actual = existing[name]
        if actual["dtype"] != expected["dtype"] or actual["shape"] != list(
            expected["shape"]
        ):
            raise RuntimeError(
                f"Existing dataset feature {name} has an incompatible dtype or shape. "
                "Use a new repo_id/date."
            )
    next_episode = int(info["total_episodes"])
    orphan_dirs = list((root / "images").glob(f"*/episode-{next_episode:06d}"))
    if orphan_dirs:
        raise RuntimeError(
            f"Unsaved images exist for episode {next_episode:06d}: {orphan_dirs[0]}. "
            "Inspect the interrupted episode before resuming."
        )


def _load_or_create(
    repo_id: str,
    *,
    fps: float,
    image_hw: int,
    root: Path,
    action_space: str,
    force: bool,
) -> LeRobotDataset:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if action_space not in {"ee", "joint"}:
        raise ValueError(f"Unsupported action space: {action_space}")
    if not np.isclose(fps, round(fps), rtol=0, atol=1e-6) or fps <= 0:
        raise ValueError(f"LeRobot v3 requires a positive integer fps, got {fps}")
    fps = round(fps)
    features = build_features(image_hw, action_space, force=force)
    if root.exists():
        if not root.is_dir():
            raise RuntimeError(f"Dataset path exists and is not a directory: {root}")
        _validate_existing(root, fps, features)
        return LeRobotDataset.resume(
            repo_id=repo_id,
            root=root,
            image_writer_threads=6,
            video_backend="pyav",
        )
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        root=root,
        features=features,
        use_videos=False,
        image_writer_threads=6,
        video_backend="pyav",
    )


def _load_or_create_dataset(
    repo_id: str, *, fps: float, image_hw: int, root: Path, action_space: str = "joint"
) -> LeRobotDataset:
    return _load_or_create(
        repo_id,
        fps=fps,
        image_hw=image_hw,
        root=root,
        action_space=action_space,
        force=False,
    )


def _load_or_create_dataset_force(
    repo_id: str, *, fps: float, image_hw: int, root: Path, action_space: str = "joint"
) -> LeRobotDataset:
    return _load_or_create(
        repo_id,
        fps=fps,
        image_hw=image_hw,
        root=root,
        action_space=action_space,
        force=True,
    )


def _prepare_episode_for_save(_dataset: LeRobotDataset) -> None:
    # The v3 writer converts shape-(1,) numeric values to scalars during save_episode().
    pass


def _prepare_episode_for_save_force(dataset: LeRobotDataset) -> None:
    _prepare_episode_for_save(dataset)


def _discard_unsaved_episode(dataset: LeRobotDataset) -> None:
    dataset.clear_episode_buffer(delete_images=True)


def _next_episode_index(dataset: LeRobotDataset) -> int:
    return int(dataset.meta.total_episodes)


def _close_dataset(dataset: LeRobotDataset) -> None:
    dataset.finalize()


class LeRobotV3Dataset(RecordingDataset):
    """LeRobot v3 writer used in the Jazzy image."""

    def _open_dataset(
        self,
        repo_id: str,
        *,
        fps: float,
        image_hw: int,
        root: Path,
        action_space: str,
        force: bool,
    ) -> LeRobotDataset:
        return _load_or_create(
            repo_id,
            fps=fps,
            image_hw=image_hw,
            root=root,
            action_space=action_space,
            force=force,
        )

    def _prepare_episode_for_save(self) -> None:
        _prepare_episode_for_save(self._dataset)

    def discard_episode(self) -> None:
        _discard_unsaved_episode(self._dataset)

    @property
    def next_episode_index(self) -> int:
        return _next_episode_index(self._dataset)

    def close(self) -> None:
        _close_dataset(self._dataset)
