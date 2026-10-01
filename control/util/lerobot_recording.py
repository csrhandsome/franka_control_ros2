"""Common recording API for the pinned LeRobot v2 and v3 dataset formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any

_BACKENDS = {
    "v2": ("control.util.lerobot_dataset", "LeRobotV2Dataset"),
    "v3": ("control.util.lerobot_dataset", "LeRobotV3Dataset"),
}


def parse_lerobot_format(value: object) -> str:
    selected = str(value).strip().lower()
    if selected not in _BACKENDS:
        raise ValueError(
            f"lerobot_format must be one of {sorted(_BACKENDS)}, got {value!r}"
        )
    return selected


class RecordingDataset(ABC):
    """Expose the same frame and episode operations for both disk formats."""

    def __init__(
        self,
        repo_id: str,
        *,
        fps: float,
        image_hw: int,
        root: Path,
        action_space: str,
        force: bool = False,
    ) -> None:
        self.force = force
        self._dataset = self._open_dataset(
            repo_id,
            fps=fps,
            image_hw=image_hw,
            root=root,
            action_space=action_space,
            force=force,
        )

    @abstractmethod
    def _open_dataset(
        self,
        repo_id: str,
        *,
        fps: float,
        image_hw: int,
        root: Path,
        action_space: str,
        force: bool,
    ) -> Any: ...

    def add_frame(self, frame: dict[str, Any]) -> None:
        self._dataset.add_frame(frame)

    def save_episode(self) -> None:
        self._prepare_episode_for_save()
        self._dataset.save_episode()

    @abstractmethod
    def _prepare_episode_for_save(self) -> None: ...

    @abstractmethod
    def discard_episode(self) -> None: ...

    @property
    @abstractmethod
    def next_episode_index(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...


def open_recording_dataset(
    dataset_format: str,
    repo_id: str,
    *,
    fps: float,
    image_hw: int,
    root: Path,
    action_space: str,
    force: bool = False,
) -> RecordingDataset:
    selected = parse_lerobot_format(dataset_format)
    module_name, class_name = _BACKENDS[selected]
    try:
        backend_type = getattr(import_module(module_name), class_name)
        return backend_type(
            repo_id,
            fps=fps,
            image_hw=image_hw,
            root=root,
            action_space=action_space,
            force=force,
        )
    except ImportError as exc:
        runtime = "franka_humble" if selected == "v2" else "franka_jazzy"
        raise RuntimeError(
            f"LeRobot {selected} is unavailable. "
            f"Run this format in the {runtime} Docker image."
        ) from exc
