"""Resolve root ``config/{collect,hitl,inference}/{robot}.yaml`` files.

Later robots add another yaml next to ``franka.yaml`` (for example ``ur.yaml``)
and pass ``--robot ur``. Do not flatten keys that collide across sections;
use ``control_mode`` and ``gripper_type``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
VALID_STAGES = {"collect", "hitl", "inference"}
VALID_CONTROL_MODES = {"ee", "joint"}
VALID_ACTION_SPACES = {"ee", "joint"}
VALID_GRIPPER_TYPES = {"franka", "dh5", "none"}


def config_path(
    *,
    stage: str,
    robot: str = "franka",
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}, got {stage!r}")
    name = str(robot).strip() or "franka"
    path = CONFIG_ROOT / stage / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Create config/{stage}/{name}.yaml for this robot."
        )
    return path


def load_mapping(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def flatten_config(path: Path) -> SimpleNamespace:
    sections = load_mapping(path)
    return SimpleNamespace(
        **{
            key: value
            for section in sections.values()
            if isinstance(section, dict)
            for key, value in section.items()
        }
    )


def section(config: dict, name: str) -> SimpleNamespace:
    payload = config.get(name) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return SimpleNamespace(**payload)


def parse_control_mode(value: object, *, default: str = "ee") -> str:
    mode = str(value if value is not None else default).strip().lower()
    if mode not in VALID_CONTROL_MODES:
        raise ValueError(
            f"control_mode must be one of {sorted(VALID_CONTROL_MODES)}, got {mode!r}"
        )
    return mode


def parse_action_space(value: object, *, default: str = "ee") -> str:
    action_space = str(value if value is not None else default).strip().lower()
    if action_space not in VALID_ACTION_SPACES:
        raise ValueError(
            f"action_space must be one of {sorted(VALID_ACTION_SPACES)}, got {action_space!r}"
        )
    return action_space


def parse_gripper_type(value: object, *, default: str = "franka") -> str:
    kind = str(value if value is not None else default).strip().lower()
    if kind not in VALID_GRIPPER_TYPES:
        raise ValueError(
            f"gripper_type must be one of {sorted(VALID_GRIPPER_TYPES)}, got {kind!r}"
        )
    return kind
