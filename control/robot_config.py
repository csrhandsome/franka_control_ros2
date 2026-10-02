"""Resolve root ``config/{collect,hitl,inference}/{robot}.yaml`` files.

Later robots add another yaml next to ``franka.yaml`` (for example ``ur.yaml``)
and pass ``--robot ur``. Do not flatten keys that collide across sections;
use ``control_mode`` and ``gripper_type``.
"""

from __future__ import annotations

from copy import deepcopy
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


def load_mapping(path: Path, *, _stack: tuple[Path, ...] = ()) -> dict:
    """Load relative `extends` files, then recursively apply local overrides.

    Mappings merge; lists and scalar values replace their inherited values.
    Each call returns independent data. Cyclic inheritance is rejected.
    """
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    parents = loaded.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(
        not isinstance(parent, str) or not parent for parent in parents
    ):
        raise TypeError(f"extends must be a path or list of paths: {path}")
    result = {}
    for parent in parents:
        _merge_mapping(result, load_mapping(path.parent / parent, _stack=(*_stack, path)))
    _merge_mapping(result, loaded)
    return result


def _merge_mapping(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_mapping(base[key], value)
        else:
            base[key] = deepcopy(value)


def flatten_config(path: Path) -> SimpleNamespace:
    sections = load_mapping(path)
    values = {
        key: value
        for name, section in sections.items()
        if name != "motion" and isinstance(section, dict)
        for key, value in section.items()
    }
    # Motion is a nested controller schema; flattening it would overwrite robot
    # and gripper workflow keys and discard its section names.
    values["motion"] = deepcopy(sections.get("motion", {}))
    return SimpleNamespace(**values)


def section(config: dict, name: str) -> SimpleNamespace:
    payload = config.get(name) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config section {name!r} must be a mapping")
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
