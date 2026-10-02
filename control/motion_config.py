"""Configuration shared by ROS streaming and timed trajectory execution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from control.robot_config import load_mapping


def default_motion_config(robot_type: str = "panda") -> dict:
    """Read concrete defaults from YAML, not Python constants."""
    if robot_type != "panda":
        raise ValueError("Only Panda motion configuration is supported")
    path = Path(__file__).resolve().parents[1] / "config" / "planer" / "panda.yaml"
    return load_mapping(path)


def _merge(base: dict, override: dict, prefix: str = "") -> None:
    for key, value in override.items():
        path = f"{prefix}{key}"
        if key not in base:
            raise ValueError(f"Unknown motion configuration key: {path}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be a mapping")
            _merge(base[key], value, path + ".")
        else:
            base[key] = deepcopy(value)


def load_motion_config(
    path: str | Path | None = None,
    *,
    robot_type: str = "panda",
    overrides: dict | None = None,
) -> dict:
    values = {}
    if path is not None:
        values = load_mapping(Path(path))
        if not isinstance(values, dict):
            raise ValueError("Motion configuration must be a mapping")
        robot_type = values.get("robot", {}).get("robot_type", robot_type)
    if robot_type != "panda":
        raise ValueError("Only Panda motion configuration is supported")
    config = default_motion_config(robot_type)
    _merge(config, values)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError("motion must be a mapping")
        _merge(config, overrides)
    if config["robot"]["robot_type"] != robot_type:
        raise ValueError("robot_type conflicts with the motion configuration")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported motion configuration schema")
    if config["robot"]["quaternion_order"] != "xyzw":
        raise ValueError("Only xyzw quaternions are supported")
    names = config["robot"]["joint_names"]
    if (
        not isinstance(names, list)
        or len(names) != 7
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != 7
    ):
        raise ValueError("joint_names must contain seven unique names")

    def validate(mapping: dict, prefix: str = "") -> None:
        for key, value in mapping.items():
            name = prefix + key
            if isinstance(value, dict):
                validate(value, name + ".")
            elif key in ("enabled", "use_fake_hardware"):
                if type(value) is not bool:
                    raise ValueError(f"{name} must be boolean")
            elif key.endswith(("_s", "_hz", "_rad", "_rad_s", "_rad_s2", "_m", "_m_s", "_m_s2")):
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise ValueError(f"{name} must be a positive number")
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be a positive finite number")
            elif key not in ("schema_version", "joint_names"):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} must be a nonempty string")

    validate(config)
    return config


def workflow_motion_config(config: dict, *, control_mode: str) -> dict:
    """Resolve workflow motion overrides before creating any ROS resources."""
    from control.robot_config import parse_control_mode, parse_gripper_type

    mode = parse_control_mode(control_mode)
    if "motion" in config and not isinstance(config["motion"], dict):
        raise ValueError("motion must be a mapping")
    robot = config.get("robot", {})
    overrides = config.get("motion", {})
    if not isinstance(overrides, dict):
        raise ValueError("motion must be a mapping")
    motion = load_motion_config(
        robot_type=robot.get("robot_type", "panda"),
        overrides=overrides,
    )
    if "use_fake_hardware" in robot:
        if type(robot["use_fake_hardware"]) is not bool:
            raise ValueError("robot.use_fake_hardware must be boolean")
        motion["robot"]["use_fake_hardware"] = robot["use_fake_hardware"]
    enabled = (
        motion["ee"]["enabled"]
        if mode == "ee"
        else motion["joint"]["streaming"]["enabled"]
    )
    if not enabled:
        raise ValueError(f"control_mode={mode} requires enabled {mode} streaming in motion")
    kind = parse_gripper_type(config.get("gripper", {}).get("gripper_type"))
    if kind == "franka" and not motion["gripper"]["enabled"]:
        raise ValueError("gripper_type=franka requires motion.gripper.enabled: true")
    if kind != "franka":
        motion["gripper"]["enabled"] = False
    return motion
