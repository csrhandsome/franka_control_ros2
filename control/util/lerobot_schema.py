"""Feature schema shared by the LeRobot v2 and v3 recording backends."""

from __future__ import annotations


def _image_feature(image_hw: int) -> dict[str, object]:
    return {
        "dtype": "image",
        "shape": (image_hw, image_hw, 3),
        "names": ["height", "width", "channel"],
    }


def _scalar_feature(name: str) -> dict[str, object]:
    return {"dtype": "float32", "shape": (1,), "names": [name]}


def build_features(
    image_hw: int, action_space: str, *, force: bool = False
) -> dict[str, dict[str, object]]:
    if action_space not in {"ee", "joint"}:
        raise ValueError(f"Unsupported action space: {action_space}")
    action_dim = 7 if action_space == "ee" else 8
    features = {
        "exterior_image_1_left": _image_feature(image_hw),
        "exterior_image_2_left": _image_feature(image_hw),
        "wrist_image_left": _image_feature(image_hw),
    }
    if force:
        features["gripper_image_left"] = _image_feature(image_hw)
        features["gripper_image_right"] = _image_feature(image_hw)
    features.update(
        {
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "ee_pose": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
            "gripper_position": _scalar_feature("gripper_position"),
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        }
    )
    if force:
        for name in (
            "external_camera_timestamp_ms",
            "wrist_camera_timestamp_ms",
            "external_camera_frame_age_s",
            "wrist_camera_frame_age_s",
        ):
            features[name] = _scalar_feature(name)
    return features
