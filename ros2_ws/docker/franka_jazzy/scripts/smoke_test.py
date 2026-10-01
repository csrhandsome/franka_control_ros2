"""Check Jazzy and LeRobot 0.6.1 together with a synthetic local dataset."""

import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import cv_bridge
import numpy as np
import rclpy
from cv_bridge import CvBridge
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    assert sys.version_info[:2] == (3, 12), sys.version
    assert version("lerobot") == "0.6.1"
    assert rclpy.__file__.startswith("/opt/ros/jazzy/"), rclpy.__file__
    assert cv_bridge.__file__.startswith("/opt/cv_bridge_ws/install/"), (
        cv_bridge.__file__
    )

    bridge = CvBridge()
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    ros_image = bridge.cv2_to_imgmsg(image, encoding="rgb8")
    np.testing.assert_array_equal(bridge.imgmsg_to_cv2(ros_image, "rgb8"), image)

    features = {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channel"],
        },
        "exterior_image_2_left": {
            "dtype": "image",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channel"],
        },
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
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },
    }

    with tempfile.TemporaryDirectory(prefix="jazzy-lerobot-") as tmp:
        root = Path(tmp) / "dataset"
        repo_id = "smoke/franka_jazzy"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            robot_type="panda",
            fps=30,
            features=features,
            use_videos=False,
            image_writer_threads=2,
            video_backend="pyav",
        )
        for index in range(3):
            dataset.add_frame(
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
                    "task": "synthetic Jazzy test",
                }
            )
        dataset.save_episode()
        dataset.finalize()

        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        assert info["codebase_version"] == "v3.0", info
        assert info["total_episodes"] == 1, info
        assert info["total_frames"] == 3, info

        replay = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
        assert len(replay) == 3
        sample = replay[1]
        assert tuple(sample["joint_position"].shape) == (7,)
        assert tuple(sample["ee_pose"].shape) == (6,)
        assert tuple(sample["actions"].shape) == (8,)
        assert tuple(sample["exterior_image_1_left"].shape) == (3, 64, 64)

        resumed = LeRobotDataset.resume(
            repo_id=repo_id, root=root, video_backend="pyav"
        )
        resumed.add_frame(
            {
                "exterior_image_1_left": image,
                "exterior_image_2_left": image,
                "wrist_image_left": image,
                "joint_position": np.zeros(7, dtype=np.float32),
                "ee_pose": np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
                "gripper_position": np.array([0.5], dtype=np.float32),
                "actions": np.ones(8, dtype=np.float32),
                "task": "synthetic Jazzy resume test",
            }
        )
        resumed.save_episode()
        resumed.finalize()
        assert (
            len(LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")) == 4
        )
        print("PASS: Jazzy rclpy/cv_bridge and LeRobot 0.6.1 create, read, resume")


if __name__ == "__main__":
    main()
