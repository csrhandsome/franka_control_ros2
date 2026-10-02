"""Manual ROS arm commands. Run with compose_safe.sh in the Humble container."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from control.robot_config import load_mapping
from control.robotic_arm_controller_ros import RoboticArmControlerRos
from control.util.robot import move_robot_to_start_pose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--use-fake-hardware", action="store_true", default=None)
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument(
        "command", choices=("status", "gripper-open", "gripper-close", "move-start")
    )
    args = parser.parse_intermixed_args(argv)
    with RoboticArmControlerRos(
        config_path=args.config,
        use_fake_hardware=args.use_fake_hardware,
        robot_type=args.robot_type,
    ) as arm:
        if args.command == "status":
            pprint(arm.get_status())
            pprint(arm.get_capabilities())
        elif args.command == "gripper-open":
            if not arm.gripper_open():
                raise RuntimeError("Gripper move failed")
        elif args.command == "gripper-close":
            if not arm.gripper_close():
                raise RuntimeError("Gripper grasp failed")
        elif args.command == "move-start":
            workflow = load_mapping(
                Path(__file__).resolve().parents[1] / "config/common/franka.yaml"
            )
            move_robot_to_start_pose(arm, workflow["robot"]["start_joint_position"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
