#!/usr/bin/env python3
"""Franka ROS 2 control CLI / import facade.

Runs in the Humble Docker image, from the repository root::

    python -m control.franka_ros2_control status

Does not silently command a real robot; missing topics are errors unless
``--use-fake-hardware`` is set.
"""

from __future__ import annotations

import argparse
import sys

from control.robotic_arm_controller_ros import RoboticArmControlerRos


class FrankaROS2Control:
    """Thin importable facade around RoboticArmControlerRos."""

    def __init__(self, **kwargs) -> None:
        self.arm = RoboticArmControlerRos(**kwargs)

    def status(self) -> str:
        return self.arm.status_text()

    def hold(self) -> None:
        self.arm.hold_current_pose()

    def gripper_open(self) -> None:
        self.arm.gripper_open()

    def gripper_close(self) -> None:
        self.arm.gripper_close()

    def move_start(self) -> None:
        self.arm.move_to_start()

    def cleanup(self) -> None:
        self.arm.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Franka ROS 2 control CLI (Humble container only)."
    )
    parser.add_argument(
        "--use-fake-hardware",
        action="store_true",
        help="Treat missing real-robot topics as non-fatal.",
    )
    parser.add_argument("--robot-type", default="fr3")
    parser.add_argument(
        "command",
        choices=["status", "hold", "gripper-open", "gripper-close", "move-start"],
        help="status | hold | gripper-open | gripper-close | move-start",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_intermixed_args(argv)
    control = FrankaROS2Control(
        use_fake_hardware=bool(args.use_fake_hardware),
        robot_type=str(args.robot_type),
    )
    try:
        if args.command == "status":
            print(control.status())
            return 0
        if args.command == "hold":
            control.hold()
            print("[FrankaROS2] hold commanded")
            return 0
        if args.command == "gripper-open":
            control.gripper_open()
            print("[FrankaROS2] gripper-open requested")
            return 0
        if args.command == "gripper-close":
            control.gripper_close()
            print("[FrankaROS2] gripper-close requested")
            return 0
        if args.command == "move-start":
            control.move_start()
            print("[FrankaROS2] move-start commanded")
            return 0
        raise SystemExit(f"unknown command {args.command}")
    finally:
        control.cleanup()


if __name__ == "__main__":
    sys.exit(main())
