"""Public Franka ROS 2 arm API. Use inside the Humble Docker container.

Construction reads configuration; connect() allocates ROS resources without
commanding motion. Online targets and blocking trajectories are separate modes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from control.robotic_arm_ros_types import (
    CommandReceipt,
    ControlSpace,
    MotionResult,
    RobotCapabilities,
    RobotState,
    RobotStatus,
)

if TYPE_CHECKING:
    from rclpy.node import Node

from control._ros import gripper, lifecycle, motion, state


class RoboticArmControlerRos:
    """Measured state, explicit control modes, and gripper operations.

    Coordinates use the configured base frame, metres, radians, and xyzw
    quaternions. Config files use the project's recursive YAML inheritance.
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        node: Node | None = None,
        use_fake_hardware: bool | None = None,
        robot_type: str | None = None,
        command_timeout_s: float | None = None,
        motion_config: dict | None = None,
    ) -> None:
        """Read the motion profile; optional overrides support existing workflows."""
        lifecycle._initialize(
            self,
            config_path=config_path,
            node=node,
            use_fake_hardware=use_fake_hardware,
            robot_type=robot_type,
            command_timeout_s=command_timeout_s,
            motion_config=motion_config,
        )

    def connect(self, *, timeout_s: float | None = None) -> RobotStatus:
        """Connect and wait for fresh joints. Does not activate a controller."""
        return lifecycle._connect(self, timeout_s=timeout_s)

    def close(self) -> None:
        """Cancel motion, stop streams, and release owned ROS resources; idempotent."""
        lifecycle._close(self)

    def __enter__(self) -> RoboticArmControlerRos:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception as close_error:
                # Preserve the original exception while making cleanup failure visible.
                import warnings

                warnings.warn(f"Robot cleanup failed: {close_error}", RuntimeWarning)

    def get_state(self, *, required: Sequence[str] = ("joints", "ee")) -> RobotState:
        """Copy measured state; required sources must exist and be fresh.

        required=() permits diagnostics, with missing values represented by None.
        """
        return state._get_state(self, required=required)

    def wait_ready(
        self,
        *,
        required: Sequence[str] = ("joints", "ee"),
        timeout_s: float | None = None,
    ) -> RobotState:
        """Wait for fresh required sources after connect()."""
        return state._wait_ready(self, required=required, timeout_s=timeout_s)

    def get_status(self) -> RobotStatus:
        """Inspect lifecycle, active mode, controllers, and measured state."""
        return state._get_status(self)

    def get_capabilities(self) -> RobotCapabilities:
        """Inspect configured and available capabilities without activating motion."""
        return state._get_capabilities(self)

    def start_stream(self, space: ControlSpace) -> None:
        """Activate one online mode and publish its current measured target."""
        lifecycle._start_stream(self, space)

    def send_joint_target(self, joints: Sequence[float]) -> CommandReceipt:
        """Publish an absolute 7D joint target in joint mode; does not wait for arrival."""
        return motion._send_joint_target(self, joints)

    def send_ee_target(
        self, position: Sequence[float], quaternion_xyzw: Sequence[float]
    ) -> CommandReceipt:
        """Publish an absolute base-frame pose in EE mode; does not wait for arrival."""
        return motion._send_ee_target(self, position, quaternion_xyzw)

    def stop_stream(self, *, timeout_s: float | None = None) -> None:
        """Publish a measured hold target, confirm stopped feedback, then deactivate.

        A failure raises and does not report that the robot has stopped.
        """
        lifecycle._stop_stream(self, timeout_s=timeout_s)

    def move_joints(self, target: Sequence[float], *, duration_s: float = 4.0) -> MotionResult:
        """Execute a validated joint move; block through measured completion."""
        return motion._move_joints(self, target, duration_s=duration_s)

    def move_ee(
        self,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
        *,
        duration_s: float = 4.0,
    ) -> MotionResult:
        """Execute a validated Cartesian move; block through measured completion."""
        return motion._move_ee(self, position, quaternion_xyzw, duration_s=duration_s)

    def execute_joint_trajectory(
        self,
        points: Sequence[dict],
        *,
        on_sample: Callable[[RobotState], None] | None = None,
    ) -> MotionResult:
        """Block on joint waypoints with time_s, positions, velocities, accelerations.

        on_sample runs on this execution thread and may record measured state.
        """
        return motion._execute_trajectory(self, points, space="joint", on_sample=on_sample)

    def execute_ee_trajectory(
        self,
        points: Sequence[dict],
        *,
        on_sample: Callable[[RobotState], None] | None = None,
    ) -> MotionResult:
        """Block on waypoints with time_s, position, quaternion_xyzw."""
        return motion._execute_trajectory(self, points, space="ee", on_sample=on_sample)

    def cancel_motion(self, *, timeout_s: float | None = None) -> MotionResult:
        """Cancel a trajectory from another thread and wait for confirmed completion.

        Returns idle when no trajectory is running. A timeout leaves cancellation
        requested and raises; it does not imply motion has stopped.
        """
        return motion._cancel_motion(self, timeout_s=timeout_s)

    def wait_until_stopped(self, *, timeout_s: float = 2.0) -> None:
        """Wait for fresh measured joint velocities to settle; raise on timeout."""
        state._wait_until_stopped(self, timeout_s=timeout_s)

    @property
    def gripper_busy(self) -> bool:
        """Whether a gripper command is queued or executing."""
        return gripper._busy(self) if self._connected else False

    def gripper_open(
        self,
        width: float = 0.05,
        speed: float = 0.2,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Move to width in metres; wait=False returns local queue acceptance."""
        return gripper._gripper_open(self, width, speed, wait=wait, timeout=timeout)

    def gripper_close(
        self,
        width: float = 0.0,
        speed: float = 0.2,
        force: float = 60.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        """Grasp with metre widths, m/s speed, and newton force."""
        return gripper._gripper_close(
            self,
            width,
            speed,
            force,
            epsilon_inner,
            epsilon_outer,
            wait=wait,
            timeout=timeout,
        )

    def wait_gripper(self, timeout: float | None = None) -> None:
        """Wait for the current gripper command using the existing API."""
        gripper._wait_gripper(self, timeout=timeout)

    def stop_gripper(self, *, timeout_s: float | None = None) -> None:
        """Request the existing gripper stop service and check its response."""
        gripper._stop_gripper(self, timeout_s=timeout_s)
