# Jazzy + LeRobot 0.6.1 experiment

This is an isolated ROS 2 Jazzy / Python 3.12 image for checking the latest
LeRobot dataset API. It does not replace the Humble image or start a robot.
The synthetic smoke test checks ROS Python and `cv_bridge` imports, creates a
LeRobot v3 dataset using the collector's core feature schema, reads it, then
resumes it for another episode. The image rebuilds `cv_bridge` 4.1.0 against
NumPy 2 because Jazzy's prebuilt extension uses the NumPy 1 ABI.

From the repository root:

```bash
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh build
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python ros2_ws/docker/franka_jazzy/scripts/smoke_test.py
```

The wrapper reuses the host settings in `franka_humble/.env`, but its image,
Compose project, Python environment, and temporary test dataset are separate.
The project mount is read-only. It has no USB device mapping and does not launch
`franka_ros2`.

This checks the new data format and ROS Python compatibility. The existing
`tests/docker_recording_smoke.py` still imports `lerobot.common.datasets`, an
API removed in 0.6.1, so the current collector needs a separate migration
before the Jazzy image can replace the Humble collection workflow.
