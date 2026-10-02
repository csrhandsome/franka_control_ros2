# Agent instructions for this repository

## Runtime: Docker is the only robot-control environment

- Docker is the only permitted environment for operating the robot. All robot control, ROS bringup, hardware connectivity tests, collection, and ROS inference must run inside the project's Humble Docker container; never run them on the host or in the host `.venv`.
- This project's ROS 2 Humble runtime is in the `franka_humble` Docker image. Run robot control, VR collection, camera access, ROS inference, and their runtime checks in the container. Do not suggest running `python vr_collect.py`, `python inference.py`, `ros2 launch`, or ROS dependent tests directly on the host.
- Use `ros2_ws/docker/franka_humble/scripts/compose_safe.sh` for processes that only need ROS topics and the mounted project. Use `compose_devices.sh` for the ROS bringup that accesses cameras or other USB devices. Both wrappers load the project's Compose overlays and `.env`.
- Run commands from the repository root. The container mounts this repository at `/workspace/data_collect` and uses its own Python environment at `/opt/uv/venv`; the host `.venv` is not the deployment environment.
- Build the Docker image and ROS overlay as described in `README.md` before first use. Keep `ros2_ws/docker/franka_humble/.env` configured for the host and robot.

## System-commanded robot motion

- All system-commanded robot motion must use the `RoboticArmControlerRos` class in `control/robotic_arm_controller_ros.py`, running inside the Humble Docker container.
- C++ ROS controller and driver changes, and the Docker/overlay builds needed to integrate them, are permitted when required to implement or repair robot control. Keep the existing public Python interfaces unchanged unless the user explicitly requests an API change.
- C++ controllers must operate through the ROS control hardware interfaces; the compatible Franka ROS driver calls libfranka. Do not introduce standalone movement programs, call libfranka directly from a test or script, or bypass `RoboticArmControlerRos` by publishing raw motion targets or sending movement actions from a test or script.
- Test robot motion through this class's existing public Python interfaces inside Humble Docker. Verify controller/driver compatibility with the active Panda/libfranka runtime before real motion; use isolated non-hardware checks first, then small-motion checks on the actual robot when authorized. Fake hardware checks are optional and must not replace requested real-hardware tests. Report unavailable interfaces and failed tests honestly rather than disabling required functionality or weakening test assertions.

## Current collection entry point

- Start the full collection workflow with `./collect.sh` (or `./collect.sh config/collect/<robot>.yaml`). This script starts and stops the ROS bringup, VR publisher, and `vr_collect.py` containers together. Treat it as the normal collection command; the three separate container commands in `README.md` are for debugging.
- Before live collection, check `config/collect/franka.yaml`: the checked-in configuration uses Panda, real hardware, ROS cameras, and dataset logging. Shared values come from `config/common/franka.yaml`; workflow values override them. Verify the intended values explicitly. Collection enables EE/joint streaming through its `motion` section. Driver compatibility must be verified separately; do not disable the required interfaces to represent an unverified driver.

## Inference and training

- Run `inference.py` and `inference_hitl.py` in the Humble container, after the ROS bringup container is running. The plain inference command is documented in the `Plain 30 Hz inference` section of `README.md` and uses `compose_safe.sh`.
- The `openpi-force` training job and policy WebSocket server run separately in a GPU environment. The ROS inference container connects to that server through `config/inference/franka.yaml` or its CLI host/port options; it does not host the model.
- Collection records robot state and actions at about 100 Hz and camera frames at 30 Hz. Training aligns these streams to 30 Hz; plain policy inference also runs at 30 Hz. Do not infer a 100 Hz inference loop from the collection rate.

## Jazzy + LeRobot 0.6.1 experiment

- `ros2_ws/docker/franka_jazzy/` is an isolated ROS 2 Jazzy / Python 3.12 image for checking the latest LeRobot dataset API. It is not the deployment runtime: Humble remains the collection and inference environment, and neither the Humble configuration nor the training repository's LeRobot version is affected. See that directory's `README.md`.
- Its synthetic smoke test checks the Jazzy `rclpy` and `cv_bridge` imports, then creates, reads back, and resumes a LeRobot v3 dataset with the collector's core feature schema. It does not connect to a robot. The image rebuilds `cv_bridge` 4.1.0 against NumPy 2 because Jazzy's prebuilt extension uses the NumPy 1 ABI.
- Re-run it from the repository root with `bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy python ros2_ws/docker/franka_jazzy/scripts/smoke_test.py` (build the image once first with the same wrapper's `build`). The wrapper reuses the host settings in `franka_humble/.env` but has its own image, Compose project, Python environment, and temporary test dataset; the project mount is read-only, and there is no USB device mapping or `franka_ros2` bringup.
- The current collector cannot switch to Jazzy yet: `tests/docker_recording_smoke.py` still imports `lerobot.common.datasets`, which LeRobot 0.6.1 removed. Do not present the Jazzy image as a drop-in replacement until that migration is done.

When giving startup, test, or deployment instructions for this repository, include the appropriate Docker wrapper commands without waiting for the user to remind you.

## Configuration inheritance

- `control.robot_config.load_mapping` resolves `extends` relative to each YAML file and recursively merges mappings. Local values override inherited values; lists replace rather than append. Use this loader instead of raw `yaml.safe_load` for workflow and Python motion profiles.
- `config/common/franka.yaml` contains shared Panda workflow values and normal-workflow joint trajectory limits. `config/planer/panda.yaml` provides self-contained motion defaults; `control.motion_config.workflow_motion_config` validates workflow `motion` overrides and each workflow passes them to `RoboticArmControlerRos`. Preserve the nested `motion` mapping when flattening collection settings. Online target limits live in `ee.streaming.limits` / `joint.streaming.limits`; top-level motion `limits` apply only to finite trajectories. ROS-native trajectory controller parameters live at `ros2_ws/src/data_collect_franka/config/panda_trajectory_controller.yaml` and do not use inheritance syntax.
