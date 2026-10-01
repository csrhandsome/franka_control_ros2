# Agent instructions for this repository

## Runtime: Docker is the default

- This project's ROS 2 Humble runtime is in the `franka_humble` Docker image. Run robot control, VR collection, camera access, ROS inference, and their runtime checks in the container. Do not suggest running `python vr_collect.py`, `python inference.py`, `ros2 launch`, or ROS dependent tests directly on the host.
- Use `ros2_ws/docker/franka_humble/scripts/compose_safe.sh` for processes that only need ROS topics and the mounted project. Use `compose_devices.sh` for the ROS bringup that accesses cameras or other USB devices. Both wrappers load the project's Compose overlays and `.env`.
- Run commands from the repository root. The container mounts this repository at `/workspace/data_collect` and uses its own Python environment at `/opt/uv/venv`; the host `.venv` is not the deployment environment.
- Build the Docker image and ROS overlay as described in `README.md` before first use. Keep `ros2_ws/docker/franka_humble/.env` configured for the host and robot.

## Current collection entry point

- Start the full collection workflow with `./collect.sh` (or `./collect.sh config/collect/<robot>.yaml`). This script starts and stops the ROS bringup, VR publisher, and `vr_collect.py` containers together. Treat it as the normal collection command; the three separate container commands in `README.md` are for debugging.
- Before live collection, check `config/collect/franka.yaml`: the checked-in defaults use fake hardware, disable cameras, and disable dataset logging. Set the intended robot, camera, and recording values explicitly.

## Inference and training

- Run `inference.py` and `vr_hitl_inference.py` in the Humble container, after the ROS bringup container is running. The plain inference command is documented in the `Plain 30 Hz inference` section of `README.md` and uses `compose_safe.sh`.
- The `openpi-force` training job and policy WebSocket server run separately in a GPU environment. The ROS inference container connects to that server through `config/inference/franka.yaml` or its CLI host/port options; it does not host the model.
- Collection records robot state and actions at about 100 Hz and camera frames at 30 Hz. Training aligns these streams to 30 Hz; plain policy inference also runs at 30 Hz. Do not infer a 100 Hz inference loop from the collection rate.

## Jazzy + LeRobot 0.6.1 experiment

- `ros2_ws/docker/franka_jazzy/` is an isolated ROS 2 Jazzy / Python 3.12 image for checking the latest LeRobot dataset API. It is not the deployment runtime: Humble remains the collection and inference environment, and neither the Humble configuration nor the training repository's LeRobot version is affected. See that directory's `README.md`.
- Its synthetic smoke test checks the Jazzy `rclpy` and `cv_bridge` imports, then creates, reads back, and resumes a LeRobot v3 dataset with the collector's core feature schema. It does not connect to a robot. The image rebuilds `cv_bridge` 4.1.0 against NumPy 2 because Jazzy's prebuilt extension uses the NumPy 1 ABI.
- Re-run it from the repository root with `bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy python ros2_ws/docker/franka_jazzy/scripts/smoke_test.py` (build the image once first with the same wrapper's `build`). The wrapper reuses the host settings in `franka_humble/.env` but has its own image, Compose project, Python environment, and temporary test dataset; the project mount is read-only, and there is no USB device mapping or `franka_ros2` bringup.
- The current collector cannot switch to Jazzy yet: `tests/docker_recording_smoke.py` still imports `lerobot.common.datasets`, which LeRobot 0.6.1 removed. Do not present the Jazzy image as a drop-in replacement until that migration is done.

When giving startup, test, or deployment instructions for this repository, include the appropriate Docker wrapper commands without waiting for the user to remind you.
