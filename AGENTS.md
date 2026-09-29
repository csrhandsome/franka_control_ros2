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

When giving startup, test, or deployment instructions for this repository, include the appropriate Docker wrapper commands without waiting for the user to remind you.
