# Franka ROS 2 Humble container

This is a headless, Linux-Docker configuration for an FR3-first setup. It pins:

- `franka_ros2` `v2.5.1` (Humble)
- `libfranka` `0.20.4` via the official `dependency.repos`
- `franka_description` `2.8.1`

Python in this image is managed with `uv`. The venv lives at `/opt/uv/venv` inside the image, not in `data_collect/.venv`. ROS Humble packages stay on the system interpreter and the venv uses `--system-site-packages` so `import rclpy` still works. The host project's Python 3.11 `pyproject.toml` is not used here.

Use the wrapper so the uv overlay and the fixed entrypoint are both applied:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble bash
```

## First use

1. Refresh the Docker group in a new login shell, or use `sg docker -c '<command>'` for the current shell. Docker was available through `sg docker` when this configuration was created.
2. Copy `.env.example` to `.env` and verify `FRANKA_ROBOT_IP` in Desk. The copied address comes only from this repository's existing controller default and is **not** a network probe result.
3. The Humble + `franka_ros2` image `data-collect/franka-ros2-humble:2.5.1` should already exist. Build the thin uv layer (does not recompile Franka):

   ```bash
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh build
   ```

4. Verify the uv venv and ROS 2 without a robot or FCI connection:

   ```bash
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble uv-env-smoke-test
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_recording_smoke.py
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_collect_pipeline_smoke.py
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble fake-hardware-smoke-test
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble bash ros2_ws/docker/franka_humble/scripts/build_overlay.sh
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble bash ros2_ws/docker/franka_humble/scripts/adapter_fake_hardware_test.sh
   ```

5. When the physical direct Ethernet link is up, run the read-only host-network test. It checks the selected NIC, route and ICMP only; it cannot move the arm:

   ```bash
   bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble verify-fci-network
   ```

## Current network finding

At configuration time `enp3s0` existed but reported `NO-CARRIER` and had no IPv4 address, so no robot ping, FCI handshake or UDP jitter test was possible. Connect/power the robot/control box and configure the correct static host address on that NIC before step 5. Do not reuse the Wi-Fi route for the robot.

## Safety boundary

The verification commands above run synthetic inputs, fake hardware, or ICMP only. They do **not** execute a real-arm launch. Before following the live collection commands in the root README, verify the Robot System version, FCI feature and Desk FCI activation; ensure no `panda_py` process is connected; and use the project migration plan's read-only and low-risk-motion stages.

## Fake hardware notes

`cartesian_pose_target_controller` can **load** on fake hardware but **activate** may fail because fake hardware does not provide cartesian pose state. Use `joint_position_target_controller` as the fake fallback. Run `python -m control.franka_ros2_control --use-fake-hardware status` from `/workspace/data_collect`; it should still print 7 joints.

## Recording in the container

The uv image contains the pinned LeRobot package and CPU-only PyTorch. It writes
LeRobot episodes directly; the collection container does not need GPU access.
The `docker_recording_smoke.py` command above writes and reloads a three-frame
synthetic episode in a temporary directory. It verifies the Python/ROS imports,
Parquet data, and image decoding without robot motion or cameras. The current
feature schema stores images in Parquet rather than MP4 files.
`docker_collect_pipeline_smoke.py` runs `vr_collect.py` with synthetic robot,
camera, and VR inputs and checks its saved actions, sync metadata, and episode.

For live recording, enable `dataset.enable_logging`, set `camera.camera_backend`
to `ros`, and follow the three-terminal commands in the root README. The ROS
bringup with cameras uses the device-enabled Compose wrapper; `vr_collect.py`
uses the safe wrapper after robot, VR, and camera topics are ready. Data is
written through the mounted project tree under `data/<repo_id>_<date>/`.
