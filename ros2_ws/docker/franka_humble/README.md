## Panda / FER runtime override

The active `Dockerfile.uv` now builds a persistent Panda driver in the image:
LCAS/franka_arm_ros2 commit `6867bde68970104118d04ae853ddded15bf857fb`
and libfranka 0.9.2 commit `f3b8d775a9c847cab32684c8a316f67867761674`.
They are installed under `/opt/panda_ws` and `/opt/panda_libfranka`.
The entrypoint prefers them over the inherited FR3 driver and does not load
its incompatible collection overlay. No runtime compilation is required.
The inherited base tag remains 2.5.1; it does not describe the active Panda driver.

From the repository root, after preparing the base image and `.env` below:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh build franka_humble
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble bash -c 'exec ros2 launch franka_bringup franka.launch.py robot_ip:=$FRANKA_ROBOT_IP load_gripper:=true use_rviz:=false'
```

Enable FCI and unlock brakes in Desk before launch. This verifies bringup;
`collect.sh` is not yet ported to the Panda driver's older controller interfaces.
The driver permits a non-RT kernel via `RealtimeConfig::kIgnore` and emits a warning.

---

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
