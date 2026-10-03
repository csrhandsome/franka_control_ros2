## Panda / FER runtime override

The active `Dockerfile.uv` now builds a persistent Panda driver in the image:
LCAS/franka_arm_ros2 commit `6867bde68970104118d04ae853ddded15bf857fb`
and libfranka 0.9.2 commit `f3b8d775a9c847cab32684c8a316f67867761674`.
They are installed under `/opt/panda_ws` and `/opt/panda_libfranka`.
The entrypoint sources the Panda underlay, then the rebuilt project overlay via
`local_setup.bash`, without importing the inherited FR3 workspace.
The image applies `patches/panda-control-startup.patch` to seed position commands
from `q_d` for joints and `O_T_EE_c` for EE motion, matching the first libfranka
control callback's last commanded pose, and fix state-broadcaster contention.
No runtime compilation is required after the image and overlay builds.
The inherited base tag remains 2.5.1; it does not describe the active Panda driver.

From the repository root, after preparing the base image and `.env` below:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh build franka_humble
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble bash ros2_ws/docker/franka_humble/scripts/build_overlay.sh
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble bash -c 'exec ros2 launch data_collect_franka franka_data_collect.launch.py robot_type:=panda robot_ip:="$FRANKA_ROBOT_IP" use_fake_hardware:=false load_gripper:=true start_cameras:=false'
```

Enable FCI and unlock brakes in Desk before launch. Project bringup loads the
joint streaming, joint trajectory and EE target controllers inactive; the public
Python class activates them when needed. Run `tests/docker_arm_connection_test.py`
through `compose_safe.sh` for measured live motion and cleanup checks.
`./collect.sh` remains the complete collection entry point with cameras and VR.
The driver permits a non-RT kernel via `RealtimeConfig::kIgnore` and emits a warning.

---

# Franka ROS 2 Humble container

The inherited base image contains the following FR3 packages; the active Panda
runtime above overrides them and must not link to these versions:

- `franka_ros2` `v2.5.1` (Humble)
- `libfranka` `0.20.4` via the official `dependency.repos`
- `franka_description` `2.8.1`

Python in this image is managed with `uv`. The venv lives at `/opt/uv/venv` inside the image, not in `data_collect/.venv`. ROS Humble packages stay on the system interpreter and the venv uses `--system-site-packages` so `import rclpy` still works. The image's `uv` project is kept under `/opt/uv/project`, independent of the host source tree.

Use the wrapper so the uv overlay and the fixed entrypoint are both applied:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble bash
```

## First use

1. Refresh the Docker group in a new login shell, or use `sg docker -c '<command>'` for the current shell. Docker was available through `sg docker` when this configuration was created.
2. Copy `.env.example` to `.env` and verify `FRANKA_ROBOT_IP` in Desk. The copied address comes only from this repository's existing controller default and is **not** a network probe result.
3. The Humble + `franka_ros2` image `data-collect/franka-ros2-humble:2.5.1` should already exist. Build the uv layer and pinned Panda driver:

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

## Camera device access

Use `compose_devices.sh` for cameras. It binds the host USB directory, explicitly
maps each current `/dev/video*` character device, and adds its host numeric GID
alongside the image's `video`/`plugdev` groups. `compose_dh5.sh` uses the same
camera mapping and adds its serial device. No privileged mode or image rebuild
is required for these runtime mappings.

Connect the cameras before creating the container. USB directory updates are
visible in a running container, but video mappings are fixed when it is created;
recreate the camera container through the wrapper after unplugging/reconnecting
a camera. Restarting an old container does not regenerate its mappings.

From the repository root, check SDK enumeration without starting the robot:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm --no-deps -T franka_humble rs-enumerate-devices -s
```

Run `rs-enumerate-devices` without `-s` to also see each camera's
`Usb Type Descriptor`. Use the wrapper's generated Compose mapping rather than
`docker compose run --device`, which is not a supported Compose run option.

## Current network finding

At configuration time `enp3s0` existed but reported `NO-CARRIER` and had no IPv4 address, so no robot ping, FCI handshake or UDP jitter test was possible. Connect/power the robot/control box and configure the correct static host address on that NIC before step 5. Do not reuse the Wi-Fi route for the robot.

## Safety boundary

The verification commands above run synthetic inputs, fake hardware, or ICMP only. They do **not** execute a real-arm launch. Before following the live collection commands in the root README, verify the Robot System version, FCI feature and Desk FCI activation; ensure no `panda_py` process is connected; and use the project migration plan's read-only and low-risk-motion stages.

## Fake hardware notes

Optional Panda fake bringup adds Cartesian GPIO interfaces to `GenericSystem`
and publishes their feedback as the EE pose. This checks ROS wiring, not Panda
dynamics, inverse kinematics or real-time performance. The adapter test uses
the same public Python class and requires all arm capabilities; it does not
accept an unavailable EE controller as a successful test.

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
