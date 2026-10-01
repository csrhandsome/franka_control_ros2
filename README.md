# Franka ROS 2 collection and Human in the Loop control

The repository root holds three entry points: `vr_collect.py` (VR teleoperation
demonstrations), `vr_hitl_inference.py` (policy rollouts with optional VR
takeover and recording), and `inference.py` (plain 30 Hz policy inference,
without recording). Everything else lives under `control/`, `config/`,
`data_analysis/`, `scripts/`, or `ros2_ws/`.

## 在 Docker 中采集 LeRobot 数据

采集入口统一为 `vr_collect.py`。通过 `dataset.lerobot_format: v2 | v3`
选择数据格式，也可用 `--lerobot-format` 临时覆盖：v2 对应 Humble 镜像中的
LeRobot 0.1，v3 对应 Jazzy 镜像中的 LeRobot 0.6.1。Jazzy 的合成记录验证
命令和当前运行范围见
[`ros2_ws/docker/franka_jazzy/README.md`](ros2_ws/docker/franka_jazzy/README.md)。

采集容器使用 ROS 2 Humble、固定版本的 LeRobot 和 CPU 版 PyTorch，直接写入
LeRobot 数据集；模型推理和训练可在本机 GPU 环境运行。以下命令均在仓库根目录执行。
首次使用时，从 `ros2_ws/docker/franka_humble/.env.example` 复制 `.env`，并填写
实际的机器人 IP、网卡及用户 UID/GID。

```bash
test -f ros2_ws/docker/franka_humble/.env || cp ros2_ws/docker/franka_humble/.env.example ros2_ws/docker/franka_humble/.env
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh build franka_humble
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble uv-env-smoke-test
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_dataset_adapter_smoke.py --format v2
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_recording_smoke.py
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_collect_pipeline_smoke.py
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble bash ros2_ws/docker/franka_humble/scripts/build_overlay.sh
```

采集检查验证统一数据集接口、LeRobot episode 的写入/回读，以及
`vr_collect.py` 的完整录制流程；它们使用合成输入和临时目录。接入真实设备前，修改
`config/collect/franka.yaml`：`dataset.enable_logging: true`、
`camera.camera_backend: ros`、`robot.use_fake_hardware: false`，将
`dataset.date` 设为本次采集标识，并核对两路相机序列号和话题。配置完成且
镜像与 ROS overlay 已构建后，在仓库根目录运行 `./collect.sh`，一次启动
控制器、VR 话题和录制进程；按 Ctrl+C 会停止对应容器。脚本读取
`config/collect/franka.yaml`，也可传入仓库内的其他采集配置路径。当前 YAML
启用真实硬件、相机和数据保存，运行前必须逐项核对。`dataset.action_space: ee`
保存 7D 的末端位姿加夹爪动作；改为 `joint` 则保存 8D 的关节角加夹爪动作。
切换动作空间后请更换 `dataset.date`。
`./collect.sh` 目前只启动 Humble 运行环境，因此要求 `dataset.lerobot_format: v2`；
Jazzy 镜像当前只完成合成记录验证，尚无机器人 bringup。

需要分别调试三个进程时，也可以在三个终端运行下面的命令：

```bash
# 终端 1：机器人控制器与两路 RealSense；此容器需要 USB 设备。
bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm franka_humble \
  bash -c 'ros2 launch data_collect_franka franka_data_collect.launch.py \
    robot_type:=fr3 robot_ip:="$FRANKA_ROBOT_IP" use_fake_hardware:=false \
    load_gripper:=true start_cameras:=true \
    external_serial:=825412070292 wrist_serial:=825412070487'

# 终端 2：VR 控制器 ROS 话题。
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble \
  python -m teleop_xr.ros2 --mode teleop

# 终端 3：开始录制。左手 Y 保存成功 episode，X 保存失败 episode。
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble \
  python vr_collect.py --config config/collect/franka.yaml
```

数据写入仓库的 `data/<repo_id>_<date>/`。上述真实设备流程需要先确认机器人
FCI、控制器、两路图像话题和 VR 话题均已就绪；本仓库的自动检查只覆盖合成
采集与假硬件。Docker 的详细构建和假硬件检查见
[Humble 容器说明](ros2_ws/docker/franka_humble/README.md)。

## Multi-rate collection

`vr_collect.py` samples the latest VR pose and sends a
robot target at the configured `control.control_frequency` (100 Hz). The ROS
controller still runs at 1 kHz. A 100 Hz loop does not make the headset produce
100 new poses per second; repeated `vr_pose_seq` values identify reused input.

The two RealSense nodes run at 640×480×30. The camera adapter crops and resizes
frames in a background thread. Only a pair in which **both** cameras have a new
frame is inserted into LeRobot. That dataset has `fps=30`; every frame records
both a 6D `ee_pose` and 7D `joint_position`. `dataset.action_space` selects
whether its `actions` feature contains 6D EE pose plus gripper (7D) or seven
joint angles plus gripper (8D). A frame action is measured at the next fresh
camera pair. Image insertion runs in a separate thread.

Each saved episode also has two files in the dataset root:

- `episode_NNNNNN.actions.jsonl`: the 100 Hz robot/VR stream. Each row has a
  monotonic sample time, measured joint and EE pose, commanded EE pose, VR pose
  and receive sequence, and the next sampled joint, EE and gripper state
  (`action_joint_position`, `action_ee_pose`, `action_gripper_position`). The last row uses the
  state read when the episode is finalized.
- `episode_NNNNNN.sync.json`: per-camera-frame times plus the action stream
  filename, configured rates, action space and one-frame target offset, observed action/camera pair rates, VR reuse count,
  and the episode's `success` label.

A left-controller button ends the current episode and returns the arm to the
start pose: `Y` saves it with `success: true` and `X` saves it with
`success: false`. Endings that nothing graded — the maximum duration, or
Ctrl+C — leave `success` at `null`.

`openpi-force` trains directly from the LeRobot `actions` feature. The 100 Hz
trace and camera sync files remain available for diagnostics. From the
`openpi-force` GPU environment, run
`bash scripts/train_franka_ros_30hz.sh <collector-data-dir> <repo-id> <experiment>`;
the script selects EE or joint training from the recorded action shape.

The current collection YAML uses real hardware, ROS cameras and dataset logging.
Verify its robot, camera, recording, date and action-space settings before a run.

## Plain 30 Hz inference

`inference.py` runs the `pi0_base_ee_ros_30hz` checkpoint trained in
`openpi-force`. It sends 224×224 front/wrist images, measured 6D EE pose, the
logical 0/1 gripper state, and the task prompt to the policy server. The server
returns 16 future absolute EE pose targets and logical gripper commands at 30 Hz.
The inference loop samples state and publishes a target at 30 Hz, requesting a
new chunk after eight model steps. This entry point does not save observations
or episodes. Before starting the cameras or arm controller, it sends five blank
observations to warm the policy server, validates each returned action chunk,
and discards those actions. Inference starts only after all five requests finish.

Start the trained checkpoint in the `openpi-force` repository (replace the
checkpoint directory with the one you trained):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_base_ee_ros_30hz \
  --policy.dir=<checkpoint-directory> --port=8001
```

Set the server host, camera topics, and robot settings in
`config/inference/franka.yaml`. Run the ROS bringup and inference in two
Humble containers from this repository root (the policy server above runs in
the `openpi-force` GPU environment). For a real FR3, set
`robot.use_fake_hardware: false` in the inference YAML and use the matching
bringup argument:

```bash
# Terminal 1: robot, gripper, and both cameras (USB devices required here).
bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm franka_humble \
  bash -c 'exec ros2 launch data_collect_franka franka_data_collect.launch.py \
    robot_type:=fr3 robot_ip:="$FRANKA_ROBOT_IP" use_fake_hardware:=false \
    load_gripper:=true start_cameras:=true joint_state_rate:=30 \
    external_serial:=825412070292 wrist_serial:=825412070487'

# Terminal 2: inference client (shares host networking and the ROS domain).
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python inference.py --prompt "Place the object into the basket"
```

The checked-in inference config uses fake hardware and requires live ROS
cameras. The `policy_server.host` value must point to the GPU server; because
the containers use host networking, `127.0.0.1` works when that server runs on
the same machine. The ROS overlay must be built before running these commands.
`--move-to-start` moves to the configured
start joints first; otherwise inference starts from the current pose. The
default run lasts 600 control ticks (20 seconds); `--max-steps 0` runs until
Ctrl+C. The policy server must expose the training config's 30 Hz metadata.

## Human in the Loop inference and recording

`vr_hitl_inference.py` runs policy inference with optional VR takeover and records the rollout. The
servo loop runs at 100 Hz, and a separate worker requests policy actions at up
to 30 Hz. Holding both VR triggers takes control; releasing them returns
control to the policy. The left controller's `Y` button marks the episode
successful and `X` marks it failed; the episode is recorded either way and the
loop resets for the next one.

Configure the policy WebSocket in `config/hitl/franka.yaml`, then run
`python vr_hitl_inference.py` inside the Humble container. For a simulated arm, use
`python vr_hitl_inference.py --dry-run`; this still requires a policy server.
For VR intervention, start `python -m teleop_xr.ros2 --mode teleop` in a
separate sourced Humble terminal. Both collection modes subscribe to its right
controller pose and left/right Joy topics. Hold both trigger buttons for the
configured `vr_long_press_s` to take control; stale pose or Joy input releases
the deadman.
`transition_server.enabled` optionally streams the original transition format
to a learning service. Local recording is independent of that service.

Each run writes to `data/hitl/<timestamp>_<id>/` by default:

- `events.jsonl` lists each episode start, VR fork, release, and
  episode result. A `branch_start` gives the exact `fork_step_index`,
  `branch_id`, `parent_branch_id`, control source, and EE pose.
- `episode_NNNNNN.msgpack` contains complete observations (including images),
  executed actions, next observations, and all fork/release events. Transitions
  include `step_start`, `step_end`, monotonic timestamps, `control_source`, and
  `branch_id`, EE orientation, and VR input sequence. Branch 0 is the policy
  path before the first takeover; after a
  takeover, the same branch ID covers the human action and the policy's
  continuation until the next fork.

The policy and camera rates are lower than the servo rate, so consecutive
servo commands are grouped into roughly one policy interval. The recorder
splits a group at every control source change, preserving each fork point.

## Manual control

The collection and inference scripts are not needed to move the arm by hand.
`control/franka_ros2_control.py` is a small CLI over
`control.robotic_arm_controller_ros.RoboticArmControlerRos`. Run it from the
repository root inside the Humble container:

```bash
python -m control.franka_ros2_control status
python -m control.franka_ros2_control hold
python -m control.franka_ros2_control gripper-open
python -m control.franka_ros2_control gripper-close
python -m control.franka_ros2_control move-start
```

Missing real-robot topics are errors unless `--use-fake-hardware` is passed, so
the CLI never silently commands a robot. The same module exposes an importable
facade: `from control.franka_ros2_control import FrankaROS2Control`.

## Tests

The tests replace the robot, cameras, and policy server with fakes, so they
never move hardware and never write a dataset. Run them in the Humble
container, which is the only environment where the ROS entry points run at all:
the host interpreter is Python 3.11 while the host ROS is Jazzy (Python 3.12),
so `import rclpy` fails outside the container.

```bash
bash ros2_ws/docker/franka_humble/scripts/run_tests.sh
bash ros2_ws/docker/franka_humble/scripts/run_tests.sh tests/test_inference.py
```

That script syncs `/opt/uv/venv` from
`ros2_ws/docker/franka_humble/python`, then runs pytest in the container.
`tests/conftest.py` skips `test_multirate.py` and `test_episode_edit.py` when
`lerobot` is missing, so the container run covers every test it can import
instead of failing collection. The entry points split the same way —
`vr_collect.py` reports an error when logging is enabled but lerobot is missing, while
`inference.py` and `vr_hitl_inference.py` need only the container venv
(`websockets`, `msgpack`).

The host `.venv` still runs the non-ROS tests without waiting for a container
sync — the `data_analysis/` tooling and the inference contract tests:

```bash
.venv/bin/python -m pytest -q tests/test_inference.py
```
