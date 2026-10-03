# Franka ROS 2 collection and Human in the Loop control

The normal collection command is `./collect.sh`. It manages the ROS bringup,
VR publisher, and `vr_collect.py` containers together on the host. Inside the
Humble Docker container, the same `bash collect.sh` command starts those three
components as local processes and stops them together on Ctrl+C. Start the
container with device access when using ROS cameras:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm franka_humble bash collect.sh
```

Use `compose_dh5.sh` instead when the collection configuration uses the DH5
gripper. The image, ROS overlay, and `.env` must be prepared before either mode.

| Entry point                      | Purpose                                                   | Configuration                  |
| -------------------------------- | --------------------------------------------------------- | ------------------------------ |
| `./collect.sh` / `vr_collect.py` | VR teleoperation and LeRobot recording                    | `config/collect/franka.yaml`   |
| `inference.py`                   | 30 Hz EE policy inference or Force RLT, without recording | `config/inference/franka.yaml` |
| `inference_hitl.py`              | Policy rollouts with VR takeover and branch recording     | `config/hitl/franka.yaml`      |

Robot control, collection, camera access, and ROS inference run in the
`franka_humble` Docker image, with Python 3.10 at `/opt/uv/venv`. The repository
is mounted at `/workspace/data_collect`; the host `.venv` is a separate
environment for offline tools. Model training and the policy WebSocket server
run separately in the `openpi-force` GPU environment.

Run the Docker commands below from the repository root. `compose_safe.sh` is
the wrapper for clients that only need ROS topics and the project mount;
`compose_devices.sh` adds USB/video access for camera bringup. Both load the
project's Compose overlays and `ros2_ws/docker/franka_humble/.env`.

The device wrapper discovers and maps the host's current `/dev/video*` nodes
and binds `/dev/bus/usb`. Connect both cameras before starting it. After a
camera is unplugged/reconnected, recreate the camera container through the
wrapper to refresh video mappings. These runtime settings need no image rebuild.
Check camera visibility from the repository root with:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm --no-deps -T franka_humble rs-enumerate-devices -s
```

## Environments and responsibilities

This repository keeps two separate Python environments, split by function so
they do not install each other's dependencies:

- **Collection runs in Docker.** The `franka_humble` image owns robot control,
  cameras, VR teleoperation, LeRobot recording, and ROS inference. It carries its
  own runtime at `/opt/uv/venv` (Python 3.10) plus the collection-only
  dependencies — ROS, `teleop-xr`, `sounddevice`, `silero-vad`, `qwen-asr`, and
  camera drivers. Collection commands never run on the host.
- **Viewing and analysis run on the host.** The root Python 3.11 `.venv`
  (created with `uv sync`) serves two offline jobs: the `replay/` workspace
  (browse and inspect recorded episodes over HTTP) and `scripts/data_analysis/`
  (quality checks, episode editing/merging, and audio tools). It installs
  `lerobot` (read and validate LeRobot datasets) and `opencv-python` (decode and
  view images), the replay stack (FastAPI, `pyarrow`, Pillow, `imageio-ffmpeg`),
  and the analysis stack (NumPy, pandas, matplotlib) — but none of the ROS or
  collection-only packages above.

The Docker image does not install the replay/analysis web stack, and the host
`.venv` does not install the ROS/collection stack. Model training and the policy
WebSocket server remain in the separate `openpi-force` GPU environment.

**Current startup blockers:** container checks found Python 3.11-only imports
in the Python 3.10 Humble runtime. `vr_collect.py --help` fails on
`typing.Self` in `control/soft_gripper_control_ros.py`; `inference_hitl.py --help`
fails on `datetime.UTC` in `control/hitl/recording.py`. Plain inference also
imports the HITL package during startup and reaches the same `datetime.UTC`
error, although `inference.py --help` succeeds. Resolve these imports before
running collection or inference; the commands below describe the intended
workflow. Python 3.10-compatible alternatives are `typing_extensions.Self`
and `datetime.timezone.utc`.

## 在 Docker 中采集 LeRobot 数据

采集入口统一为 `vr_collect.py`。通过 `dataset.lerobot_format: v2 | v3`
选择数据格式，也可用 `--lerobot-format` 临时覆盖：v2 对应 Humble 镜像中的
固定 LeRobot 0.1 版本，v3 对应独立 Jazzy 镜像中的 LeRobot 0.6.1。
`./collect.sh` 只支持 Humble / v2；Jazzy / v3 目前用于合成验证，尚不能直接
启动真机采集。两种格式共用采集接口和核心字段，切换格式时需使用新的
`dataset.repo_id` 或 `dataset.date`。

采集容器使用 ROS 2 Humble、固定版本的 LeRobot 和 CPU 版 PyTorch，直接写入
LeRobot 数据集；神经网络和训练在独立 GPU 环境运行，ROS 推理客户端仍在
Humble 容器中运行。以下命令均在仓库根目录执行。
首次使用时，从 `ros2_ws/docker/franka_humble/.env.example` 复制 `.env`，并填写
实际的机器人 IP、网卡及用户 UID/GID。

```bash
test -f ros2_ws/docker/franka_humble/.env || cp ros2_ws/docker/franka_humble/.env.example ros2_ws/docker/franka_humble/.env
```

核对 `.env` 中的 `FRANKA_ROBOT_IP`、`FRANKA_INTERFACE`、`USER_UID`、
`USER_GID`、`PROJECT_ROOT` 和 `ROS_DOMAIN_ID`。如果尚无基础镜像
`data-collect/franka-ros2-humble:2.5.1`，先构建它：

```bash
bash ros2_ws/docker/franka_humble/scripts/compose.sh build franka_humble
```

再构建 uv 运行镜像和 ROS overlay，并检查环境：

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh build franka_humble
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble bash ros2_ws/docker/franka_humble/scripts/build_overlay.sh
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble uv-env-smoke-test
```

合成采集检查命令如下；应在修复上述 Python 3.10 兼容性问题后执行：

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_dataset_adapter_smoke.py --format v2
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_recording_smoke.py
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm franka_humble python tests/docker_collect_pipeline_smoke.py
```

采集检查验证统一数据集接口、LeRobot episode 的写入/回读，以及
`vr_collect.py` 的完整录制流程；它们使用合成输入和临时目录。接入真实设备前，修改
`config/collect/franka.yaml`：`dataset.enable_logging: true`、
`camera.camera_backend: ros`、`robot.use_fake_hardware: false`，将
`dataset.date` 设为本次采集标识，并核对两路相机序列号和话题。配置完成且
镜像与 ROS overlay 已构建后，在仓库根目录运行以下命令，一次启动
控制器、VR 话题和录制进程；按 Ctrl+C 会停止对应容器：

```bash
./collect.sh
# 使用仓库内的其他配置时，将路径替换为实际文件。
./collect.sh config/collect/franka.yaml
```

脚本读取 `config/collect/franka.yaml`，也可传入仓库内的其他采集配置路径。当前 YAML
启用真实硬件、相机和数据保存，运行前必须逐项核对。`dataset.action_space: ee`
保存 7D 的末端位姿加夹爪动作；改为 `joint` 则保存 8D 的关节角加夹爪动作。
切换动作空间后请更换 `dataset.date`。
`control.control_mode` 选择控制模式，与记录动作的 `dataset.action_space`
是两个独立配置。只调试假硬件时，应在采集配置中明确设置
`robot.use_fake_hardware: true`、`camera.camera_backend: none` 和
`dataset.enable_logging: false`。
`./collect.sh` 目前只启动 Humble 运行环境，因此要求 `dataset.lerobot_format: v2`；
Jazzy 镜像当前只完成合成记录验证，尚无机器人 bringup。

长按双手扳机约 0.5 秒启用运动，右手位姿控制末端；松开并再次长按会重新
锚定 VR 零位。右手 A/B 分别关闭/打开夹爪；检测到运动或夹爪动作后自动
开始录制。采集进程启动时会打开夹爪并移动到配置的起始关节位置。

使用 DH5 时，设置 `gripper.gripper_type: dh5`，核对
`gripper.soft_gripper_port`、夹爪相机话题及 `.env` 中的
`DH5_SERIAL_DEVICE`。`collect.sh` 会自动使用 `compose_dh5.sh` 启动带
串口和 USB/video 映射的 bringup；右手摇杆 Y 轴也可逐步调整 DH5 开合。

当前机器人为 **Panda**，公共配置见 `config/common/franka.yaml`，继承与文件用途见
[配置说明](config/README.md)。采集的 `motion` 明确启用 EE、joint 连续控制和夹爪，
连续目标使用独立步长/范围保护；小范围轨迹测试参数不再限制采集。Panda
Docker 使用 LCAS 驱动与 libfranka 0.9.2，collection overlay 的 joint/EE 目标控制器
通过该驱动的 ROS 硬件接口执行。请使用下面的项目 bringup，而非上游默认 launch；
上游默认 launch 不加载项目目标控制器。实机验证命令见文末测试说明。

需要分别调试三个进程时，也可以在三个终端运行下面的命令：

```bash
# 终端 1：机器人控制器与两路 RealSense；此容器需要 USB 设备。
bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm franka_humble \
  bash -c 'ros2 launch data_collect_franka franka_data_collect.launch.py \
    robot_type:=panda robot_ip:="$FRANKA_ROBOT_IP" use_fake_hardware:=false \
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

## Jazzy / LeRobot v3 合成验证

`ros2_ws/docker/franka_jazzy/` 使用 ROS 2 Jazzy、Python 3.12 和
LeRobot 0.6.1。它复用 Humble 的 `.env` 主机设置，但有独立镜像、Compose
项目和 Python 环境。当前项目挂载可写，合成检查使用临时数据目录；该容器
没有 USB 映射，也没有 Franka ROS 包、机器人 overlay、VR 发布器或相机
bringup。它不替换 Humble 部署环境，也不改变训练仓库的 LeRobot 版本。

```bash
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh build
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python ros2_ws/docker/franka_jazzy/scripts/smoke_test.py
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python tests/docker_dataset_adapter_smoke.py --format v3
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python tests/docker_recording_smoke.py --backend jazzy
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python tests/docker_collect_pipeline_smoke.py --backend jazzy
```

这些检查覆盖 ROS Python / `cv_bridge` 导入、v3 创建和回读、续写与丢弃、
DH5 字段、生产写入器和合成采集流程。`cv_bridge` 4.1.0 在镜像中针对
NumPy 2 重建。详细说明见 [Jazzy 容器说明](ros2_ws/docker/franka_jazzy/README.md)。

## Multi-rate collection

`vr_collect.py` samples the latest VR pose and sends a
robot target at the configured `control.control_frequency` (100 Hz). The ROS
controller still runs at 1 kHz. A 100 Hz loop does not make the headset produce
100 new poses per second; repeated `vr_pose_seq` values identify reused input.

The collection loop samples the latest robot state; its 100 Hz rate does not
guarantee 100 fresh ROS state messages per second. The current bringup defaults
to `joint_state_rate:=30`, independently of the controller's 1 kHz update rate.

The two RealSense nodes run at 640×480×30. The camera adapter crops and resizes
frames in a background thread. Only a pair in which **both** cameras have a new
frame is inserted into LeRobot. That dataset has `fps=30`; every frame records
both a 6D `ee_pose` and 7D `joint_position`. `dataset.action_space` selects
whether its `actions` feature contains 6D EE pose plus gripper (7D) or seven
joint angles plus gripper (8D). A frame action is measured at the next fresh
camera pair. Image insertion runs in a separate thread. `exterior_image_2_left`
is a blank placeholder for schema compatibility. DH5 recordings additionally
store both gripper images and front/wrist timestamp and frame-age fields.

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

`inference.py` selects its protocol from the policy server's metadata. The
checked-in config connects to the merged Force RLT server on **port 8002**;
the EE workflow below must use **`--port 8001`** (or set that port in the YAML).
Both modes run at 30 Hz and do not save observations or episodes. The current
Python 3.10 import blockers listed above must be fixed before either mode runs.

For the EE protocol, `inference.py` runs the `pi0_base_ee_ros_30hz` checkpoint trained in
`openpi-force`. It sends 224×224 front/wrist images, measured 6D EE pose, the
logical 0/1 gripper state, and the task prompt to the policy server. The server
returns 16 future absolute EE pose targets and logical gripper commands at 30 Hz.
The inference loop samples state and publishes a target at 30 Hz, requesting a
new chunk after eight model steps. This entry point does not save observations
or episodes. Before creating its camera and arm adapters, it sends five blank
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
Humble containers from this repository root. Workflow `motion` settings enable EE/joint
streaming and gripper support with separate online target limits. The Panda runtime uses the project joint/EE controllers through the LCAS driver
(see [configuration notes](config/README.md)); build the matching overlay first. The policy server runs in
the `openpi-force` GPU environment. For the current Panda, set
`robot.use_fake_hardware: false` in the inference YAML and use the matching
bringup argument:

```bash
# Terminal 1: robot, gripper, and both cameras (USB devices required here).
bash ros2_ws/docker/franka_humble/scripts/compose_devices.sh run --rm franka_humble \
  bash -c 'exec ros2 launch data_collect_franka franka_data_collect.launch.py \
    robot_type:=panda robot_ip:="$FRANKA_ROBOT_IP" use_fake_hardware:=false \
    load_gripper:=true start_cameras:=true joint_state_rate:=30 \
    external_serial:=825412070292 wrist_serial:=825412070487'

# Terminal 2: inference client (shares host networking and the ROS domain).
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python inference.py --port 8001 --prompt "Place the object into the basket"
```

The checked-in inference config uses fake hardware and requires live ROS
cameras. The `policy_server.host` value must point to the GPU server; because
the containers use host networking, `127.0.0.1` works when that server runs on
the same machine. The ROS overlay must be built before running these commands.
`--move-to-start` moves to the configured
start joints first; otherwise inference starts from the current pose. The
default run lasts 600 control ticks (20 seconds); `--max-steps 0` runs until
Ctrl+C. The policy server must expose the training config's 30 Hz metadata.

### Force RLT

Start the merged VLA/readout/actor server in the separate GPU environment, then
use the same ROS bringup and the default client port:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python inference.py --port 8002 --prompt "Place the object into the basket"
```

The client sends both 224×224 RGB images, seven measured joints, logical
gripper state, the measured `O_T_TCP` transform, a monotonic observation time,
and the prompt. It validates the server's Force RLT protocols, 30 Hz timing,
16-step prediction horizon, actor hash, and observation timestamp. The actor
proposes seven physical joint targets per step; this protocol does not use the
EE mode's 7D pose-and-gripper action layout.

`force_rlt.execute_joint_targets: false` is the default: the loop requests and
checks actor proposals while publishing measured joints to hold the arm.
`--move-to-start` still requests a move to the configured starting pose.
Joint-target execution is supported only with fake hardware and explicit
seven-value arrays for `force_rlt.joint_lower`, `joint_upper`,
`max_joint_step_rad`, and `reference_radius_rad`. Real-hardware execution is
rejected until a calibrated TCP/IK safety projector is available. Group B is
supported; group C is rejected because the client has no synchronized finger
camera history pipeline. The default reference age limit is 0.2 seconds.

## Human in the Loop inference and recording

`inference_hitl.py` runs policy inference with optional VR takeover and records the rollout. The
servo loop runs at 100 Hz, and a separate worker requests policy actions at up
to 30 Hz. Holding both VR triggers takes control; releasing them returns
control to the policy. The left controller's `Y` button marks the episode
successful and `X` marks it failed; the episode is recorded either way and the
loop resets for the next one.

Configure the policy WebSocket in `config/hitl/franka.yaml`. The checked-in
HITL config uses fake hardware, disables ROS cameras, and enables recording.
For a live run, set `robot.use_fake_hardware: false` and
`camera.camera_backend: ros`, then start the matching ROS bringup with
`compose_devices.sh` as shown above. Once the Python 3.10 import blockers are
resolved, run the client and VR publisher in separate Humble containers:

```bash
# Terminal 2: VR topics for intervention.
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python -m teleop_xr.ros2 --mode teleop

# Terminal 3: policy rollouts and local branch recording.
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python inference_hitl.py --config config/hitl/franka.yaml
```

For dummy arm/cameras, the following does not need ROS bringup or VR topics,
but still requires a compatible policy server and writes local recordings:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python inference_hitl.py --dry-run --max-episodes 1
```

The HITL client expects an XYZ displacement and a gripper-close flag in the
policy's `actions` response. Its observation layout and action interpretation
differ from both plain inference protocols; use a server compatible with this
HITL contract. Both VR workflows subscribe to the publisher's right
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
`control/robotic_arm_controller_ros.py` exposes only the public
`RoboticArmControlerRos` API. Private ROS resources, callbacks, motion execution,
and gripper workers live in `control/_ros/`. State, status, capability and result
contracts live in `control/robotic_arm_ros_types.py`; pose utilities are reused
from `control/util/pose.py`. The separate CLI is `control/robotic_arm_ros_cli.py`.
Run it from the
repository root inside the Humble container:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python -m control.robotic_arm_ros_cli status
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python -m control.robotic_arm_ros_cli gripper-open
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python -m control.robotic_arm_ros_cli gripper-close
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python -m control.robotic_arm_ros_cli move-start
```

Start the matching ROS bringup first. Missing measured joints are errors on
both real and fake hardware. `--use-fake-hardware` identifies the fake runtime;
it does not bypass readiness or activate a motion controller. Import the class directly: `from control.robotic_arm_controller_ros import RoboticArmControlerRos`.

## Tests

The unit and synthetic tests use fake robot/camera inputs and fake or local test policy servers.
Dataset tests write temporary synthetic datasets; they do not move real
hardware. Run ROS runtime checks in the Humble container after resolving the
startup blockers above:

```bash
bash ros2_ws/docker/franka_humble/scripts/run_tests.sh
bash ros2_ws/docker/franka_humble/scripts/run_tests.sh tests/test_inference.py
```

For **live tests of `RoboticArmControlerRos`**, start the existing Panda ROS
bringup in one terminal (skip this if it is already running):

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble bash -c 'exec ros2 launch data_collect_franka franka_data_collect.launch.py robot_type:=panda robot_ip:="$FRANKA_ROBOT_IP" use_fake_hardware:=false load_gripper:=true start_cameras:=false'
```

Then run the test in another terminal from the repository root:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python tests/docker_arm_connection_test.py
```

This is an ordinary `unittest` suite running in one process. It tests joint/EE
streaming, direct moves, trajectories and cancellation through the class's
public interfaces. Joint 1 moves by 0.006 rad and EE moves by 0.004 m along
base-frame +Z; successful cases check measured motion and return to the start.
The extended EE streaming case keeps the measured starting orientation and
follows two continuous ellipses in the base-frame X/Z plane over 40 seconds:
X ranges from -0.02 to +0.02 m relative to the start, and Z from 0 to +0.05 m.
It sends targets at 100 Hz, uses quintic timing for smooth starts and stops,
then holds the starting pose for two seconds before stopping the stream.
It checks both loops' measured range, tracking error, orientation and return
within 1 mm. These tests request 0.8 mm EE completion accuracy while retaining
the 1 mm arrival assertions.

Run only the EE cases (including the extended stream), stopping on the first
failure:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python tests/docker_arm_connection_test.py -k ee -f
```

For only the larger continuous motion:

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  franka_humble python tests/docker_arm_connection_test.py \
  RoboticArmControlerRosTest.test_09_ee_continuous_motion
```

The test container must use the same `ROS_DOMAIN_ID` as the bringup; add
`-e ROS_DOMAIN_ID=<bringup-domain>` before `franka_humble` if the bringup
overrides the `.env` value. `FRANKA_TEST_EE_TRACE_PATH` optionally records the
extended stream's target poses and measured feedback as CSV; use a mounted
output path to keep it after the test container exits.

Each test closes its client and checks ROS entities, owned context and threads.
Movement and cleanup errors appear as ordinary test failures with their
original tracebacks. Stop the temporary bringup with Ctrl-C after testing.

`run_tests.sh` syncs `/opt/uv/venv` from
`ros2_ws/docker/franka_humble/python`, then runs pytest in the container.
`tests/conftest.py` skips `test_multirate.py` and `test_episode_edit.py` when
`lerobot` is missing, so the container run covers every test it can import
instead of failing collection. The entry points split the same way —
`vr_collect.py` reports an error when logging is enabled but lerobot is missing, while
`inference.py` and `inference_hitl.py` use the policy client dependencies
(`websockets`, `msgpack`) without loading a model locally. The dataset smoke
commands above cover the v2 and v3 recording adapters separately. Synthetic
checks do not verify FCI, live camera/VR input, or real-arm execution.

## Repository layout

| Directory                          | Contents                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| `control/`                         | ROS arm, gripper, camera and VR adapters; collection recorder; HITL loop      |
| `config/{collect,inference,hitl}/` | Workflow overrides inheriting shared Panda configuration                      |
| `config/common/`                   | Shared robot/camera workflow values                                           |
| `config/planer/panda.yaml`         | Panda interfaces and motion defaults; workflows apply `motion` overrides      |
| `scripts/data_analysis/`           | Offline dataset quality checks, episode editing/merging, and audio tools      |
| `scripts/modelscope/`              | ModelScope dataset/model transfer scripts (ignored; contains API credentials) |
| `ros2_ws/src/data_collect_franka/` | ROS bringup, controllers, DH5 and camera nodes                                |
| `ros2_ws/docker/`                  | Humble deployment runtime and isolated Jazzy experiment                       |
| `tests/`                           | Unit/contract tests and synthetic Docker recording checks                     |
