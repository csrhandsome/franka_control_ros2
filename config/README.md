# 配置文件说明

当前机器人是 Panda。工作流文件名 `franka.yaml` 是配置名称，实际型号由
继承后的 `robot.robot_type: panda` 决定；命令行 `--robot franka` 选择的是文件名。

| 文件 | 用途 / 加载入口 |
| --- | --- |
| `common/franka.yaml` | 三种工作流共享的机器人、起始姿态、相机、夹爪和正常工作流的关节轨迹约束 |
| `collect/franka.yaml` | 真实采集入口：数据集、VR、控制模式、相机和 `motion` 连续控制设置；`collect.sh` / `vr_collect.py` |
| `collect/franka_fake.yaml` | 继承真实采集参数，明确切换为假硬件、无相机、无夹爪、不保存数据 |
| `inference/franka.yaml` | 普通推理 / Force RLT：策略服务器、推理周期与执行选项；`inference.py` |
| `hitl/franka.yaml` | 人工接管推理：VR、策略服务器、分支记录、工作空间；`inference_hitl.py` |
| `planer/panda.yaml` | Panda 接口与 Python 运动默认值；`RoboticArmControlerRos` 加载后应用工作流的 `motion` 覆盖 |
| `../ros2_ws/src/data_collect_franka/config/panda_trajectory_controller.yaml` | ROS 原生关节轨迹控制器参数，供 controller_manager / spawner 加载 |


## 继承规则

工作流使用 `control.robot_config.load_mapping` 加载：

```yaml
extends: ../common/franka.yaml

robot:
  use_fake_hardware: true
camera:
  image_hw: 128
```

`extends` 路径相对于声明它的 YAML 文件，不受命令执行目录影响。也支持路径列表，
依次合并，后面的父文件覆盖前面的值，当前文件最后覆盖。字典递归合并；列表和
标量整体替换，不追加。循环继承、非法继承字段、非字典配置会报错。加载结果中
不保留 `extends` 元数据。现有没有 `extends` 的独立配置仍可使用。

`collect.sh` 的 bringup 参数解析、采集、普通推理、HITL 和采集
smoke test 使用同一继承加载器。不要用原始 `yaml.safe_load` 直接读取这些入口文件，
否则只会读到局部覆盖值。ROS 原生参数文件不使用 `extends`。

## 运动配置的生效顺序

1. `planer/panda.yaml` 提供关节名、坐标系、ROS 话题、控制器接口、超时和运动默认值。
2. `common/franka.yaml` 与入口 YAML 经 `extends` 合并；入口的 `motion` 覆盖运动默认值。
3. 工作流的 `robot.use_fake_hardware` 是硬件选择的最终值；`gripper_type: dh5 | none`
   自动关闭 Franka 夹爪客户端。

`workflow_motion_config()` 校验并返回完整运动配置，各入口通过构造参数
`motion_config` 将其交给 `RoboticArmControlerRos`。`motion` 保持嵌套，不能像旧的
采集参数一样展开，否则其中的 `robot` / `gripper` 会覆盖工作流字段。
未知运动字段、非法数值和机器人型号冲突会报错。

采集明确启用 EE、joint 连续接口和 Franka 夹爪；普通推理使用 30 Hz 运动设置，
采集和 HITL 使用 100 Hz。`force_rlt.execute_joint_targets` 仍是是否执行策略输出的
独立开关。若选择了某控制模式，却在 `motion` 禁用所需接口，启动检查直接报出
冲突；`collect.sh` 会在启动 bringup、VR、采集容器前完成此检查。

例如，只修改采集的连续目标保护：

```yaml
extends: franka.yaml
motion:
  ee:
    enabled: true
    streaming:
      limits:
        max_translation_step_m: 0.02
        max_translation_m: 0.9
```

## 连续控制与轨迹约束

| 设置 | 生效范围 |
| --- | --- |
| `control.max_ee_*`（collect）、`vr.vr_max_*`（HITL） | VR 映射器的每轴增量和每次按住扳机后的位移范围 |
| `inference.max_ee_translation_step_m` / `max_ee_rotation_step_rad` | 策略目标相对实测位姿的允许步长 |
| `motion.ee.streaming.limits` | EE 指令相对上一目标的整体平移/旋转步长，以及相对 stream 起点的整体范围 |
| `motion.joint.streaming.limits` | joint 指令的每关节步长和相对 stream 起点的关节位移范围 |
| `motion.limits` | 有限轨迹的位移、速度、加速度校验，包括回起始姿态 |

连续控制不再使用轨迹检查的速度/加速度与 2 cm 测试范围。连续目标保留步长和
范围保护，实际平滑与硬件运动约束由 ROS 控制器/驱动执行；这些目标保护不提供
碰撞检测。collect 的 mapper 使用每轴 8 mm / 0.008 rad，因此三轴合成步长最多约
14 mm / 0.014 rad，连续指令保护为 20 mm / 0.02 rad。每轴 0.5 m / 1.2 rad 的
VR 范围对应整体约 0.87 m / 2.08 rad，stream 保护为 0.9 m / 2.1 rad。

独立加载 `planer/panda.yaml` 的有限轨迹仍使用小范围验证参数。
正常工作流通过 common 的 `motion.limits` 将关节轨迹范围设为 3.2 rad、速度
0.5 rad/s、加速度 1.0 rad/s²。回起始姿态使用有限轨迹，不向连续接口发送一次
大幅跳变。入口文件不再声明未传给控制器的 `ee_filter_coeff`、
`ee_nullspace_stiffness`；ROS 原生控制器参数在 ROS 包中设置。

## 运行与验证

采集配置使用真实硬件、ROS 相机并启用数据保存；普通推理使用假硬件和 ROS 相机；
HITL 使用假硬件、关闭相机并启用分支记录。各工作流的开关保留在入口文件中，
图像尺寸、VR 增益、策略端口等差异保留。上述工作流均明确启用所需的在线接口。

配置允许使用接口，与 Docker 内控制器是否配置并可激活是两项独立检查。Panda
镜像使用固定提交的 LCAS 驱动和 libfranka 0.9.2；项目 bringup 将 joint/EE 目标
控制器及关节轨迹控制器预配置为 inactive，由 Python 类按需激活。EE 控制器使用
该驱动的 `ee_cartesian_position/00` 至 `/15` 硬件接口，未链接新版 FR3 的语义接口。
实际激活仍检查新鲜测量反馈，配置不会通过全局禁用开关隐藏驱动问题。

配置与控制相关测试在仓库根目录通过 Humble 容器运行：

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble \
  bash ros2_ws/docker/franka_humble/scripts/pytest_in_container.sh \
  tests/test_robot_config.py tests/test_planer_control.py tests/test_inference.py
```

集成验证完成后，真实采集入口为 `./collect.sh config/collect/franka.yaml`，
假硬件配置为 `./collect.sh config/collect/franka_fake.yaml`。分开调试 ROS bringup 时使用
`compose_devices.sh`，采集/推理客户端使用 `compose_safe.sh`。具体命令见根目录 README。
