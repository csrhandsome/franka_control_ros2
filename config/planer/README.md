# 机械臂轨迹与在线控制配置

`panda.yaml` 是此目录唯一的 YAML，包含当前 Panda 的完整 Python 运动参数：
关节名、状态话题、控制器接口、超时、完成条件、限速和功能开关。
不继承其他运动参数文件；`control/motion_config.py` 从这份文件读取默认值并校验覆盖值。
各工作流通过自己的 `motion` 节覆盖这些默认值，并将解析结果传给控制类。

ROS 原生关节轨迹控制器参数放在
`ros2_ws/src/data_collect_franka/config/panda_trajectory_controller.yaml`，供 ROS spawner
读取。它的格式与 Python 运动配置不同，不能合并后直接传给 spawner。

EE/joint streaming 和夹爪接口默认启用。连续目标的步长/范围保护放在
`ee.streaming.limits` / `joint.streaming.limits`；顶层 `limits` 只校验有限轨迹。
独立轨迹检查保留小范围参数，工作流的关节轨迹约束在 common 中覆盖。

所有机械臂控制在 Humble 容器中经 `RoboticArmControlerRos` 执行，
不直接调用 libfranka 运动。配置不包含自动运动，也不包含相机或数据保存设置；
这些属于 `config/common/franka.yaml` 和各工作流配置。
采集会按工作流中继承的 `robot_type: panda` 加载 Panda profile，再应用自己的
`motion`。配置支持连续控制；当前 Panda Docker 驱动与 collection overlay 的
兼容性仍需集成验证，详见 [配置说明](../README.md)。

```python
arm = RoboticArmControlerRos(config_path="config/planer/panda.yaml")
try:
    arm.connect()
    state = arm.wait_ready(required=("joints",))
    result = arm.execute_joint_trajectory(points)
    # result: {"status": "succeeded", "space": "joint", "duration_s": ...}
finally:
    arm.close()
```

这是同步接口，不创建后台 MotionHandle。执行失败或超时会抛异常。
`on_sample(state)` 可以在等待执行期间记录状态，不必另建运动任务框架。
`cancel_motion(timeout_s=...)` 从另一个线程发起取消，并等待执行线程确认结果。
超时抛异常，取消请求仍保留。`on_sample` 在执行线程中调用，因此不要在该回调内
调用阻塞的取消接口。VR 和策略在线控制使用 `start_stream("joint" | "ee")` 和
`send_joint_target` / `send_ee_target`；返回值只确认本地发布，不表示到达目标。
在线模式与规划运动互斥。切换前调用 `stop_stream()`，该方法发布实测保持目标、
等待新鲜速度反馈确认静止，然后停用控制器；失败会抛异常。

关节轨迹点包含 `time_s`、7D `positions`、`velocities`、`accelerations`，
从当前状态和时间 0 开始，以零速度、零加速度结束；ROS 按五次多项式插值。
EE 点包含 `time_s`、3D `position`、4D `quaternion_xyzw`；Python 用平滑时间曲线
插值位置并对四元数执行 SLERP，逐帧发送给 EE 目标控制器。
这不提供障碍物规划或碰撞检测，也不替代驱动的运动保护。

`get_state(required=...)` 拒绝缺失或过期的实测数据；`required=()` 可读取诊断快照，
未接收到的字段为 `None`。快照包含 `valid`、`age_s`、`stamp_s` 和基坐标系，
不同状态源不承诺时间同步。夹爪总开口宽度字段为 `gripper_width_m`。
`get_status()` 返回连接与控制状态；`get_capabilities()` 返回各能力的启用状态、
可用状态和不可用原因，不激活运动。

主类只定义公开接口，私有函数位于 `control/_ros/`。调试命令运行
`python -m control.robotic_arm_ros_cli`，构造后需显式 `connect()`，也可使用上下文管理器。
工作流复用 `control/util/robot.py` 完成初始位运动，夹爪保留
`gripper_open`、`gripper_close`、`wait_gripper`、`stop_gripper`，不提供操作句柄。

在仓库根目录进行无真机运动的验证：

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble \
  bash ros2_ws/docker/franka_humble/scripts/pytest_in_container.sh \
  tests/test_planer_control.py tests/test_inference.py

bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps \
  -e ROS_DOMAIN_ID=145 franka_humble python tests/docker_planer_smoke.py
```

启动真实 Panda 后加载轨迹控制器，保持未激活：

```bash
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble \
  ros2 run controller_manager spawner joint_trajectory_controller --inactive \
  --controller-type joint_trajectory_controller/JointTrajectoryController \
  --param-file /workspace/data_collect/ros2_ws/src/data_collect_franka/config/panda_trajectory_controller.yaml
```

完整真实轨迹记录与 replay 可视化流程见 `replay/README.md`。
