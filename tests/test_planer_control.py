"""ROS control contract tests. Run in Humble; never connect to real hardware."""

import threading
import time
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import JointState

from control.motion_config import load_motion_config, workflow_motion_config
from control.robot_config import load_mapping
from pathlib import Path
from control.robotic_arm_controller_ros import (
    RoboticArmControlerRos,
)
from control._ros import lifecycle, motion, state
from control._ros.motion import _joint_points
from control.util.pose import normalize_quat_xyzw as _quaternion, slerp_quat_xyzw as _slerp
from control.util.robot import move_robot_to_start_pose


def points(duration=4.0, distance=0.02):
    return [
        {
            "time_s": t,
            "positions": [q] + [0.0] * 6,
            "velocities": np.zeros(7),
            "accelerations": np.zeros(7),
        }
        for t, q in ((0.0, 0.0), (duration, distance))
    ]


def test_profiles_and_unknown_keys(tmp_path):
    panda = load_motion_config("config/planer/panda.yaml")
    assert panda["robot"]["joint_names"][0] == "panda_joint1"
    assert panda["ee"]["enabled"]
    assert panda["joint"]["streaming"]["enabled"]
    assert load_motion_config()["robot"]["robot_type"] == "panda"
    with pytest.raises(ValueError, match="Only Panda"):
        load_motion_config(robot_type="fr3")
    config = tmp_path / "bad.yaml"
    config.write_text("execution:\n  state_timout_s: 0.5\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_motion_config(config)
    config.write_text("limits:\n  max_joint_velocity_rad_s: .nan\n")
    with pytest.raises(ValueError, match="finite"):
        load_motion_config(config)


def test_joint_limits_include_between_waypoints():
    cfg = load_motion_config("config/planer/panda.yaml")
    assert len(_joint_points(points(), np.zeros(7), cfg)) == 2
    # Endpoints are valid; the short segment violates velocity/acceleration.
    with pytest.raises(ValueError, match="exceeds"):
        _joint_points(points(duration=0.01), np.zeros(7), cfg)
    with pytest.raises(ValueError, match="finite"):
        _joint_points(points(distance=float("nan")), np.zeros(7), cfg)
    with pytest.raises(ValueError, match="measured"):
        _joint_points(points(), np.ones(7), cfg)


def test_quaternion_shortest_path():
    q = _quaternion([0, 0, 0, 2])
    np.testing.assert_allclose(_slerp(q, -q, 0.5), q)
    with pytest.raises(ValueError, match="nonzero"):
        _quaternion([0, 0, 0, 0])


def bare_arm():
    arm = RoboticArmControlerRos(config_path="config/planer/panda.yaml")
    arm._connected = True
    arm._joint_positions = np.zeros(7)
    arm._joint_velocities = np.zeros(7)
    arm._have_joint_velocity = True
    arm._last_ee_pos = np.zeros(3)
    arm._last_ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
    arm._node = None
    arm._gripper_lock = threading.Lock()
    arm._gripper_command = arm._gripper_active_command = None
    return arm


def test_wrong_joint_names_do_not_mark_state_ready():
    arm = bare_arm()
    msg = JointState()
    msg.name = ["fr3_joint1"]
    msg.position = [1.0]
    state._on_joint_states(arm, msg)
    assert not arm.get_state(required=())["valid"]["joints"]
    msg.name = arm._config["robot"]["joint_names"]
    msg.position = [0.0] * 7
    msg.velocity = [0.0] * 7
    state._on_joint_states(arm, msg)
    assert arm.get_state(required=("joints",))["valid"]["joints"]
    assert not arm.get_state(required=())["valid"]["gripper"]
    arm._state_received["joints"] = time.monotonic() - 1
    with pytest.raises(RuntimeError, match="stale"):
        arm.get_state(required=("joints",))


def test_disabled_ee_never_activates():
    arm = bare_arm()
    arm._config["ee"]["enabled"] = False
    with pytest.raises(RuntimeError, match="disabled"):
        arm.execute_ee_trajectory([])
    assert arm._active_arm_controller is None
    assert not arm._trajectory_running


@pytest.mark.parametrize("space", ["ee", "joint"])
def test_collection_stream_runs_beyond_small_trajectory_limits(monkeypatch, space):
    arm = bare_arm()
    arm._config = workflow_motion_config(
        load_mapping(Path("config/collect/franka.yaml")), control_mode=space
    )
    arm._state_received["joints"] = arm._state_received["ee"] = time.monotonic()
    published = []
    monkeypatch.setattr(lifecycle, "_activate_arm_controller", lambda arm, controller: True)
    monkeypatch.setattr(
        motion, "_publish_ee_target", lambda arm, pos, quat: published.append(pos.copy())
    )
    monkeypatch.setattr(
        motion, "_publish_joint_target", lambda arm, joints: published.append(joints.copy())
    )
    arm.start_stream(space)
    # No timing-dependent sleeps: these are discrete target bounds, not
    # timed trajectory derivative limits. Targets stay inside the workspace.
    for tick in range(1, 101):
        if space == "ee":
            arm.send_ee_target([0.008 * tick, 0, 0], [0, 0, 0, 1])
        else:
            arm.send_joint_target([0.008 * tick] + [0.0] * 6)
    assert published[-1][0] == pytest.approx(0.8)
    count = len(published)
    with pytest.raises(ValueError, match="step"):
        if space == "ee":
            arm.send_ee_target([0.84, 0, 0], [0, 0, 0, 1])
        else:
            arm.send_joint_target([0.9] + [0.0] * 6)
    with pytest.raises(ValueError, match="translation|excursion"):
        if space == "ee":
            arm.send_ee_target([1.0, 0, 0], [0, 0, 0, 1])
        else:
            arm.send_joint_target([3.0] + [0.0] * 6)
    assert len(published) == count


def test_reset_uses_timed_trajectory_with_streaming_enabled(monkeypatch):
    arm = bare_arm()
    arm._config = workflow_motion_config(
        load_mapping(Path("config/collect/franka.yaml")), control_mode="ee"
    )
    arm._state_received["joints"] = time.monotonic()
    targets = []
    monkeypatch.setattr(
        arm, "move_joints", lambda target, duration_s: targets.append((target, duration_s))
    )
    monkeypatch.setattr(arm, "wait_until_stopped", lambda: None)
    move_robot_to_start_pose(arm, [2.0] + [0.0] * 6, motion_config=arm._config)
    target, duration = targets[0]
    assert duration > 4.0
    assert (
        len(_joint_points(points(duration=duration, distance=target[0]), np.zeros(7), arm._config))
        == 2
    )
    assert arm._stream_space is None


def test_synchronous_dispatch_and_callback(monkeypatch):
    arm = bare_arm()
    samples = []

    def execute(arm, points, callback):
        callback({"measured": True})

    monkeypatch.setattr(motion, "_execute_joint_trajectory", execute)
    result = arm.execute_joint_trajectory([], on_sample=samples.append)
    assert result["status"] == "succeeded"
    assert samples == [{"measured": True}]
    assert not arm._trajectory_running


def test_action_failure_is_not_success(monkeypatch):
    arm = bare_arm()
    arm._state_received["joints"] = time.monotonic()
    result = SimpleNamespace(
        status=6, result=SimpleNamespace(error_code=-4, error_string="tracking failed")
    )
    future = SimpleNamespace(done=lambda: True, result=lambda: result)
    goal = SimpleNamespace(accepted=True, get_result_async=lambda: future)
    client = SimpleNamespace(
        wait_for_server=lambda **kwargs: True,
        send_goal_async=lambda message: goal,
        destroy=lambda: None,
    )
    from control._ros import motion as module

    monkeypatch.setattr(module, "ActionClient", lambda *args: client)
    monkeypatch.setattr(lifecycle, "_ensure_loaded", lambda arm, controller: None)
    monkeypatch.setattr(motion, "_wait_future", lambda arm, value, timeout: value)
    monkeypatch.setattr(lifecycle, "_activate_arm_controller", lambda arm, controller: True)
    monkeypatch.setattr(lifecycle, "_deactivate_controller", lambda arm: None)
    with pytest.raises(RuntimeError, match="tracking failed"):
        arm.execute_joint_trajectory(points())


def test_ee_interpolation_uses_measured_state(monkeypatch):
    arm = bare_arm()
    arm._config = deepcopy(arm._config)
    arm._config["ee"]["enabled"] = True
    arm._config["ee"]["completion"]["settle_time_s"] = 0.01
    arm._state_received["joints"] = arm._state_received["ee"] = time.monotonic()
    arm._have_ee_pose = True
    published = []

    def publish(arm, pos, quat):
        published.append(pos.copy())
        arm._last_ee_pos, arm._last_ee_quat = pos, quat
        arm._state_received["joints"] = arm._state_received["ee"] = time.monotonic()

    monkeypatch.setattr(motion, "_publish_ee_target", publish)
    monkeypatch.setattr(lifecycle, "_activate_arm_controller", lambda arm, controller: True)
    result = arm.execute_ee_trajectory(
        [
            {"time_s": 0.0, "position": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
            {"time_s": 0.1, "position": [0.00001, 0, 0], "quaternion_xyzw": [0, 0, 0, -1]},
        ]
    )
    assert result["status"] == "succeeded"
    assert len(published) > 2
    np.testing.assert_allclose(published[-1], [0.00001, 0, 0])


def test_missing_state_is_none_and_snapshots_are_independent():
    arm = bare_arm()
    missing = arm.get_state(required=())
    assert missing["joint_positions"] is None
    assert missing["ee_position"] is None
    assert missing["gripper_width_m"] is None
    arm._state_received["joints"] = time.monotonic()
    snapshot = arm.get_state(required=("joints",))
    snapshot["joint_positions"][0] = 9.0
    assert arm.get_state(required=("joints",))["joint_positions"][0] == 0.0
    with pytest.raises(RuntimeError, match="ee.*missing"):
        arm.get_state()


def test_online_commands_reject_wrong_mode_and_report_publication(monkeypatch):
    arm = bare_arm()
    arm._state_received["joints"] = time.monotonic()
    arm._stream_space = "joint"
    arm._stream_origin = np.zeros(7)
    published = []
    monkeypatch.setattr(
        motion, "_publish_joint_target", lambda arm, target: published.append(target)
    )
    receipt = arm.send_joint_target(np.zeros(7))
    assert receipt["space"] == "joint"
    assert receipt["published_monotonic_s"] <= time.monotonic()
    assert len(published) == 1
    with pytest.raises(RuntimeError, match="EE stream is not active"):
        arm.send_ee_target([0, 0, 0], [0, 0, 0, 1])
    with pytest.raises(RuntimeError, match="stream is active"):
        arm.execute_joint_trajectory([])


def test_cancel_waits_for_execution_and_close_does_not_deadlock(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    arm = bare_arm()
    arm._state_received["joints"] = time.monotonic()
    arm._config["joint"]["completion"]["settle_time_s"] = 0.01
    started = threading.Event()
    release = threading.Event()

    def execute(arm, points, callback):
        started.set()
        assert arm._cancel_requested.wait(2.0)
        assert release.wait(2.0)

    monkeypatch.setattr(motion, "_execute_joint_trajectory", execute)
    monkeypatch.setattr(lifecycle, "_deactivate_controller", lambda arm: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        execution = pool.submit(arm.execute_joint_trajectory, [])
        assert started.wait(1.0)
        cancelling = pool.submit(arm.cancel_motion, timeout_s=2.0)
        assert arm._cancel_requested.wait(1.0)
        assert not cancelling.done()
        release.set()
        assert cancelling.result(timeout=2.0) == execution.result(timeout=2.0)
        assert cancelling.result()["status"] == "cancelled"
    # Closing during a blocking execution must release the command lock.
    arm = bare_arm()
    arm._state_received["joints"] = time.monotonic()
    arm._config["joint"]["completion"]["settle_time_s"] = 0.01
    started.clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        execution = pool.submit(arm.execute_joint_trajectory, [])
        assert started.wait(1.0)
        closing = pool.submit(arm.close)
        closing.result(timeout=2.0)
        assert execution.result(timeout=2.0)["status"] == "cancelled"
    arm.close()
    assert not arm.get_state(required=())["valid"]["joints"]


def test_cancel_timeout_does_not_claim_stopped(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    arm = bare_arm()
    arm._state_received["joints"] = time.monotonic()
    arm._config["joint"]["completion"]["settle_time_s"] = 0.01
    started = threading.Event()
    release = threading.Event()

    def execute(arm, points, callback):
        started.set()
        assert release.wait(2.0)

    monkeypatch.setattr(motion, "_execute_joint_trajectory", execute)
    monkeypatch.setattr(lifecycle, "_deactivate_controller", lambda arm: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        execution = pool.submit(arm.execute_joint_trajectory, [])
        assert started.wait(1.0)
        with pytest.raises(TimeoutError, match="not confirmed"):
            arm.cancel_motion(timeout_s=0.01)
        assert arm._trajectory_running
        assert arm._cancel_requested.is_set()
        release.set()
        assert execution.result(timeout=2.0)["status"] == "cancelled"


def test_stop_stream_requires_feedback_and_deactivates_after_settling(monkeypatch):
    arm = bare_arm()
    arm._state_received["joints"] = time.monotonic()
    arm._stream_space = "joint"
    arm._active_arm_controller = "test"
    arm._config["joint"]["completion"]["settle_time_s"] = 0.01
    operations = []
    monkeypatch.setattr(
        motion, "_publish_joint_target", lambda arm, target: operations.append("hold")
    )
    monkeypatch.setattr(
        lifecycle,
        "_switch_controllers",
        lambda arm, **kwargs: operations.append("deactivate") or True,
    )
    arm._joint_velocities = np.ones(7)
    with pytest.raises(TimeoutError, match="did not stop"):
        arm.stop_stream(timeout_s=0.01)
    assert operations == ["hold"]
    assert arm._stream_space == "joint"
    arm._joint_velocities = np.zeros(7)
    arm.stop_stream(timeout_s=0.1)
    assert operations == ["hold", "hold", "deactivate"]
    assert arm._stream_space is None
    arm._have_joint_velocity = False
    with pytest.raises(RuntimeError, match="velocities are unavailable"):
        arm.wait_until_stopped(timeout_s=0.1)


def test_capabilities_explain_disabled_and_disconnected():
    arm = RoboticArmControlerRos(motion_config={"ee": {"enabled": False}})
    caps = arm.get_capabilities()
    assert caps["ee_stream"] == {
        "enabled": False,
        "ready": False,
        "reason": "Disabled in motion profile",
    }
    assert caps["joint_stream"]["enabled"]
    assert "not connected" in caps["joint_stream"]["reason"]
    arm.close()
    arm.close()


def test_ee_reflex_reports_fault_and_recovery_without_activating(monkeypatch):
    from franka_msgs.msg import FrankaState

    arm = bare_arm()
    arm._config["gripper"]["enabled"] = False
    arm._state_received["joints"] = time.monotonic()
    controller = arm._config["ee"]["streaming"]["controller"]
    monkeypatch.setattr(state, "_list_controllers", lambda arm: [(controller, "inactive")])

    def unexpected_activation(*args):
        raise AssertionError("Faulted hardware must not be activated")

    monkeypatch.setattr(lifecycle, "_activate_arm_controller", unexpected_activation)
    message = FrankaState()
    message.o_t_ee = np.eye(4).reshape(16, order="F").tolist()
    message.robot_mode = FrankaState.ROBOT_MODE_REFLEX
    # The driver can clear current_errors while robot_mode is still Reflex.
    message.last_motion_errors.cartesian_motion_generator_joint_acceleration_discontinuity = True
    state._on_panda_state(arm, message)
    capability = arm.get_capabilities()["ee_stream"]
    assert capability["enabled"] and not capability["ready"]
    assert "Reflex" in capability["reason"]
    assert "joint_acceleration_discontinuity" in capability["reason"]
    with pytest.raises(RuntimeError, match="Reflex"):
        arm.start_stream("ee")
    # Existing targets and completion must also report a newly received fault.
    arm._stream_space = "ee"
    with pytest.raises(RuntimeError, match="Reflex"):
        arm.send_ee_target([0, 0, 0], [0, 0, 0, 1])
    with pytest.raises(RuntimeError, match="Reflex"):
        motion._wait_completion(arm, "ee", (0, np.zeros(3), [0, 0, 0, 1]), None)
    arm._stream_space = None
    # Old last_motion_errors do not block successfully recovered hardware.
    message.robot_mode = FrankaState.ROBOT_MODE_IDLE
    state._on_panda_state(arm, message)
    assert arm.get_capabilities()["ee_stream"]["ready"]


def test_active_controller_is_reused_without_switch(monkeypatch):
    arm = bare_arm()
    controller = arm._config["joint"]["trajectory"]["controller"]
    monkeypatch.setattr(state, "_list_controllers", lambda arm: [(controller, "active")])

    def unexpected_switch(*args, **kwargs):
        raise AssertionError("An active controller must not be activated again")

    monkeypatch.setattr(lifecycle, "_switch_controllers", unexpected_switch)
    assert lifecycle._activate_arm_controller(arm, controller)
    assert arm._active_arm_controller == controller


def test_connect_and_close_preserve_external_node(monkeypatch):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray

    rclpy.init()
    node = Node("arm_public_api_ownership_test")
    node.create_publisher(Float64MultiArray, "/ownership_test", 10)
    original_publishers = list(node.publishers)
    arm = RoboticArmControlerRos(node=node, motion_config={"gripper": {"enabled": False}})
    monkeypatch.setattr(state, "_wait_ready", lambda arm, **kwargs: arm.get_state(required=()))
    monkeypatch.setattr(state, "_list_controllers", lambda arm: [])
    try:
        assert not arm.get_status()["connected"]
        assert arm.connect()["connected"]
        assert arm.connect()["connected"]
        assert len(list(node.publishers)) == len(original_publishers) + 2
        arm.close()
        assert rclpy.ok()
        assert node.executor is None
        assert list(node.publishers) == original_publishers
        assert not list(node.subscriptions)
        assert not list(node.clients)
        with pytest.raises(RuntimeError, match="closed"):
            arm.connect()
    finally:
        arm.close()
        node.destroy_node()
        rclpy.shutdown()


def controller_manager(monkeypatch, arm, initial_state):
    name = arm._config["joint"]["streaming"]["controller"]
    controllers = {} if initial_state is None else {name: initial_state}
    calls = []
    arm._load_client, arm._configure_client, arm._switch_client = "load", "configure", "switch"
    arm._node = SimpleNamespace(get_logger=lambda: SimpleNamespace(info=lambda _: None))
    arm._state_received["joints"] = time.monotonic()
    monkeypatch.setattr(state, "_list_controllers", lambda _: list(controllers.items()))

    def service(_, client, request, **kwargs):
        calls.append(client)
        if client == "load":
            controllers[request.name] = "unconfigured"
        elif client == "configure":
            assert controllers[request.name] == "unconfigured"
            controllers[request.name] = "inactive"
        else:
            for target in request.activate_controllers:
                assert controllers[target] == "inactive"
                controllers[target] = "active"
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(lifecycle, "_call_service", service)
    monkeypatch.setattr(motion, "_publish_joint_target", lambda *_: calls.append("target"))
    return calls


@pytest.mark.parametrize("initial, expected", [
    (None, ["load", "configure", "switch", "target"]),
    ("unconfigured", ["configure", "switch", "target"]),
    ("inactive", ["switch", "target"]),
    ("active", ["target"]),
])
def test_public_stream_configures_before_motion(monkeypatch, initial, expected):
    arm = bare_arm()
    calls = controller_manager(monkeypatch, arm, initial)
    arm.start_stream("joint")
    assert calls == expected
    assert arm.get_status()["stream_space"] == "joint"


@pytest.mark.parametrize("reply, error", [(None, TimeoutError), (SimpleNamespace(ok=False), RuntimeError)])
def test_configuration_failure_never_activates_or_publishes(monkeypatch, reply, error):
    arm = bare_arm()
    calls = controller_manager(monkeypatch, arm, "unconfigured")

    def service(*args, **kwargs):
        calls.append("configure")
        return reply

    monkeypatch.setattr(lifecycle, "_call_service", service)
    with pytest.raises(error, match="configur"):
        arm.start_stream("joint")
    assert calls == ["configure"]
    assert arm._stream_space is None


def test_joint_action_is_discovered_after_configuration(monkeypatch):
    arm = bare_arm()
    controller = arm._config["joint"]["trajectory"]["controller"]
    arm._config["joint"]["streaming"]["controller"] = controller
    calls = controller_manager(monkeypatch, arm, None)

    def wait_for_server(**kwargs):
        assert calls == ["load", "configure"]
        calls.append("action")
        return False

    monkeypatch.setattr(motion, "ActionClient", lambda *args: SimpleNamespace(
        wait_for_server=wait_for_server, destroy=lambda: None
    ))
    with pytest.raises(RuntimeError, match="action server"):
        arm.execute_joint_trajectory(points())
    assert calls == ["load", "configure", "action"]
    assert not arm._trajectory_running


def test_close_joins_real_executor_workers_across_reconnects(monkeypatch):
    import rclpy

    monkeypatch.setattr(state, "_wait_ready", lambda arm, **kwargs: arm.get_state(required=()))
    monkeypatch.setattr(state, "_list_controllers", lambda _: [])
    for _ in range(3):
        arm = RoboticArmControlerRos(motion_config={"gripper": {"enabled": False}})
        try:
            arm.connect()
            callback = threading.Event()
            arm._executor.create_task(callback.set)
            assert callback.wait(2.0)
            arm.close()
            arm.close()
            assert not arm._spin_thread.is_alive()
            assert not arm._gripper_thread.is_alive()
            assert arm._executor._executor._threads
            assert all(not thread.is_alive() for thread in arm._executor._executor._threads)
            assert arm._ros_entities == []
            assert not rclpy.ok()
        finally:
            arm.close()


def test_external_nondefault_context_is_preserved(monkeypatch):
    import rclpy
    from rclpy.context import Context
    from rclpy.node import Node

    context = Context()
    context.init()
    node = Node("arm_external_context_test", context=context)
    arm = RoboticArmControlerRos(node=node, motion_config={"gripper": {"enabled": False}})
    monkeypatch.setattr(state, "_wait_ready", lambda arm, **kwargs: arm.get_state(required=()))
    monkeypatch.setattr(state, "_list_controllers", lambda _: [])
    try:
        arm.connect()
        assert arm._executor.context is context
        arm.close()
        assert context.ok()
        assert not rclpy.ok()
    finally:
        arm.close()
        node.destroy_node()
        context.shutdown()
        context.destroy()
