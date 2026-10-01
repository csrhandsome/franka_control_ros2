"""Contract checks for the plain 30 Hz OpenPI inference entry point."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from typing import ClassVar

import numpy as np
import pytest
import websockets.asyncio.server
import websockets.exceptions

from control.robot_config import config_path, load_mapping
from control.util import msgpack_numpy
from inference import (
    ACTION_DIM,
    ACTION_HORIZON,
    PERIOD_NS,
    WARMUP_CHUNKS,
    _command_gripper,
    action_at,
    build_observation,
    checked_ee_target,
    main,
    parse_action_plan,
    run,
    validate_policy_metadata,
)

FRONT_FILL = 7
WRIST_FILL = 9
# Keys the training config's RepackTransform reads from the request.
REPACK_KEYS = {
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/ee_pose",
    "observation/gripper_position",
    "prompt",
}
POLICY_ACTION = np.tile(
    np.array([0.01, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    (ACTION_HORIZON, 1),
)


def _frames():
    front = np.full((224, 224, 3), FRONT_FILL, dtype=np.uint8)
    wrist = np.full((224, 224, 3), WRIST_FILL, dtype=np.uint8)
    stamp = types.SimpleNamespace(host_capture_monotonic_ns=time.monotonic_ns())
    return front, wrist, stamp, stamp


class FakePolicy:
    server_metadata: ClassVar[dict] = {
        "action_fps": 30.0,
        "action_horizon": 16,
        "action_space": "ee",
        "action_dim": 7,
    }

    def __init__(self, **_kwargs):
        self.observations: list[dict] = []
        self.closed = False

    def infer(self, observation):
        self.observations.append(observation)
        return {"actions": POLICY_ACTION.copy()}

    def close(self):
        self.closed = True


class FakeCameras:
    def __init__(self, **_kwargs):
        self.closed = False

    def wait_for_frames(self, **_kwargs):
        return None

    def get_frames(self):
        return _frames()

    def close(self):
        self.closed = True


class StaleCameras(FakeCameras):
    def get_frames(self):
        front, wrist, _, _ = _frames()
        stamp = types.SimpleNamespace(
            host_capture_monotonic_ns=time.monotonic_ns() - 5 * 10**9
        )
        return front, wrist, stamp, stamp


class FakeArm:
    def __init__(self, **_kwargs):
        self.joints = np.zeros(7)
        self.position = np.array([0.0, 0.0, 0.5])
        self.gripper_position = 0.04
        self.gripper_busy = False
        self.joint_state_ready = True
        self.ee_pose_ready = True
        self.ee_streaming_active = False
        self.calls: list[str] = []
        self.commands: list[np.ndarray] = []
        self.gripper_calls: list[tuple[str, dict]] = []
        self.closed = False

    @property
    def state(self):
        transform = np.eye(4)
        transform[:3, 3] = self.position
        return {
            "joint_positions": self.joints.copy(),
            "end_effector_pose": transform,
            "gripper_position": self.gripper_position,
        }

    def start_ee_streaming(self):
        self.calls.append("start_ee_streaming")
        self.ee_streaming_active = True

    def move_to_start(self):
        self.calls.append("move_to_start")

    def set_ee_control(self, position, quaternion, _qpos):
        self.calls.append("set_ee_control")
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.commands.append(self.position.copy())

    def gripper_open(self, **kwargs):
        self.calls.append("gripper_open")
        self.gripper_calls.append(("open", kwargs))
        return True

    def gripper_close(self, **kwargs):
        self.calls.append("gripper_close")
        self.gripper_calls.append(("close", kwargs))
        return True

    def cleanup(self):
        self.closed = True


def _recorder(created, key, factory):
    def build(**kwargs):
        instance = factory(**kwargs)
        created[key] = instance
        return instance

    return build


def _install_ros_stubs(
    monkeypatch, *, policy=FakePolicy, cameras=FakeCameras, arm=FakeArm
):
    """Replace the modules ``run`` imports; the real ones need ROS hardware."""
    created: dict[str, object] = {}
    stubs = [
        (
            "control.dual_camera_manager_ros",
            "DualRealsenseManagerRos",
            "cameras",
            cameras,
        ),
        ("control.robotic_arm_controller_ros", "RoboticArmControlerRos", "arm", arm),
    ]
    if policy is not None:
        stubs.append(("control.hitl.policy", "PolicyClient", "policy", policy))
    for module_name, attr, key, factory in stubs:
        module = types.ModuleType(module_name)
        setattr(module, attr, _recorder(created, key, factory))
        monkeypatch.setitem(sys.modules, module_name, module)
    return created


@pytest.fixture
def config():
    return load_mapping(config_path(stage="inference", robot="franka"))


def test_observation_matches_franka_ros_repack_contract():
    front = np.full((224, 224, 3), FRONT_FILL, dtype=np.uint8)
    wrist = np.full((224, 224, 3), WRIST_FILL, dtype=np.uint8)
    payload = build_observation(front, wrist, np.arange(6), 1.0, "pick", 12)
    assert set(payload) == REPACK_KEYS
    assert payload["observation/gripper_position"].shape == (1,)
    with pytest.raises(ValueError):
        build_observation(front[:112], wrist, np.arange(6), 1.0, "pick", 0)
    with pytest.raises(ValueError):
        build_observation(front, wrist, np.arange(7), 1.0, "pick", 0)
    with pytest.raises(ValueError):
        build_observation(front, wrist, np.arange(6), 1.2, "pick", 0)


def test_policy_metadata_and_chunk_timing():
    validate_policy_metadata(FakePolicy.server_metadata)
    with pytest.raises(ValueError):
        validate_policy_metadata({**FakePolicy.server_metadata, "action_fps": 100.0})
    with pytest.raises(ValueError):
        validate_policy_metadata(
            {**FakePolicy.server_metadata, "action_space": "joint"}
        )
    with pytest.raises(ValueError):
        validate_policy_metadata({})
    actions = np.tile(
        np.arange(ACTION_HORIZON, dtype=np.float32)[:, None], (1, ACTION_DIM)
    )
    plan = parse_action_plan({"actions": actions}, observation_ns=10 * PERIOD_NS)
    assert action_at(plan, 10 * PERIOD_NS)[0] is None
    assert action_at(plan, 11 * PERIOD_NS)[1] == 0
    assert action_at(plan, 14 * PERIOD_NS)[0][0] == 3
    assert action_at(plan, 27 * PERIOD_NS)[0] is None


def test_invalid_or_large_ee_targets_are_rejected():
    from control.robot_state import EEPose

    current = EEPose.from_vector(np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        checked_ee_target(
            np.array([0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]), current, 0.03, 0.15
        )
    with pytest.raises(ValueError):
        parse_action_plan({"actions": np.full((16, 7), np.nan)}, 0)
    with pytest.raises(ValueError):
        parse_action_plan({"actions": np.zeros((8, 7))}, 0)


def test_gripper_command_uses_collector_logical_open_state():
    class GripperArm:
        def __init__(self, busy=False):
            self.gripper_busy = busy
            self.calls: list[tuple[str, dict]] = []

        def gripper_open(self, **kwargs):
            self.calls.append(("open", kwargs))
            return True

        def gripper_close(self, **kwargs):
            self.calls.append(("close", kwargs))
            return True

    arm = GripperArm()
    with pytest.raises(ValueError):
        _command_gripper(arm, 1.2, None)
    assert arm.calls == []
    assert _command_gripper(arm, 0.9, None) == 1.0
    assert arm.calls == [("open", {"width": 0.05, "wait": False})]
    arm.calls.clear()
    assert _command_gripper(arm, 0.8, 1.0) == 1.0
    assert arm.calls == []
    assert _command_gripper(arm, 0.0, 1.0) == 0.0
    assert arm.calls == [("close", {"wait": False})]
    busy = GripperArm(busy=True)
    assert _command_gripper(busy, 0.0, 1.0) == 1.0
    assert busy.calls == []


def test_plain_inference_commands_ee_without_writing_files(
    monkeypatch, tmp_path, config
):
    created = _install_ros_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    run(config, host=None, port=None, prompt=None, max_steps=10, move_to_start=False)

    arm = created["arm"]
    assert created["policy"].observations
    assert any(np.allclose(command, [0.01, 0.0, 0.5]) for command in arm.commands)
    assert arm.calls.index("start_ee_streaming") < arm.calls.index("set_ee_control")
    assert "move_to_start" not in arm.calls
    assert created["policy"].closed and arm.closed and created["cameras"].closed
    assert list(tmp_path.iterdir()) == []


def test_warmup_finishes_before_camera_and_arm_setup(monkeypatch, config):
    created = {}

    class CheckingCameras(FakeCameras):
        def __init__(self, **kwargs):
            assert len(created["policy"].observations) == WARMUP_CHUNKS
            super().__init__(**kwargs)

    class CheckingArm(FakeArm):
        def __init__(self, **kwargs):
            assert len(created["policy"].observations) == WARMUP_CHUNKS
            super().__init__(**kwargs)

    created = _install_ros_stubs(monkeypatch, cameras=CheckingCameras, arm=CheckingArm)
    run(config, host=None, port=None, prompt=None, max_steps=1, move_to_start=False)

    warmup_requests = created["policy"].observations[:WARMUP_CHUNKS]
    assert len(warmup_requests) == WARMUP_CHUNKS
    for observation in warmup_requests:
        assert set(observation) == REPACK_KEYS
        assert observation["prompt"] == ""
        assert not observation["observation/exterior_image_1_left"].any()
        assert not observation["observation/wrist_image_left"].any()
        assert not observation["observation/ee_pose"].any()
        assert not observation["observation/gripper_position"].any()


def test_invalid_warmup_chunk_aborts_before_camera_and_arm_setup(monkeypatch, config):
    class InvalidThirdChunk(FakePolicy):
        def infer(self, observation):
            self.observations.append(observation)
            if len(self.observations) == 3:
                return {"actions": np.zeros((8, ACTION_DIM), dtype=np.float32)}
            return {"actions": POLICY_ACTION.copy()}

    created = _install_ros_stubs(monkeypatch, policy=InvalidThirdChunk)
    with pytest.raises(RuntimeError, match="Policy warmup failed on request 3/5"):
        run(config, host=None, port=None, prompt=None, max_steps=1, move_to_start=False)

    assert len(created["policy"].observations) == 3
    assert created["policy"].closed
    assert "cameras" not in created and "arm" not in created


def test_replanning_keeps_inference_off_the_30_hz_path(monkeypatch, tmp_path, config):
    created = _install_ros_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    replan = int(config["inference"]["replan_every_steps"])
    steps = replan * 4
    run(config, host=None, port=None, prompt=None, max_steps=steps, move_to_start=False)

    policy = created["policy"]
    assert policy.observations
    # Every tick still published an EE target, so the arm never stalls while inferring,
    # plus the shutdown hold published on exit.
    assert len(created["arm"].commands) == steps + 1


def test_stale_camera_frames_abort_and_hold_position(monkeypatch, tmp_path, config):
    created = _install_ros_stubs(monkeypatch, cameras=StaleCameras)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="stale"):
        run(
            config, host=None, port=None, prompt=None, max_steps=10, move_to_start=False
        )

    arm = created["arm"]
    assert len(created["policy"].observations) == WARMUP_CHUNKS
    assert arm.commands and np.allclose(arm.commands[-1], [0.0, 0.0, 0.5])
    assert arm.gripper_calls == []
    assert created["policy"].closed and arm.closed and created["cameras"].closed


def test_large_ee_jump_aborts_and_holds_position(monkeypatch, tmp_path, config):
    class JumpingPolicy(FakePolicy):
        def infer(self, observation):
            self.observations.append(observation)
            action = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return {"actions": np.tile(action, (ACTION_HORIZON, 1))}

    created = _install_ros_stubs(monkeypatch, policy=JumpingPolicy)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="differs from measured"):
        run(
            config, host=None, port=None, prompt=None, max_steps=10, move_to_start=False
        )

    arm = created["arm"]
    assert all(np.allclose(command, [0.0, 0.0, 0.5]) for command in arm.commands)
    assert arm.gripper_calls == []
    assert created["policy"].closed and arm.closed and created["cameras"].closed


def test_policy_failure_stops_inference_and_releases_resources(
    monkeypatch, tmp_path, config
):
    class FailingPolicy(FakePolicy):
        def infer(self, observation):
            if observation["prompt"] == "":
                return super().infer(observation)
            raise RuntimeError("server exploded")

    created = _install_ros_stubs(monkeypatch, policy=FailingPolicy)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Policy inference failed"):
        run(
            config, host=None, port=None, prompt=None, max_steps=10, move_to_start=False
        )

    arm = created["arm"]
    assert arm.commands and np.allclose(arm.commands[-1], [0.0, 0.0, 0.5])
    assert created["policy"].closed and arm.closed and created["cameras"].closed


def test_missing_robot_state_refuses_to_run(monkeypatch, tmp_path, config):
    class UnreadyArm(FakeArm):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.joint_state_ready = False

    created = _install_ros_stubs(monkeypatch, arm=UnreadyArm)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="No measured joint or EE state"):
        run(
            config, host=None, port=None, prompt=None, max_steps=10, move_to_start=False
        )

    assert "start_ee_streaming" not in created["arm"].calls
    assert (
        created["policy"].closed and created["arm"].closed and created["cameras"].closed
    )


def test_incompatible_policy_metadata_fails_before_moving_the_robot(
    monkeypatch, tmp_path, config
):
    class HundredHzPolicy(FakePolicy):
        server_metadata: ClassVar[dict] = {
            **FakePolicy.server_metadata,
            "action_fps": 100.0,
        }

    created = _install_ros_stubs(monkeypatch, policy=HundredHzPolicy)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Incompatible policy metadata"):
        run(
            config, host=None, port=None, prompt=None, max_steps=10, move_to_start=False
        )

    assert created["policy"].closed
    assert "cameras" not in created and "arm" not in created


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: c["camera"].update({"camera_backend": "usb"}), id="camera-backend"
        ),
        pytest.param(lambda c: c["camera"].update({"image_hw": 128}), id="image-size"),
        pytest.param(
            lambda c: c["camera"].update({"camera_fps": 15.0}), id="camera-fps"
        ),
        pytest.param(
            lambda c: c["gripper"].update({"gripper_type": "dh5"}), id="gripper-type"
        ),
        pytest.param(
            lambda c: c["inference"].update({"action_fps": 60.0}), id="action-fps"
        ),
        pytest.param(
            lambda c: c["inference"].update({"replan_every_steps": 16}),
            id="replan-horizon",
        ),
        pytest.param(
            lambda c: c["inference"].update({"replan_every_steps": 0}), id="replan-zero"
        ),
        pytest.param(
            lambda c: c["inference"].update({"max_ee_translation_step_m": 0.0}),
            id="ee-step",
        ),
        pytest.param(
            lambda c: c["inference"].update({"max_camera_age_s": 0.0}), id="camera-age"
        ),
        pytest.param(lambda c: c["inference"].update({"prompt": "  "}), id="prompt"),
    ],
)
def test_config_validation_rejects_bad_settings(monkeypatch, config, mutate):
    created = _install_ros_stubs(monkeypatch)
    mutate(config)
    with pytest.raises(ValueError):
        run(config, host=None, port=None, prompt=None, max_steps=1, move_to_start=False)
    assert created == {}


class StubPolicyServer:
    """Mirror of openpi-force's WebsocketPolicyServer wire behavior."""

    def __init__(self, *, metadata=None, actions=None):
        self.metadata = (
            metadata
            if metadata is not None
            else {
                "action_fps": 30.0,
                "action_horizon": 16,
                "action_space": "ee",
                "action_dim": 7,
                "checkpoint": "pi0_base_ee_ros_30hz",
            }
        )
        self.actions = POLICY_ACTION if actions is None else actions
        self.observations: list[dict] = []
        self.port: int | None = None
        self.error: BaseException | None = None
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._serve, name="stub-policy", daemon=True
        )
        self._thread.start()
        assert self._ready.wait(10.0), "stub policy server did not start"
        if self.error is not None:
            raise self.error
        return self

    def __exit__(self, *_exc):
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _serve(self):
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # surfaced to the test thread by __enter__
            self.error = exc
            self._ready.set()

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with websockets.asyncio.server.serve(
            self._handler, "127.0.0.1", 0, compression=None, max_size=None
        ) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        try:
            while True:
                self.observations.append(msgpack_numpy.unpackb(await websocket.recv()))
                await websocket.send(
                    packer.pack(
                        {"actions": self.actions, "server_timing": {"infer_ms": 1.0}}
                    )
                )
        except websockets.exceptions.ConnectionClosed:
            return


def test_real_policy_client_round_trips_observations(monkeypatch, tmp_path, config):
    """End-to-end over the real websocket client against a server-shaped peer."""
    created = _install_ros_stubs(monkeypatch, policy=None)
    with StubPolicyServer() as server:
        monkeypatch.chdir(tmp_path)
        run(
            config,
            host="127.0.0.1",
            port=server.port,
            prompt="pick the cube",
            max_steps=12,
            move_to_start=False,
        )
    assert server.error is None
    assert WARMUP_CHUNKS + 1 <= len(server.observations) <= WARMUP_CHUNKS + 3
    assert all(obs["prompt"] == "" for obs in server.observations[:WARMUP_CHUNKS])
    observation = server.observations[WARMUP_CHUNKS]
    assert set(observation) == REPACK_KEYS
    assert observation["prompt"] == "pick the cube"
    assert observation["observation/exterior_image_1_left"].dtype == np.uint8
    assert observation["observation/exterior_image_1_left"].shape == (224, 224, 3)
    assert int(observation["observation/exterior_image_1_left"][0, 0, 0]) == FRONT_FILL
    assert int(observation["observation/wrist_image_left"][0, 0, 0]) == WRIST_FILL
    assert observation["observation/ee_pose"].shape == (6,)
    assert observation["observation/gripper_position"].shape == (1,)
    assert observation["observation/gripper_position"][0] == 1.0
    # The server's 16x7 reply reached the arm as absolute Cartesian targets.
    assert any(
        np.allclose(command, [0.01, 0.0, 0.5]) for command in created["arm"].commands
    )
    assert len(created["arm"].gripper_calls) == 1
    name, kwargs = created["arm"].gripper_calls[0]
    assert name == "open" and kwargs["wait"] is False
    assert kwargs["width"] == pytest.approx(0.05, abs=1e-6)
    assert list(tmp_path.iterdir()) == []


def test_cli_exposes_overrides_and_rejects_bad_steps(monkeypatch):
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    created = _install_ros_stubs(monkeypatch)
    with pytest.raises(ValueError, match="max_steps"):
        main(["--robot", "franka", "--max-steps", "-1"])
    assert created == {}
