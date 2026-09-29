import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from control.ee_safety import EESafety
from control.hitl.collector import HumanInTheLoopCollector
from control.hitl.env import DummyRobotEnv
from control.hitl.recording import LocalEpisodeRecorder
from control.hitl.types import EpisodeStats, RobotCommand, StepRecord
from control.recording_writer import AsyncDatasetFrames, FreshCameraPair
from control.util import msgpack_numpy
from control.util.lerobot_util import _check_dataset_fps
from control.vr_input import DualTriggerDeadman


class _Dataset:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        time.sleep(0.002)
        self.frames.append(frame)


def test_image_writer_drain_preserves_order():
    dataset = _Dataset()
    writer = AsyncDatasetFrames(dataset)
    for index in range(12):
        writer.submit({"index": index})
    writer.drain()
    assert [frame["index"] for frame in dataset.frames] == list(range(12))
    writer.close()


def test_100_hz_polling_keeps_only_fresh_30_hz_camera_pairs():
    gate = FreshCameraPair()
    accepted = 0
    for tick in range(100):
        camera_frame = tick * 30 // 100 + 1
        capture_ns = camera_frame * 33_333_333
        accepted += gate.accept(capture_ns, capture_ns)
    assert accepted == 30
    assert not gate.accept(30 * 33_333_333, 30 * 33_333_333)


def test_old_20_fps_dataset_cannot_receive_new_30_fps_frames(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"fps": 20}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="new dataset repo_id/date"):
        _check_dataset_fps(tmp_path, 30)


class _SlowPolicy:
    server_metadata = {}

    def __init__(self):
        self.calls = 0

    def infer(self, _obs):
        self.calls += 1
        time.sleep(0.08)
        return {"actions": np.array([0.005, 0.0, 0.0, 0.0], dtype=np.float32)}

    def close(self):
        pass


class _Transitions:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def close(self):
        pass


def test_slow_policy_does_not_set_servo_step_rate():
    env = DummyRobotEnv([0.0] * 7, image_size=8)
    policy = _SlowPolicy()
    transitions = _Transitions()
    safety = EESafety(
        min_xyz=np.array([0.0, -1.0, 0.0]),
        max_xyz=np.array([1.0, 1.0, 1.0]),
        max_step_xyz=np.array([0.002, 0.002, 0.002]),
    )
    loop = HumanInTheLoopCollector(
        env=env,
        safety=safety,
        fps=100.0,
        policy_fps=30.0,
        episode_time_s=10.0,
        policy=policy,
        transitions=transitions,
        vr_cfg=SimpleNamespace(),
        max_episodes=1,
        max_steps=35,
    )
    started = time.monotonic()
    loop.run()
    elapsed = time.monotonic() - started
    assert elapsed < 0.9
    assert policy.calls >= 1
    assert env.ee_pos[0] > 0.4
    replay_steps = [m for m in transitions.messages if m["type"] == "transition"]
    assert 6 <= len(replay_steps) <= 15


def test_vr_takeover_records_fork_and_continuation(tmp_path):
    env = DummyRobotEnv([0.0] * 7, image_size=4)
    safety = EESafety(
        min_xyz=np.array([0.0, -1.0, 0.0]),
        max_xyz=np.array([1.0, 1.0, 1.0]),
        max_step_xyz=np.array([0.002, 0.002, 0.002]),
    )
    recorder = LocalEpisodeRecorder(tmp_path)
    collector = HumanInTheLoopCollector(
        env=env,
        safety=safety,
        fps=100,
        policy_fps=30,
        episode_time_s=1,
        policy=None,
        transitions=None,
        vr_cfg=SimpleNamespace(),
        recorder=recorder,
    )
    collector.on_episode_start(1)
    for step, source in enumerate(("policy", "vr", "policy")):
        obs = {"observation.state": np.array([step], dtype=np.float32)}
        next_obs = {"observation.state": np.array([step + 1], dtype=np.float32)}
        collector.on_transition(
            StepRecord(
                episode_index=1,
                step_index=step,
                timestamp_monotonic_ns=step + 100,
                ee_pos_start_xyz=np.array([0.4 + step * 0.001, 0.0, 0.4]),
                ee_quat_start_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                obs=obs,
                command=RobotCommand(
                    ee_delta_xyz=np.array([0.001, 0.0, 0.0]),
                    intervening=source == "vr",
                    control_source=source,
                ),
                executed_xyz=np.array([0.001, 0.0, 0.0]),
                gripper_close=False,
                reward=0.0,
                next_obs=next_obs,
                done=step == 2,
                truncated=False,
                clipped=False,
                success=False,
            )
        )
    collector.on_episode_end(EpisodeStats(1, 0.0, False, 3, 0))
    collector.teardown()

    events = [
        json.loads(line)
        for line in (recorder.run_dir / "events.jsonl").read_text().splitlines()
    ]
    fork = next(event for event in events if event["type"] == "branch_start")
    release = next(event for event in events if event["type"] == "branch_release")
    assert (fork["fork_step_index"], fork["branch_id"], fork["parent_branch_id"]) == (
        1,
        1,
        0,
    )
    assert (release["step_index"], release["branch_id"]) == (2, 1)
    with (recorder.run_dir / "episode_000001.msgpack").open("rb") as stream:
        records = list(msgpack_numpy.Unpacker(stream, raw=False))
    transitions = [record for record in records if record["type"] == "transition"]
    assert [(item["control_source"], item["branch_id"]) for item in transitions] == [
        ("policy", 0),
        ("vr", 1),
        ("policy", 1),
    ]
    assert transitions[1]["state"]["observation.state"][0] == 1


def test_vr_deadman_requires_both_fresh_triggers():
    deadman = DualTriggerDeadman(hold_s=0.5, stale_after_s=0.5)
    deadman.update_left(True, 1.0)
    deadman.update_right(True, 1.0)
    assert not deadman.active(1.4)
    deadman.update_left(True, 1.4)
    deadman.update_right(True, 1.4)
    assert deadman.active(1.51)
    assert not deadman.active(2.0)
    deadman.update_left(False, 2.1)
    deadman.update_right(True, 2.1)
    assert not deadman.active(2.2)
