"""Policy rollout with VR takeover, branch markers, and local trajectory capture."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from control.ee_safety import EESafety
from control.hitl.env import RobotEnv, make_env
from control.hitl.loop import RobotLoop
from control.hitl.policy import PolicyClient
from control.hitl.recording import LocalEpisodeRecorder
from control.hitl.transition import AsyncTransitionSink, TransitionClient
from control.hitl.types import (
    EpisodeStats,
    LoopEvents,
    RobotCommand,
    StepContext,
    StepRecord,
    parse_ee_gripper_action,
)
from control.robot_config import section as _section
from control.util.pose import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

logger = logging.getLogger(__name__)


class _ButtonEdge:
    def __init__(self) -> None:
        self._prev = False

    def rising(self, pressed: bool) -> bool:
        rose = bool(pressed) and not self._prev
        self._prev = bool(pressed)
        return rose


def _maybe_vr(vr_cfg: SimpleNamespace):
    if not bool(getattr(vr_cfg, "enable_vr", False)):
        return None
    try:
        from control.vr_input import VRInputRos

        return VRInputRos(
            pose_topic=str(vr_cfg.vr_pose_topic),
            left_joy_topic=str(vr_cfg.vr_left_joy_topic),
            right_joy_topic=str(vr_cfg.vr_right_joy_topic),
            long_press_s=float(vr_cfg.vr_long_press_s),
            assume_arm_enabled=bool(vr_cfg.vr_assume_arm_enabled),
        )
    except Exception as exc:
        logger.warning(
            "[HITL] VR disabled, initialization failed: %s", exc, exc_info=True
        )
        return None


def _maybe_vr_mapper(vr_cfg: SimpleNamespace, safety: EESafety):
    from control.vr_input import VREEPoseMapper

    max_step = float(np.min(np.asarray(safety.max_step_xyz, dtype=np.float64)))
    return VREEPoseMapper(
        translation_scale=float(getattr(vr_cfg, "vr_translation_scale", 0.04)),
        rotation_scale=float(getattr(vr_cfg, "vr_rotation_scale", 0.10)),
        max_translation_step=float(
            getattr(vr_cfg, "vr_max_translation_step", max_step)
        ),
        max_rotation_step=float(getattr(vr_cfg, "vr_max_rotation_step", 0.04)),
        translation_limit=float(getattr(vr_cfg, "vr_translation_limit", 0.5)),
        rotation_limit=float(getattr(vr_cfg, "vr_rotation_limit", 1.2)),
        sensitivity=float(getattr(vr_cfg, "vr_sensitivity", 1.0)),
        enable_rotation=bool(getattr(vr_cfg, "vr_enable_rotation", True)),
    )


class HumanInTheLoopCollector(RobotLoop):
    """Record policy, human intervention, and the path after control returns."""

    def __init__(
        self,
        *,
        env: RobotEnv,
        safety: EESafety,
        fps: float,
        policy_fps: float,
        episode_time_s: float,
        policy: PolicyClient,
        transitions: TransitionClient | None,
        vr_cfg: SimpleNamespace,
        vr: Any = None,
        recorder: LocalEpisodeRecorder | None = None,
        max_episodes: int = 0,
        max_steps: int = 0,
    ) -> None:
        super().__init__(
            env=env,
            safety=safety,
            fps=fps,
            episode_time_s=episode_time_s,
            policy=policy,
            max_episodes=max_episodes,
            max_steps=max_steps,
        )
        self._transitions = AsyncTransitionSink(transitions) if transitions else None
        self._recorder = recorder
        self._policy_fps = float(policy_fps)
        if self._policy_fps <= 0 or self._policy_fps > fps:
            raise ValueError(
                "policy_fps must be positive and no greater than servo fps"
            )
        self._policy_lock = threading.Lock()
        self._policy_stop = threading.Event()
        self._policy_thread: threading.Thread | None = None
        self._policy_observation: dict | None = None
        self._policy_generation = 0
        self._policy_allowed = True
        self._policy_result: dict | None = None
        self._policy_result_seq = 0
        self._policy_consumed_seq = 0
        self._policy_target: np.ndarray | None = None
        self._vr_cfg = vr_cfg
        self._vr = vr
        self._vr_mapper = None
        if self._vr is not None:
            self._vr_mapper = _maybe_vr_mapper(vr_cfg, safety)
        self._y_edge = _ButtonEdge()
        self._x_edge = _ButtonEdge()
        self._gripper_closed = False
        self._prev_arm_enabled = False
        self._intervention_steps = 0
        self._transition_period_s = 1.0 / self._policy_fps
        self._transition_pending: dict | None = None
        self._next_transition_due = 0.0
        self._branch_id = 0
        self._branch_start_step: int | None = None
        self._was_intervening = False

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        dry_run: bool,
        max_episodes: int = 0,
        max_steps: int = 0,
    ) -> HumanInTheLoopCollector:
        policy_cfg = _section(config, "policy_server")
        trans_cfg = _section(config, "transition_server")
        recording_cfg = _section(config, "recording")
        control_cfg = _section(config, "control")
        vr_cfg = _section(config, "vr")
        env = make_env(config, dry_run=dry_run)
        safety = EESafety.from_config(
            dict(config["control"]["end_effector_bounds"]),
            dict(config["control"]["end_effector_step_sizes"]),
        )
        policy = PolicyClient(host=str(policy_cfg.host), port=int(policy_cfg.port))
        transitions = (
            TransitionClient(host=str(trans_cfg.host), port=int(trans_cfg.port))
            if bool(getattr(trans_cfg, "enabled", False))
            else None
        )
        recorder = (
            LocalEpisodeRecorder(getattr(recording_cfg, "output_dir", "data/hitl"))
            if bool(getattr(recording_cfg, "enabled", True))
            else None
        )
        vr = None if dry_run else _maybe_vr(vr_cfg)
        return cls(
            env=env,
            safety=safety,
            fps=float(control_cfg.fps),
            policy_fps=float(
                getattr(control_cfg, "policy_fps", min(float(control_cfg.fps), 30.0))
            ),
            episode_time_s=float(control_cfg.control_time_s),
            policy=policy,
            transitions=transitions,
            recorder=recorder,
            vr_cfg=vr_cfg,
            vr=vr,
            max_episodes=max_episodes,
            max_steps=max_steps,
        )

    def setup(self) -> None:
        self._policy_thread = threading.Thread(
            target=self._policy_loop, name="policy-inference", daemon=True
        )
        self._policy_thread.start()
        if self._vr is not None:
            self._vr.start()
        if self._recorder is not None:
            logger.info("[HITL] recording to %s", self._recorder.run_dir)
        if self.policy is not None:
            logger.info("[HITL] policy metadata: %s", self.policy.server_metadata)
        if self._vr is not None:
            logger.info(
                "[HITL] VR subscribed: hold both triggers to take over, "
                "release to give control back to the policy"
            )
            logger.info("[HITL] VR keys: left Y=success  left X=failure")

    def teardown(self) -> None:
        self._policy_stop.set()
        if self._policy_thread is not None:
            self._policy_thread.join(timeout=2.0)
        try:
            self._flush_transition()
        finally:
            if self._vr is not None:
                with contextlib.suppress(Exception):
                    self._vr.stop()
            try:
                if self._transitions is not None:
                    self._transitions.close()
            finally:
                if self._recorder is not None:
                    self._recorder.close()

    def _policy_loop(self) -> None:
        period_s = 1.0 / self._policy_fps
        while not self._policy_stop.is_set():
            started = time.monotonic()
            with self._policy_lock:
                obs = self._policy_observation
                generation = self._policy_generation
                allowed = self._policy_allowed
            if obs is not None and allowed and self.policy is not None:
                try:
                    result = self.policy.infer(obs)
                except Exception:
                    logger.exception("[HITL] policy inference failed")
                    self.request_stop()
                    return
                with self._policy_lock:
                    if self._policy_generation == generation and self._policy_allowed:
                        self._policy_result = result
                        self._policy_result_seq += 1
            self._policy_stop.wait(max(0.0, period_s - (time.monotonic() - started)))

    def on_episode_start(self, episode_index: int) -> None:
        if self._vr_mapper is not None:
            self._vr_mapper.reset()
        self._prev_arm_enabled = False
        self._intervention_steps = 0
        self._transition_pending = None
        self._next_transition_due = 0.0
        self._gripper_closed = False
        self._policy_target = None
        self._policy_consumed_seq = 0
        self._branch_id = 0
        self._branch_start_step = None
        self._was_intervening = False
        self._record(
            {
                "type": "episode_start",
                "schema_version": 1,
                "episode_index": episode_index,
                "timestamp_monotonic_ns": time.monotonic_ns(),
            }
        )
        with self._policy_lock:
            self._policy_generation += 1
            self._policy_observation = None
            self._policy_result = None
            self._policy_result_seq = 0
            self._policy_allowed = True

    def poll_events(self) -> LoopEvents:
        vr_input = self._vr.latest if self._vr is not None else None
        if vr_input is None:
            return LoopEvents()
        return LoopEvents(
            success=bool(self._y_edge.rising(vr_input.y_pressed)),
            failure=bool(self._x_edge.rising(vr_input.x_pressed)),
        )

    def select_action(self, obs: dict, context: StepContext) -> RobotCommand:
        del context
        vr_input = self._vr.latest if self._vr is not None else None
        arm_enabled = bool(vr_input is not None and vr_input.arm_enabled)
        if self._vr_mapper is not None:
            if arm_enabled and not self._prev_arm_enabled:
                self._vr_mapper.reset()
                logger.info("[HITL] VR deadman engaged, policy paused")
            elif not arm_enabled and self._prev_arm_enabled:
                self._vr_mapper.reset()
                logger.info("[HITL] VR deadman released, policy resumes")
            self._prev_arm_enabled = arm_enabled
        intervening = bool(arm_enabled)
        with self._policy_lock:
            if self._policy_allowed == intervening:
                # A policy result computed before a takeover/release is stale.
                self._policy_generation += 1
                self._policy_result = None
                self._policy_result_seq = 0
                self._policy_consumed_seq = 0
                self._policy_target = None
            self._policy_allowed = not intervening
            self._policy_observation = obs
            policy_result = self._policy_result
            policy_result_seq = self._policy_result_seq

        if intervening:
            delta = np.zeros(3, dtype=np.float64)
            apply_quat = None
            if arm_enabled and self._vr_mapper is not None and vr_input is not None:
                current_xyz, current_quat_xyzw = self.env.ee_pose()
                target_pos, target_quat_wxyz = self._vr_mapper.map(
                    vr_input,
                    current_xyz,
                    quat_xyzw_to_wxyz(current_quat_xyzw),
                )
                delta = np.asarray(target_pos, dtype=np.float64) - current_xyz
                apply_quat = quat_wxyz_to_xyzw(target_quat_wxyz)
            if vr_input is not None:
                if vr_input.gripper_close:
                    self._gripper_closed = True
                if vr_input.gripper_open:
                    self._gripper_closed = False
            return RobotCommand(
                ee_delta_xyz=delta,
                gripper_close=self._gripper_closed,
                ee_quat_xyzw=apply_quat,
                intervening=True,
                control_source="vr",
                vr_pose_seq=int(vr_input.pose_seq) if vr_input is not None else 0,
                vr_pose_monotonic_ns=(
                    int(vr_input.pose_monotonic_ns) if vr_input is not None else 0
                ),
            )

        current_xyz, _ = self.env.ee_pose()
        if policy_result is not None and policy_result_seq != self._policy_consumed_seq:
            delta, gripper_close = parse_ee_gripper_action(policy_result)
            self._policy_target = current_xyz + delta
            self._gripper_closed = gripper_close
            self._policy_consumed_seq = policy_result_seq
        delta = (
            np.zeros(3, dtype=np.float64)
            if self._policy_target is None
            else self._policy_target - current_xyz
        )
        return RobotCommand(
            ee_delta_xyz=delta,
            gripper_close=self._gripper_closed,
            intervening=False,
        )

    def on_transition(self, record: StepRecord) -> None:
        intervening = bool(record.command.intervening)
        if intervening and not self._was_intervening:
            parent_branch = self._branch_id
            self._branch_id += 1
            self._branch_start_step = record.step_index
            self._flush_transition()
            self._record(
                {
                    "type": "branch_start",
                    "episode_index": record.episode_index,
                    "branch_id": self._branch_id,
                    "parent_branch_id": parent_branch,
                    "fork_step_index": record.step_index,
                    "timestamp_monotonic_ns": record.timestamp_monotonic_ns,
                    "control_source": record.command.control_source,
                    "fork_state": record.obs,
                    "fork_ee_pos_xyz": record.ee_pos_start_xyz,
                    "fork_ee_quat_xyzw": record.ee_quat_start_xyzw,
                }
            )
        elif not intervening and self._was_intervening:
            self._flush_transition()
            self._record(
                {
                    "type": "branch_release",
                    "episode_index": record.episode_index,
                    "branch_id": self._branch_id,
                    "step_index": record.step_index,
                    "timestamp_monotonic_ns": record.timestamp_monotonic_ns,
                    "resume_state": record.obs,
                    "resume_ee_pos_xyz": record.ee_pos_start_xyz,
                    "resume_ee_quat_xyzw": record.ee_quat_start_xyzw,
                }
            )
        self._was_intervening = intervening
        if intervening:
            self._intervention_steps += 1
        now = time.monotonic()
        if self._transition_pending is not None and (
            self._transition_pending["control_source"] != record.command.control_source
        ):
            self._flush_transition()
        if self._transition_pending is None:
            if self._next_transition_due == 0.0:
                self._next_transition_due = now + self._transition_period_s
            self._transition_pending = {
                "type": "transition",
                "episode_index": int(record.episode_index),
                "state": record.obs,
                "action": np.array(
                    [
                        *record.executed_xyz.tolist(),
                        1.0 if record.gripper_close else 0.0,
                    ],
                    dtype=np.float32,
                ),
                "reward": float(record.reward),
                "next_state": record.next_obs,
                "done": bool(record.done),
                "truncated": bool(record.truncated),
                "is_intervention": intervening,
                "clipped": bool(record.clipped),
                "control_source": record.command.control_source,
                "branch_id": self._branch_id,
                "branch_start_step": self._branch_start_step,
                "step_start": record.step_index,
                "step_end": record.step_index,
                "timestamp_start_ns": record.timestamp_monotonic_ns,
                "timestamp_end_ns": record.timestamp_monotonic_ns,
                "ee_pos_start_xyz": record.ee_pos_start_xyz,
                "ee_quat_start_xyzw": record.ee_quat_start_xyzw,
                "ee_quat_command_xyzw": (
                    record.command.ee_quat_xyzw
                    if record.command.ee_quat_xyzw is not None
                    else record.ee_quat_start_xyzw
                ),
                "vr_pose_seq_start": record.command.vr_pose_seq,
                "vr_pose_seq_end": record.command.vr_pose_seq,
                "vr_pose_monotonic_ns_end": record.command.vr_pose_monotonic_ns,
            }
        else:
            pending = self._transition_pending
            pending["action"][:3] += np.asarray(record.executed_xyz, dtype=np.float32)
            pending["action"][3] = 1.0 if record.gripper_close else 0.0
            pending["next_state"] = record.next_obs
            pending["reward"] += float(record.reward)
            pending["done"] = bool(record.done)
            pending["truncated"] = bool(record.truncated)
            pending["clipped"] = bool(pending["clipped"] or record.clipped)
            pending["step_end"] = record.step_index
            pending["timestamp_end_ns"] = record.timestamp_monotonic_ns
            pending["ee_quat_command_xyzw"] = (
                record.command.ee_quat_xyzw
                if record.command.ee_quat_xyzw is not None
                else record.ee_quat_start_xyzw
            )
            pending["vr_pose_seq_end"] = record.command.vr_pose_seq
            pending["vr_pose_monotonic_ns_end"] = record.command.vr_pose_monotonic_ns
        if record.done or now >= self._next_transition_due:
            self._flush_transition()
            while self._next_transition_due <= now:
                self._next_transition_due += self._transition_period_s

    def _flush_transition(self) -> None:
        if self._transition_pending is not None:
            self._record(self._transition_pending)
            if self._transitions is not None:
                keys = (
                    "type",
                    "episode_index",
                    "state",
                    "action",
                    "reward",
                    "next_state",
                    "done",
                    "truncated",
                    "is_intervention",
                    "clipped",
                )
                self._transitions.send(
                    {key: self._transition_pending[key] for key in keys}
                )
            self._transition_pending = None

    def _record(self, event: dict) -> None:
        if self._recorder is not None:
            self._recorder.record(event)

    def on_episode_end(self, stats: EpisodeStats) -> None:
        self._flush_transition()
        rate = self._intervention_steps / max(stats.steps, 1)
        clip_rate = stats.clip_steps / max(stats.steps, 1)
        stats_event = {
            "type": "episode_stats",
            "episode_index": int(stats.episode_index),
            "episodic_reward": float(stats.episodic_reward),
            "success": bool(stats.success),
            "intervention_rate": float(rate),
            "clip_rate": float(clip_rate),
            "steps": int(stats.steps),
            "branch_count": self._branch_id,
            "interrupted": bool(stats.extras.get("interrupted", False)),
        }
        self._record(stats_event)
        if self._transitions is not None:
            self._transitions.send(
                {
                    key: stats_event[key]
                    for key in (
                        "type",
                        "episode_index",
                        "episodic_reward",
                        "success",
                        "intervention_rate",
                        "clip_rate",
                        "steps",
                    )
                }
            )
        logger.info(
            "[HITL] episode %s done success=%s steps=%s intervene=%.2f clip=%.2f",
            stats.episode_index,
            stats.success,
            stats.steps,
            rate,
            clip_rate,
        )
