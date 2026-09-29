"""Fixed-rate robot runner with policy and human control hooks."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np

from control.ee_safety import EESafety
from control.hitl.env import RobotEnv
from control.hitl.policy import PolicyClient
from control.hitl.types import (
    EpisodeStats,
    LoopEvents,
    RobotCommand,
    StepContext,
    StepRecord,
)


class RobotLoop(ABC):
    """Owns env, optional policy, safety clip, and the episode/step cadence.

    Subclasses implement ``select_action``. Reward, human labels, and replay
    traffic stay in hooks so a pure eval loop can ignore them.
    """

    def __init__(
        self,
        *,
        env: RobotEnv,
        safety: EESafety,
        fps: float,
        episode_time_s: float,
        policy: PolicyClient | None = None,
        max_episodes: int = 0,
        max_steps: int = 0,
    ) -> None:
        self.env = env
        self.safety = safety
        self.policy = policy
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self.episode_time_s = float(episode_time_s)
        self.max_episodes = int(max_episodes)
        self.max_steps = int(max_steps)
        self._stop = False

    def run(self) -> None:
        try:
            self.setup()
            self._run_episodes()
        finally:
            try:
                self.teardown()
            except Exception:
                logging.exception("[HITL] teardown failed")
            if self.policy is not None:
                self.policy.close()
            self.env.close()

    def setup(self) -> None:
        return

    def teardown(self) -> None:
        return

    def poll_events(self) -> LoopEvents:
        return LoopEvents()

    def compute_reward(self, *, success: bool, done: bool) -> float:
        return 1.0 if success else 0.0

    @abstractmethod
    def select_action(self, obs: dict, context: StepContext) -> RobotCommand:
        raise NotImplementedError

    def on_episode_start(self, episode_index: int) -> None:
        return

    def on_transition(self, record: StepRecord) -> None:
        return

    def on_episode_end(self, stats: EpisodeStats) -> None:
        return

    def request_stop(self) -> None:
        self._stop = True

    def _run_episodes(self) -> None:
        episode_index = 0
        while not self._stop:
            self.env.reset()
            episode_index += 1
            self.on_episode_start(episode_index)
            obs = self.env.observation()
            episode_reward = 0.0
            total_steps = 0
            clip_steps = 0
            started = time.monotonic()
            done = False
            success = False
            period_s = 1.0 / self.fps
            next_tick = time.monotonic()
            logging.info("[HITL] episode %s start", episode_index)

            while not done and not self._stop:
                remaining = next_tick - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                if self._stop:
                    break
                next_tick += period_s
                if next_tick < time.monotonic():
                    next_tick = time.monotonic() + period_s
                if self.max_steps > 0 and total_steps >= self.max_steps:
                    done = True
                    break
                events = self.poll_events()
                if events.success:
                    success = True
                    done = True
                if events.failure:
                    success = False
                    done = True
                elapsed_s = time.monotonic() - started
                timed_out = elapsed_s >= self.episode_time_s
                if timed_out:
                    done = True

                command = self.select_action(
                    obs,
                    StepContext(
                        episode_index=episode_index,
                        step_index=total_steps,
                        elapsed_s=elapsed_s,
                        episode_time_s=self.episode_time_s,
                    ),
                )
                current_xyz, current_quat = self.env.ee_pose()
                target, executed, clipped = self.safety.apply_delta(
                    current_xyz, command.ee_delta_xyz
                )
                self.env.apply(
                    target,
                    command.gripper_close,
                    target_quat_xyzw=command.ee_quat_xyzw,
                )
                next_obs = self.env.observation()
                reward = self.compute_reward(success=success, done=done)
                episode_reward += reward
                total_steps += 1
                if clipped:
                    clip_steps += 1
                self.on_transition(
                    StepRecord(
                        episode_index=episode_index,
                        step_index=total_steps - 1,
                        timestamp_monotonic_ns=time.monotonic_ns(),
                        ee_pos_start_xyz=np.asarray(current_xyz, dtype=np.float64),
                        ee_quat_start_xyzw=np.asarray(current_quat, dtype=np.float64),
                        obs=obs,
                        command=command,
                        executed_xyz=np.asarray(executed, dtype=np.float64),
                        gripper_close=bool(command.gripper_close),
                        reward=float(reward),
                        next_obs=next_obs,
                        done=bool(done),
                        truncated=bool((not success) and timed_out),
                        clipped=bool(clipped),
                        success=bool(success),
                    )
                )
                obs = next_obs

            if self.max_episodes > 0 and episode_index >= self.max_episodes:
                self._stop = True
            stats = EpisodeStats(
                episode_index=episode_index,
                episodic_reward=float(episode_reward),
                success=bool(success),
                steps=int(total_steps),
                clip_steps=int(clip_steps),
                extras={"interrupted": bool(self._stop)},
            )
            self.on_episode_end(stats)
            logging.info(
                "[HITL] episode %s done success=%s steps=%s clip=%.2f",
                episode_index,
                success,
                total_steps,
                clip_steps / max(total_steps, 1),
            )
