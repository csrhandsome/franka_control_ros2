"""Episode samples and persistence for VR collection."""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from control.recording_writer import FreshCameraPair
from control.robot_state import EEPose, JointPose
from control.util.robot import current_ee_pose, current_joint_position


def _float32_list(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float32).tolist()


@dataclass
class ActionSample:
    sample_index: int
    host_sample_monotonic_ns: int
    joint_position: np.ndarray
    gripper_position: float
    ee_position: np.ndarray
    ee_orientation_xyzw: np.ndarray
    ee_pose: np.ndarray
    target_ee_position: np.ndarray
    target_ee_orientation_xyzw: np.ndarray
    vr_pose_seq: int
    vr_pose_monotonic_ns: int
    vr_position: tuple[float, float, float]
    vr_orientation_xyzw: tuple[float, float, float, float]
    vr_arm_enabled: bool

    def to_record(
        self, *, action: tuple[np.ndarray, np.ndarray, float] | None = None
    ) -> dict:
        record = {
            "sample_index": self.sample_index,
            "host_sample_monotonic_ns": self.host_sample_monotonic_ns,
            "joint_position": _float32_list(self.joint_position),
            "gripper_position": float(self.gripper_position),
            "ee_position": _float32_list(self.ee_position),
            "ee_orientation_xyzw": _float32_list(self.ee_orientation_xyzw),
            "ee_pose": _float32_list(self.ee_pose),
            "target_ee_position": _float32_list(self.target_ee_position),
            "target_ee_orientation_xyzw": _float32_list(
                self.target_ee_orientation_xyzw
            ),
            "action_gripper_position": float(self.gripper_position),
            "vr_pose_seq": int(self.vr_pose_seq),
            "vr_pose_monotonic_ns": int(self.vr_pose_monotonic_ns),
            "vr_position": [float(value) for value in self.vr_position],
            "vr_orientation_xyzw": [float(value) for value in self.vr_orientation_xyzw],
            "vr_arm_enabled": bool(self.vr_arm_enabled),
        }
        if action is not None:
            qpos, ee_pose, gripper = action
            record["action_joint_position"] = JointPose(qpos).vector.tolist()
            record["action_ee_pose"] = EEPose.from_vector(ee_pose).vector.tolist()
            record["action_gripper_position"] = float(gripper)
        return record


@dataclass
class FrameSample:
    frame_index: int
    host_frame_monotonic_ns: int
    external_camera_timestamp: float
    wrist_camera_timestamp: float
    external_host_capture_monotonic_ns: int
    wrist_host_capture_monotonic_ns: int
    joint_position: np.ndarray
    gripper_position: float
    ee_position: np.ndarray
    ee_orientation_xyzw: np.ndarray
    ee_pose: np.ndarray
    external_img: np.ndarray
    wrist_img: np.ndarray
    gripper_left_img: np.ndarray | None = None
    gripper_right_img: np.ndarray | None = None

    def to_record(
        self, *, action: tuple[np.ndarray, np.ndarray, float] | None = None
    ) -> dict:
        record = {
            "frame_index": self.frame_index,
            "host_frame_monotonic_ns": self.host_frame_monotonic_ns,
            "external_camera_timestamp": float(self.external_camera_timestamp),
            "wrist_camera_timestamp": float(self.wrist_camera_timestamp),
            "external_host_capture_monotonic_ns": int(
                self.external_host_capture_monotonic_ns
            ),
            "wrist_host_capture_monotonic_ns": int(
                self.wrist_host_capture_monotonic_ns
            ),
            "joint_position": _float32_list(self.joint_position),
            "gripper_position": float(self.gripper_position),
            "ee_position": _float32_list(self.ee_position),
            "ee_orientation_xyzw": _float32_list(self.ee_orientation_xyzw),
            "ee_pose": _float32_list(self.ee_pose),
        }
        if action is not None:
            qpos, ee_pose, gripper = action
            record["action_joint_position"] = JointPose(qpos).vector.tolist()
            record["action_ee_pose"] = EEPose.from_vector(ee_pose).vector.tolist()
            record["action_gripper_position"] = float(gripper)
        return record

    def to_dataset_frame(
        self,
        action: tuple[np.ndarray, np.ndarray, float],
        *,
        task: str,
        action_space: str,
        force: bool,
    ) -> dict:
        qpos, ee_pose, gripper = action
        joint = JointPose(qpos)
        ee = EEPose.from_vector(ee_pose)
        actions = ee.action(gripper) if action_space == "ee" else joint.action(gripper)
        frame = {
            "exterior_image_1_left": self.external_img,
            "exterior_image_2_left": np.zeros_like(self.external_img),
            "wrist_image_left": self.wrist_img,
            "joint_position": np.asarray(self.joint_position, dtype=np.float32),
            "ee_pose": np.asarray(self.ee_pose, dtype=np.float32),
            "gripper_position": np.asarray([self.gripper_position], dtype=np.float32),
            "actions": actions,
            "task": task,
        }
        if force:
            frame["gripper_image_left"] = self.gripper_left_img
            frame["gripper_image_right"] = self.gripper_right_img
            frame["external_camera_timestamp_ms"] = np.asarray(
                [self.external_camera_timestamp * 1000.0], dtype=np.float32
            )
            frame["wrist_camera_timestamp_ms"] = np.asarray(
                [self.wrist_camera_timestamp * 1000.0], dtype=np.float32
            )
            frame["external_camera_frame_age_s"] = np.asarray(
                [
                    max(
                        0,
                        self.host_frame_monotonic_ns
                        - self.external_host_capture_monotonic_ns,
                    )
                    * 1e-9
                ],
                dtype=np.float32,
            )
            frame["wrist_camera_frame_age_s"] = np.asarray(
                [
                    max(
                        0,
                        self.host_frame_monotonic_ns
                        - self.wrist_host_capture_monotonic_ns,
                    )
                    * 1e-9
                ],
                dtype=np.float32,
            )
        return frame


class CollectionEpisodeRecorder:
    """Own the one-tick action offset, camera frames, and episode files."""

    def __init__(
        self,
        *,
        dataset: Any,
        frame_writer: Any,
        root: Path | None,
        action_space: str,
        force: bool,
        task: str,
        label: str,
        control_frequency: float,
        camera_fps: float,
        camera_serials: dict[str, str],
    ) -> None:
        self.dataset = dataset
        self.frame_writer = frame_writer
        self.root = root
        self.action_space = action_space
        self.force = force
        self.task = task
        self.label = label
        self.control_frequency = control_frequency
        self.camera_fps = camera_fps
        self.camera_serials = camera_serials
        self.camera_pair_gate = FreshCameraPair()
        self.reset()

    def reset(self) -> None:
        self.recording_started_at: float | None = None
        self.episode_start_monotonic_ns: int | None = None
        self.episode_index: int | None = None
        self.frame_records: list[dict] = []
        self.pending_frame: FrameSample | None = None
        self.pending_action: ActionSample | None = None
        self.action_trace_file: TextIO | None = None
        self.action_trace_tmp_path: Path | None = None
        self.action_count = 0
        self.first_action_sample_ns = 0
        self.last_action_sample_ns = 0
        self.last_recorded_vr_seq = 0
        self.unique_vr_samples = 0
        self.frame_count = 0
        self.camera_pair_gate.reset()

    @property
    def is_recording(self) -> bool:
        return self.episode_index is not None

    @property
    def has_samples(self) -> bool:
        return self.frame_count > 0 or self.pending_frame is not None

    def start(self) -> int:
        if self.dataset is None or self.root is None:
            raise RuntimeError("Logging state is not initialized")
        if self.is_recording:
            raise RuntimeError("Episode is already recording")
        self.episode_index = int(self.dataset.next_episode_index)
        self.episode_start_monotonic_ns = time.monotonic_ns()
        self.recording_started_at = time.time()
        self.action_trace_tmp_path = (
            self.root / f"episode_{self.episode_index:06d}.actions.jsonl.tmp"
        )
        self.action_trace_file = self.action_trace_tmp_path.open("w", encoding="utf-8")
        return self.episode_index

    def queue_action(self, sample: ActionSample) -> None:
        self.pending_action = sample

    def complete_action(
        self, qpos: np.ndarray, ee_pose: np.ndarray, gripper: float
    ) -> None:
        sample = self.pending_action
        if sample is None or self.action_trace_file is None:
            return
        record = sample.to_record(action=(qpos, ee_pose, gripper))
        self.action_trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        sample_ns = sample.host_sample_monotonic_ns
        if self.first_action_sample_ns == 0:
            self.first_action_sample_ns = sample_ns
        self.last_action_sample_ns = sample_ns
        if sample.vr_pose_seq > 0 and sample.vr_pose_seq != self.last_recorded_vr_seq:
            self.unique_vr_samples += 1
            self.last_recorded_vr_seq = sample.vr_pose_seq
        self.action_count += 1
        self.pending_action = None

    def queue_frame(self, sample: FrameSample) -> None:
        self.pending_frame = sample

    def complete_frame(
        self, qpos: np.ndarray, ee_pose: np.ndarray, gripper: float
    ) -> None:
        sample = self.pending_frame
        if sample is None or self.dataset is None:
            return
        if self.frame_writer is None:
            raise RuntimeError("LeRobot frame writer is unavailable")
        action = (qpos, ee_pose, float(gripper))
        self.frame_writer.submit(
            sample.to_dataset_frame(
                action, task=self.task, action_space=self.action_space, force=self.force
            )
        )
        self.frame_records.append(sample.to_record(action=action))
        self.frame_count += 1
        self.pending_frame = None
        if self.frame_count % 50 == 0:
            print(f"[Recording] {self.frame_count} frames", end="\r")

    def _write_sync(self, success: bool | None) -> Path:
        if self.root is None or self.episode_index is None:
            raise RuntimeError(
                "Episode sync JSON cannot be written without logging state"
            )
        index = self.episode_index
        frame_records = self.frame_records
        frame_count = self.frame_count
        action_count = self.action_count
        action_span = self.last_action_sample_ns - self.first_action_sample_ns
        frame_span = (
            frame_records[-1]["host_frame_monotonic_ns"]
            - frame_records[0]["host_frame_monotonic_ns"]
            if frame_count > 1
            else 0
        )
        payload = {
            "episode_index": index,
            "task": self.task,
            "label": self.label,
            "success": success,
            "divergence_time": None,
            "control_frequency": float(self.control_frequency),
            "camera_fps": float(self.camera_fps),
            "action_space": self.action_space,
            "action_target_offset_frames": 1,
            "action_frequency": float(self.control_frequency),
            "action_records": action_count,
            "action_effective_hz": (action_count - 1) * 1e9 / action_span
            if action_count > 1 and action_span > 0
            else 0.0,
            "vr_unique_pose_samples": self.unique_vr_samples,
            "vr_reused_pose_samples": action_count - self.unique_vr_samples,
            "action_trace": f"episode_{index:06d}.actions.jsonl",
            "episode_start_monotonic_ns": self.episode_start_monotonic_ns,
            "video_frames": frame_count,
            "camera_pair_effective_hz": (frame_count - 1) * 1e9 / frame_span
            if frame_count > 1 and frame_span > 0
            else 0.0,
            "frame_records": frame_records,
            "camera_serials": self.camera_serials,
        }
        path = self.root / f"episode_{index:06d}.sync.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @staticmethod
    def _cleanup_files(*paths: Path | None) -> None:
        for path in paths:
            if path is not None:
                with contextlib.suppress(Exception):
                    path.unlink(missing_ok=True)

    def save(
        self,
        *,
        save_episode: bool,
        success: bool | None,
        arm: Any,
        final_gripper: float,
    ) -> Path | None:
        try:
            if save_episode:
                final_qpos = current_joint_position(arm)
                position, quat = current_ee_pose(arm)
                final_ee_pose = EEPose.from_position_quat(position, quat).vector
                self.complete_frame(final_qpos, final_ee_pose, final_gripper)
                self.complete_action(final_qpos, final_ee_pose, final_gripper)
            if self.action_trace_file is not None:
                self.action_trace_file.close()
            if self.frame_writer is not None:
                self.frame_writer.drain()
        except Exception:
            if self.action_trace_file is not None and not self.action_trace_file.closed:
                self.action_trace_file.close()
            self._cleanup_files(self.action_trace_tmp_path)
            if self.dataset is not None:
                self.dataset.discard_episode()
            self.reset()
            raise
        if not (
            save_episode
            and self.frame_count > 0
            and self.action_count > 0
            and self.dataset is not None
            and self.episode_index is not None
        ):
            self._cleanup_files(self.action_trace_tmp_path)
            if self.dataset is not None:
                with contextlib.suppress(Exception):
                    self.dataset.discard_episode()
            self.reset()
            return None
        sync_path: Path | None = None
        action_path: Path | None = None
        try:
            if self.root is None or self.action_trace_tmp_path is None:
                raise RuntimeError("Action trace was not initialized")
            index = self.episode_index
            action_path = self.root / f"episode_{index:06d}.actions.jsonl"
            self.action_trace_tmp_path.replace(action_path)
            sync_path = self._write_sync(success)
            self.dataset.save_episode()
            print(
                f"[Recording] Saved episode {index:06d}: {self.action_count} action samples, {self.frame_count} camera frames"
            )
            print(f"[Recording] Saved sync metadata: {sync_path}")
            return sync_path
        except Exception as exc:
            print(f"[Error] Failed to save episode: {exc}")
            self._cleanup_files(sync_path, self.action_trace_tmp_path, action_path)
            with contextlib.suppress(Exception):
                self.dataset.discard_episode()
            raise
        finally:
            self.reset()

    def read_sync(self, episode_index: int) -> dict:
        if self.root is None:
            raise RuntimeError("Dataset root is unavailable")
        path = self.root / f"episode_{episode_index:06d}.sync.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def read_actions(self, episode_index: int) -> list[dict]:
        if self.root is None:
            raise RuntimeError("Dataset root is unavailable")
        path = self.root / f"episode_{episode_index:06d}.actions.jsonl"
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]
