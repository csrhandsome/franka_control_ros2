"""构造最小可用的 LeRobot 数据集，供 data_analysis 的测试使用。

**不要拿真实数据集做这些测试** —— 仓库的 `data/` 在 .gitignore 里，删除类工具
改坏了就找不回来了。所有测试都在 tmpdir 里的合成数据集上跑。

合成出来的目录结构（4 个 episode，长度 5/7/3/6，共 21 帧）：

    <root>/meta/info.json
    <root>/meta/episodes.jsonl
    <root>/meta/episodes_stats.jsonl
    <root>/meta/tasks.jsonl
    <root>/data/chunk-000/episode_00000{0..3}.parquet
    <root>/audio/episode_00000{0..3}.wav
    <root>/audio/episode_00000{0..3}.audio.json
    <root>/audio/episode_00000{0..3}.sync.json
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_LENGTHS = (5, 7, 3, 6)
SAMPLE_RATE = 16000
TASKS = ["pick up the block", "put the block in the box"]

# `diverge=True` 时 detour（奇数号）episode 从这一帧起往侧面偏 0.25 m，
# 用来给 route_labels 的分叉检测造出一个真实可测的分叉点（默认 fps=20 → 0.5 s）。
DIVERGE_FRAME = 10
DIVERGE_OFFSET_M = 0.25


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _write_silence_wav(
    path: Path, duration_sec: float, sample_rate: int = SAMPLE_RATE
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def _write_voiced_wav(
    path: Path, duration_sec: float, sample_rate: int = SAMPLE_RATE
) -> None:
    """合成一段「像人声」的信号：谐波堆 + 4 Hz 音节调制 + 少量噪声。

    只为让 Silero-VAD 在合成夹具上真能检出片段 —— 纯静音和纯噪声都检出 0 段，
    那样很多测试会变成空跑。不是要仿真人声，别拿它当音频素材用。

    固定随机种子，保证同一份夹具每次重建都逐字节一致。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(duration_sec * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    voiced = np.zeros(n, dtype=np.float64)
    for k in range(1, 25):
        freq = 120 * k
        if freq > 4000:
            break
        voiced += np.sin(2 * np.pi * freq * t + k) / k
    voiced /= max(1e-9, float(np.abs(voiced).max()))
    syllables = np.clip(np.sin(2 * np.pi * 4.0 * t), 0.0, 1.0) ** 0.7
    gate = ((t % 1.4) < 1.1).astype(np.float64)
    noise = np.random.default_rng(20260929).uniform(-0.05, 0.05, size=n)
    signal = np.clip(voiced * syllables * gate * 0.4 + noise, -1.0, 1.0)
    pcm = np.rint(signal * np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _stats_for(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n if n else 0.0
    var = sum((v - mean) ** 2 for v in values) / n if n else 0.0
    return {
        "min": [min(values)] if values else [0],
        "max": [max(values)] if values else [0],
        "mean": [mean],
        "std": [var**0.5],
        "count": [n],
    }


def build_lerobot_dataset(
    root: Path,
    *,
    lengths: tuple[int, ...] = DEFAULT_LENGTHS,
    with_vad: bool = False,
    fps: float = 20.0,
    diverge: bool = False,
    speech_audio: bool = False,
) -> Path:
    """在 `root` 下生成一个合成 LeRobot 数据集，返回 root。

    `with_vad=True` 时额外在每个 sync.json 里写入 vad_segments 与
    vad_metadata，供 window / route_labels / asr 相关测试使用。

    `diverge=True` 时 detour（奇数号）episode 的 ee_position 从
    `DIVERGE_FRAME` 起侧向偏移 `DIVERGE_OFFSET_M`，让 route_labels 的
    配对分叉检测有东西可测（不开启时同一条轨迹，只会得到 no_divergence）。

    `speech_audio=True` 时把 WAV 换成 `_write_voiced_wav` 的合成人声，
    否则是纯静音（Silero-VAD 在静音上检出 0 段）。
    """
    root = Path(root)
    chunks_size = 1000
    total_frames = sum(lengths)

    for name in ("meta", "data/chunk-000", "audio", "videos/chunk-000"):
        (root / name).mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "franka",
        "fps": fps,
        "chunks_size": chunks_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_episodes": len(lengths),
        "total_frames": total_frames,
        "total_chunks": 1,
        "total_tasks": len(TASKS),
        "total_videos": 0,
        "splits": {"train": f"0:{len(lengths)}"},
        "features": {
            "action": {"dtype": "float32", "shape": [7], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    _write_json(root / "meta" / "info.json", info)

    _write_jsonl(
        root / "meta" / "tasks.jsonl",
        [{"task_index": i, "task": task} for i, task in enumerate(TASKS)],
    )

    episodes_rows: list[dict] = []
    stats_rows: list[dict] = []
    global_start = 0

    for episode_index, length in enumerate(lengths):
        episode_rows: list[dict] = []
        frame_indices = list(range(length))
        global_indices = list(range(global_start, global_start + length))
        task_index = episode_index % len(TASKS)

        table = pa.table(
            {
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "index": pa.array(global_indices, type=pa.int64()),
                "task_index": pa.array([task_index] * length, type=pa.int64()),
                "timestamp": pa.array(
                    [i / fps for i in frame_indices], type=pa.float32()
                ),
                "action": pa.array(
                    [[0.01 * (i + 1)] * 7 for i in frame_indices],
                    type=pa.list_(pa.float32()),
                ),
            }
        )
        parquet_path = (
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        pq.write_table(table, parquet_path)

        episodes_rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "tasks": [TASKS[task_index]],
            }
        )
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "episode_index": _stats_for([float(episode_index)] * length),
                    "frame_index": _stats_for([float(i) for i in frame_indices]),
                    "index": _stats_for([float(i) for i in global_indices]),
                    "task_index": _stats_for([float(task_index)] * length),
                    "timestamp": _stats_for([i / fps for i in frame_indices]),
                },
            }
        )

        stem = f"episode_{episode_index:06d}"
        duration_sec = round(length / fps, 3)
        wav_path = root / "audio" / f"{stem}.wav"
        if speech_audio:
            _write_voiced_wav(wav_path, duration_sec)
        else:
            _write_silence_wav(wav_path, duration_sec)
        _write_json(
            root / "audio" / f"{stem}.audio.json",
            {
                "audio_path": str((root / "audio" / f"{stem}.wav").resolve()),
                "sample_rate": SAMPLE_RATE,
                "duration_sec": duration_sec,
            },
        )

        sync_payload: dict = {
            "episode_index": episode_index,
            "audio_path": f"audio/{stem}.wav",
            "audio_metadata_path": f"audio/{stem}.audio.json",
            "audio_duration_sec": duration_sec,
            "control_frequency": fps,
            "label": "straight" if episode_index % 2 == 0 else "detour",
            "divergence_time": None,
            "frame_records": [
                {
                    "frame_index": i,
                    "ee_position": [
                        0.3 + 0.01 * i,
                        DIVERGE_OFFSET_M
                        if (diverge and episode_index % 2 == 1 and i >= DIVERGE_FRAME)
                        else 0.0,
                        0.2,
                    ],
                }
                for i in range(length)
            ],
        }
        if with_vad:
            segments = [
                {
                    "seg_id": 0,
                    "start_sec": 0.05,
                    "end_sec": max(0.06, duration_sec - 0.05),
                    "start_sample": int(0.05 * SAMPLE_RATE),
                    "end_sample": int(max(0.06, duration_sec - 0.05) * SAMPLE_RATE),
                    "audio_wav_path": f"audio/vad_segments/{stem}/seg_000.wav",
                    "asr_transcript": None,
                    "verified": False,
                }
            ]
            sync_payload["vad_segments"] = segments
            sync_payload["vad_metadata"] = {
                "processed": True,
                "stage": "offline_preprocess",
                "audio_sample_rate": SAMPLE_RATE,
                "audio_duration_sec": duration_sec,
                "vad_sample_rate": SAMPLE_RATE,
                "num_segments": len(segments),
                "segments_dir": f"audio/vad_segments/{stem}",
                "parameters": {"threshold": 0.25},
            }
        _write_json(root / "audio" / f"{stem}.sync.json", sync_payload)

        global_start += length

    _write_jsonl(root / "meta" / "episodes.jsonl", episodes_rows)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", stats_rows)
    return root


def snapshot_tree(root: Path) -> dict[str, bytes | None]:
    """递归快照一棵目录树：相对路径 → 文件字节（目录为 None）。

    用于断言重构前后两次操作产生**逐字节相同**的结果。
    """
    root = Path(root)
    snapshot: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        snapshot[rel] = None if path.is_dir() else path.read_bytes()
    return snapshot


def diff_snapshots(
    left: dict[str, bytes | None], right: dict[str, bytes | None]
) -> list[str]:
    """比较两份快照，返回人类可读的差异列表（空列表表示完全一致）。"""
    problems: list[str] = []
    for key in sorted(set(left) | set(right)):
        if key not in left:
            problems.append(f"only in right: {key}")
        elif key not in right:
            problems.append(f"only in left: {key}")
        elif left[key] != right[key]:
            problems.append(f"content differs: {key}")
    return problems
