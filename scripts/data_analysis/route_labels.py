"""路线标注：straight/detour 的配对分叉检测与 6_28 标签回填。

两个子命令写的是 `audio/episode_*.sync.json` 里**互补**的字段，互不覆盖，
也互相不 import（合并前就是两个独立脚本，这里只是收进同一个文件）：

- ``route_labels annotate``   —— 读采集端写好的 ``label``，算配对轨迹的分叉
  时刻并写 ``divergence_time``。「哪两集是一对」是脚本期概念，不落盘；
  默认相邻配对 0:1、2:3…，可用 ``--pair 3:4`` 指定。
- ``route_labels apply-6-28`` —— 给 6_28 复制集批量写 ``label``（前
  ``--straight-count`` 集 straight、其余 detour）和 ``instruction_audio_window``
  （有 VAD 用 VAD；straight 且无 VAD 借配对 episode 的 VAD 窗口；再退到
  全集中位数；都没有则留 pending）。

`label` 字段同时容忍 ``route_label`` 这个旧名字，见 `_label`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.data_analysis import lerobot_meta as meta
from scripts.data_analysis.audio.window import (
    DEFAULT_POST_MARGIN_SEC,
    DEFAULT_PRE_MARGIN_SEC,
    build_imputed_instruction_audio_window,
    build_vad_instruction_audio_window,
    instruction_audio_window_end_time,
)

# 路线词表。采集端写 `label`，本模块 annotate 写 `divergence_time`，两者不冲突。
ROUTE_LABELS = {"straight", "detour"}


# --------------------------------------------------------------------------
# 两个子命令共用的路径约定
# --------------------------------------------------------------------------


def _sync_path(dataset_root: Path, episode_index: int) -> Path:
    return meta.sync_json_path(dataset_root, episode_index)


def _sync_index(path: Path) -> int:
    """`episode_000012.sync.json` → 12；解析不出来直接报错。

    与 `meta.parse_episode_filename` 的差别是刻意的：那个返回 None（供扫描时
    跳过畸形文件），这里要当字典键用，None 会一路静默传到比较 `index <
    straight_count` 时才炸，不如当场报错。
    """
    index = meta.parse_episode_filename(path.name)
    if index is None:
        raise ValueError(f"sync 文件名里没有 episode 序号: {path.name}")
    return index


def _require_audio_dir(dataset_root: Path) -> None:
    if not (dataset_root / "audio").is_dir():
        raise FileNotFoundError(f"Dataset has no audio directory: {dataset_root}")


# --------------------------------------------------------------------------
# annotate：配对分叉检测（← annotate_route_pairs.py）
# --------------------------------------------------------------------------


def _label(sync: dict[str, Any]) -> str | None:
    value = sync.get("label")
    if value is None:
        value = sync.get("route_label")
    if value is None:
        return None
    text = str(value).strip()
    return text if text in ROUTE_LABELS else None


def _instruction_end_time(sync: dict[str, Any]) -> float:
    end = instruction_audio_window_end_time(sync)
    if end is not None:
        return end
    segments = sync.get("vad_segments")
    if isinstance(segments, list):
        ends = [
            meta.safe_float(segment.get("end_sec"))
            for segment in segments
            if isinstance(segment, dict)
        ]
        ends = [value for value in ends if value is not None]
        if ends:
            return max(ends)
    return 0.0


def _load_trajectory(
    *,
    sync: dict[str, Any],
    episode_index: int,
    field: str,
    fallback_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    records = sync.get("frame_records")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError(f"episode {episode_index}: missing frame_records")

    audio_start_ns = sync.get("audio_start_monotonic_ns")
    try:
        audio_start_ns = int(audio_start_ns) if audio_start_ns is not None else None
    except (TypeError, ValueError, OverflowError):
        audio_start_ns = None
    fps = meta.safe_float(sync.get("control_frequency")) or fallback_fps

    times: list[float] = []
    positions: list[list[float]] = []
    for fallback_index, record in enumerate(records):
        if not isinstance(record, dict) or field not in record:
            continue
        position = np.asarray(record[field], dtype=np.float64).reshape(-1)
        if position.size == 0 or not np.all(np.isfinite(position)):
            continue
        frame_index = int(record.get("frame_index", fallback_index))
        host_ns = record.get("host_frame_monotonic_ns")
        try:
            host_ns = int(host_ns) if host_ns is not None else None
        except (TypeError, ValueError, OverflowError):
            host_ns = None
        if audio_start_ns is not None and host_ns is not None:
            t = float(host_ns - audio_start_ns) / 1e9
        else:
            t = float(frame_index) / float(fps)
        if np.isfinite(t):
            times.append(t)
            positions.append(position.tolist())

    if len(times) < 2:
        raise ValueError(f"episode {episode_index}: no usable {field} trajectory")
    order = np.argsort(np.asarray(times, dtype=np.float64))
    return np.asarray(times, dtype=np.float64)[order], np.asarray(
        positions, dtype=np.float64
    )[order]


def _interp(
    source_times: np.ndarray, source_positions: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    out = np.empty((target_times.shape[0], source_positions.shape[1]), dtype=np.float64)
    for dim in range(source_positions.shape[1]):
        out[:, dim] = np.interp(target_times, source_times, source_positions[:, dim])
    return out


def _find_divergence_time(
    *,
    a_sync: dict[str, Any],
    b_sync: dict[str, Any],
    a_index: int,
    b_index: int,
    field: str,
    fallback_fps: float,
    threshold: float,
    sustained_sec: float,
) -> tuple[float | None, float]:
    """返回 (分叉时刻, 最大间距)。分叉点必须**持续**超过阈值才认。"""
    a_times, a_positions = _load_trajectory(
        sync=a_sync, episode_index=a_index, field=field, fallback_fps=fallback_fps
    )
    b_times, b_positions = _load_trajectory(
        sync=b_sync, episode_index=b_index, field=field, fallback_fps=fallback_fps
    )
    if a_positions.shape[1] != b_positions.shape[1]:
        raise ValueError("paired trajectory dimensions do not match")

    start = max(float(a_times[0]), float(b_times[0]))
    end = min(float(a_times[-1]), float(b_times[-1]))
    times = a_times[(a_times >= start) & (a_times <= end)]
    if times.size < 2:
        raise ValueError("paired episodes have too little overlapping time")

    distances = np.linalg.norm(
        _interp(a_times, a_positions, times) - _interp(b_times, b_positions, times),
        axis=1,
    )
    # 只在两条指令都说完之后再找分叉 —— 之前的分叉是说话/起手差异，不算路线。
    search_start = max(_instruction_end_time(a_sync), _instruction_end_time(b_sync))
    valid_start = int(np.searchsorted(times, search_start, side="left"))
    if valid_start >= times.shape[0]:
        return None, float(np.max(distances))

    dt = float(np.median(np.diff(times))) if times.size >= 2 else 0.0
    index = valid_start
    while index < times.shape[0]:
        if distances[index] <= threshold:
            index += 1
            continue
        end_index = index
        while end_index + 1 < times.shape[0] and distances[end_index + 1] > threshold:
            end_index += 1
        if float(times[end_index] - times[index] + dt) >= sustained_sec:
            return float(times[index]), float(distances[index])
        index = end_index + 1
    return None, float(np.max(distances))


def _parse_pair(value: str) -> tuple[int, int]:
    left, sep, right = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError("pair must look like 0:1")
    try:
        return int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair indices must be integers") from exc


def _default_pairs(indices: list[int]) -> list[tuple[int, int]]:
    return [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]


def _run_annotate(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_path.resolve()
    _require_audio_dir(dataset_root)

    indices = meta.episode_indices(dataset_root)
    pairs = args.pair if args.pair is not None else _default_pairs(indices)
    results: list[dict[str, Any]] = []
    changed: set[int] = set()

    for a_index, b_index in pairs:
        a_path = _sync_path(dataset_root, a_index)
        b_path = _sync_path(dataset_root, b_index)
        if not a_path.is_file() or not b_path.is_file():
            results.append({"pair": f"{a_index}:{b_index}", "status": "missing_sync"})
            continue
        a_sync = meta.read_json(a_path)
        b_sync = meta.read_json(b_path)
        labels = {_label(a_sync), _label(b_sync)}
        if labels != ROUTE_LABELS:
            results.append(
                {
                    "pair": f"{a_index}:{b_index}",
                    "status": "bad_labels",
                    "labels": sorted(label for label in labels if label is not None),
                }
            )
            continue
        try:
            divergence_time, distance = _find_divergence_time(
                a_sync=a_sync,
                b_sync=b_sync,
                a_index=a_index,
                b_index=b_index,
                field=str(args.trajectory_field),
                fallback_fps=float(args.fallback_fps),
                threshold=float(args.divergence_threshold),
                sustained_sec=float(args.sustained_sec),
            )
        except Exception as exc:
            results.append(
                {"pair": f"{a_index}:{b_index}", "status": "error", "error": str(exc)}
            )
            continue
        if divergence_time is None:
            results.append(
                {
                    "pair": f"{a_index}:{b_index}",
                    "status": "no_divergence",
                    "max_distance": round(distance, 6),
                }
            )
            continue

        for index, sync in ((a_index, a_sync), (b_index, b_sync)):
            if args.overwrite or sync.get("divergence_time") in (None, ""):
                sync["divergence_time"] = round(divergence_time, 6)
                if not args.dry_run:
                    meta.write_json(
                        _sync_path(dataset_root, index),
                        sync,
                        indent=meta.SYNC_JSON_INDENT,
                    )
                changed.add(index)
        results.append(
            {
                "pair": f"{a_index}:{b_index}",
                "status": "annotated",
                "divergence_time": round(divergence_time, 6),
                "distance": round(distance, 6),
            }
        )

    print(f"Dataset: {dataset_root}")
    print(f"Pairs checked: {len(pairs)}")
    print(f"Episodes updated: {len(changed)}")
    print(f"Dry run: {bool(args.dry_run)}")
    for result in results:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# apply-6-28：批量写 label + instruction_audio_window（← apply_6_28_route_labels.py）
# --------------------------------------------------------------------------


def _vad_audio_duration(sync_data: dict[str, Any]) -> float | None:
    """只认 `vad_metadata.audio_duration_sec`。

    与 `audio/window.py` 的 `_audio_duration` **不同**（那个还会回退到顶层的
    `audio_duration_sec`）。这里刻意保留原 `apply_6_28_route_labels` 的窄语义：
    放宽会让没有 vad_metadata 的数据集凭空多出一个裁剪上界，改变写盘结果。
    """
    metadata = sync_data.get("vad_metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("audio_duration_sec")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _has_vad(sync_data: dict[str, Any]) -> bool:
    segments = sync_data.get("vad_segments")
    return isinstance(segments, list) and len(segments) > 0


def _first_vad_index(sync_by_index: dict[int, dict[str, Any]]) -> int | None:
    for index in sorted(sync_by_index):
        if _has_vad(sync_by_index[index]):
            return index
    return None


def _paired_window(
    *,
    reference_index: int,
    reference_sync: dict[str, Any],
    pre_margin_sec: float,
    post_margin_sec: float,
) -> dict[str, Any] | None:
    """straight 集没有自己的 VAD 时，借配对（detour）集的语音窗口。"""
    window = build_vad_instruction_audio_window(
        reference_sync.get("vad_segments"),
        audio_duration_sec=_vad_audio_duration(reference_sync),
        pre_margin_sec=pre_margin_sec,
        post_margin_sec=post_margin_sec,
    )
    if window is None:
        return None
    return {
        "start_sec": window["start_sec"],
        "end_sec": window["end_sec"],
        "source": "imputed_from_paired_vad",
        "pre_margin_sec": window["pre_margin_sec"],
        "post_margin_sec": window["post_margin_sec"],
        "audio_valid": False,
        "reference_episode_index": int(reference_index),
        "reference_source": "vad",
    }


def apply_labels_and_windows(
    dataset_path: str | Path,
    *,
    straight_count: int = 40,
    pair_source_start: int | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_path).resolve()
    _require_audio_dir(dataset_root)

    paths = meta.iter_sync_paths(dataset_root)
    sync_by_index = {_sync_index(path): meta.read_json(path) for path in paths}
    if pair_source_start is None:
        pair_source_start = _first_vad_index(sync_by_index)
    if pair_source_start is None:
        raise RuntimeError("No VAD segments found for paired window references")

    reference_windows: list[dict[str, Any]] = []
    for sync in sync_by_index.values():
        window = build_vad_instruction_audio_window(
            sync.get("vad_segments"),
            audio_duration_sec=_vad_audio_duration(sync),
            pre_margin_sec=pre_margin_sec,
            post_margin_sec=post_margin_sec,
        )
        if window is not None:
            reference_windows.append(window)

    updated = 0
    straight = 0
    detour = 0
    vad_windows = 0
    paired_windows = 0
    median_windows = 0
    pending_windows = 0
    examples: list[dict[str, Any]] = []

    for path in paths:
        index = _sync_index(path)
        sync = sync_by_index[index]
        label = "straight" if index < straight_count else "detour"
        if label == "straight":
            straight += 1
        else:
            detour += 1

        if _has_vad(sync):
            window = build_vad_instruction_audio_window(
                sync.get("vad_segments"),
                audio_duration_sec=_vad_audio_duration(sync),
                pre_margin_sec=pre_margin_sec,
                post_margin_sec=post_margin_sec,
            )
            vad_windows += 1
        elif index < straight_count:
            reference_index = int(pair_source_start) + index
            reference_sync = sync_by_index.get(reference_index)
            window = (
                _paired_window(
                    reference_index=reference_index,
                    reference_sync=reference_sync,
                    pre_margin_sec=pre_margin_sec,
                    post_margin_sec=post_margin_sec,
                )
                if reference_sync is not None
                else None
            )
            if window is not None:
                paired_windows += 1
        else:
            window = None

        if window is None:
            window = build_imputed_instruction_audio_window(
                reference_windows,
                audio_duration_sec=_vad_audio_duration(sync),
                pre_margin_sec=pre_margin_sec,
                post_margin_sec=post_margin_sec,
            )
            if window is not None:
                median_windows += 1
            else:
                pending_windows += 1

        before_label = sync.get("label")
        before_window = sync.get("instruction_audio_window")
        sync["label"] = label
        sync["instruction_audio_window"] = window
        changed = before_label != label or before_window != window
        if changed:
            updated += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "episode_index": index,
                        "label": label,
                        "window_source": window.get("source")
                        if isinstance(window, dict)
                        else None,
                        "start_sec": window.get("start_sec")
                        if isinstance(window, dict)
                        else None,
                        "end_sec": window.get("end_sec")
                        if isinstance(window, dict)
                        else None,
                    }
                )
            if not dry_run:
                meta.write_json(path, sync, indent=meta.SYNC_JSON_INDENT)

    return {
        "dataset_path": str(dataset_root),
        "episodes": len(paths),
        "updated": updated,
        "straight": straight,
        "detour": detour,
        "pair_source_start": int(pair_source_start),
        "vad_windows": vad_windows,
        "paired_windows": paired_windows,
        "median_windows": median_windows,
        "pending_windows": pending_windows,
        "reference_window_count": len(reference_windows),
        "dry_run": bool(dry_run),
        "examples": examples,
    }


def _run_apply_6_28(args: argparse.Namespace) -> int:
    summary = apply_labels_and_windows(
        args.dataset_path,
        straight_count=int(args.straight_count),
        pair_source_start=args.pair_source_start,
        pre_margin_sec=float(args.pre_margin_sec),
        post_margin_sec=float(args.post_margin_sec),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route labels: paired divergence detection and 6_28 label backfill."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser(
        "annotate",
        description="Detect paired straight/detour divergence and write divergence_time.",
        help="Write divergence_time for paired straight/detour episodes.",
    )
    annotate.add_argument("--dataset-path", type=Path, required=True)
    annotate.add_argument(
        "--pair",
        type=_parse_pair,
        action="append",
        default=None,
        help="Explicit episode pair such as 0:1. Repeat for multiple pairs. Defaults to adjacent pairs.",
    )
    annotate.add_argument("--trajectory-field", default="ee_position")
    annotate.add_argument("--fallback-fps", type=float, default=20.0)
    annotate.add_argument("--divergence-threshold", type=float, default=0.03)
    annotate.add_argument("--sustained-sec", type=float, default=0.25)
    annotate.add_argument("--overwrite", action="store_true")
    annotate.add_argument("--dry-run", action="store_true")
    annotate.set_defaults(handler=_run_annotate)

    apply_6_28 = subparsers.add_parser(
        "apply-6-28",
        description="Set 6_28 straight/detour labels and instruction windows.",
        help="Backfill 6_28 straight/detour labels and instruction windows.",
    )
    apply_6_28.add_argument("--dataset-path", type=Path, required=True)
    apply_6_28.add_argument("--straight-count", type=int, default=40)
    apply_6_28.add_argument(
        "--pair-source-start",
        type=int,
        default=None,
        help="First VAD episode used to provide windows for straight episodes. Defaults to first episode with VAD.",
    )
    apply_6_28.add_argument(
        "--pre-margin-sec", type=float, default=DEFAULT_PRE_MARGIN_SEC
    )
    apply_6_28.add_argument(
        "--post-margin-sec", type=float, default=DEFAULT_POST_MARGIN_SEC
    )
    apply_6_28.add_argument("--dry-run", action="store_true")
    apply_6_28.set_defaults(handler=_run_apply_6_28)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
