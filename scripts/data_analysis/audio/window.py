"""从 VAD 元数据推导 instruction audio window。

原始 VAD 结果留在 ``vad_segments`` 里；这里写入派生的
``instruction_audio_window``，供音频裁剪和动作对齐使用。没有 VAD 的 episode
可以用「有 VAD 的那些 episode 的中位数」对齐，同时标记为 imputed 而不是真实语音。

本模块是纯标准库 + `lerobot_meta`，没有 numpy / torch 依赖，可以放心被
`route_labels` 和 `audio/vad` 共同引用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from scripts.data_analysis import lerobot_meta as meta
from scripts.data_analysis.lerobot_meta import safe_float

DEFAULT_PRE_MARGIN_SEC = 0.2
DEFAULT_POST_MARGIN_SEC = 0.4


def _safe_float(value: Any) -> float | None:
    return safe_float(value)


def _round_time(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _clip_time(
    value: float, *, lower: float = 0.0, upper: float | None = None
) -> float:
    clipped = max(float(lower), float(value))
    if upper is not None:
        clipped = min(clipped, float(upper))
    return clipped


def make_pending_instruction_audio_window(
    *,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
) -> dict[str, Any]:
    return {
        "start_sec": None,
        "end_sec": None,
        "source": "pending_vad",
        "pre_margin_sec": round(float(pre_margin_sec), 6),
        "post_margin_sec": round(float(post_margin_sec), 6),
        "audio_valid": False,
    }


def build_vad_instruction_audio_window(
    vad_segments: Any,
    *,
    audio_duration_sec: float | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
) -> dict[str, Any] | None:
    if not isinstance(vad_segments, list):
        return None

    starts: list[float] = []
    ends: list[float] = []
    for segment in vad_segments:
        if not isinstance(segment, dict):
            continue
        start = _safe_float(segment.get("start_sec"))
        end = _safe_float(segment.get("end_sec"))
        if start is None or end is None or end <= start:
            continue
        starts.append(start)
        ends.append(end)

    if not starts or not ends:
        return None

    raw_start = min(starts)
    raw_end = max(ends)
    duration = _safe_float(audio_duration_sec)
    start_sec = _clip_time(raw_start - float(pre_margin_sec), upper=duration)
    end_sec = _clip_time(raw_end + float(post_margin_sec), upper=duration)
    end_sec = max(end_sec, start_sec)

    return {
        "start_sec": _round_time(start_sec),
        "end_sec": _round_time(end_sec),
        "source": "vad",
        "pre_margin_sec": round(float(pre_margin_sec), 6),
        "post_margin_sec": round(float(post_margin_sec), 6),
        "audio_valid": True,
        "vad_start_sec": _round_time(raw_start),
        "vad_end_sec": _round_time(raw_end),
        "num_vad_segments": len(starts),
    }


def build_imputed_instruction_audio_window(
    reference_windows: list[dict[str, Any]],
    *,
    audio_duration_sec: float | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
) -> dict[str, Any] | None:
    starts: list[float] = []
    ends: list[float] = []
    for window in reference_windows:
        if not isinstance(window, dict):
            continue
        start = _safe_float(window.get("start_sec"))
        end = _safe_float(window.get("end_sec"))
        if start is None or end is None or end < start:
            continue
        starts.append(start)
        ends.append(end)

    if not starts or not ends:
        return None

    duration = _safe_float(audio_duration_sec)
    start_sec = _clip_time(float(median(starts)), upper=duration)
    end_sec = _clip_time(float(median(ends)), upper=duration)
    end_sec = max(end_sec, start_sec)

    return {
        "start_sec": _round_time(start_sec),
        "end_sec": _round_time(end_sec),
        "source": "imputed_dataset_median",
        "pre_margin_sec": round(float(pre_margin_sec), 6),
        "post_margin_sec": round(float(post_margin_sec), 6),
        "audio_valid": False,
        "reference_window_count": len(starts),
    }


def instruction_audio_window_end_time(sync_data: dict[str, Any]) -> float | None:
    for key in (
        "instruction_audio_window",
        "instruction_window",
        "pseudo_instruction_window",
    ):
        window = sync_data.get(key)
        if isinstance(window, dict):
            end = _safe_float(window.get("end_sec"))
            if end is not None:
                return end
    return _safe_float(sync_data.get("instruction_end_time"))


def _audio_duration(sync_data: dict[str, Any]) -> float | None:
    metadata = sync_data.get("vad_metadata")
    if isinstance(metadata, dict):
        duration = _safe_float(metadata.get("audio_duration_sec"))
        if duration is not None:
            return duration
    return _safe_float(sync_data.get("audio_duration_sec"))


def refresh_dataset_instruction_audio_windows(
    dataset_path: str | Path,
    *,
    episode_indices: list[int] | None = None,
    pre_margin_sec: float = DEFAULT_PRE_MARGIN_SEC,
    post_margin_sec: float = DEFAULT_POST_MARGIN_SEC,
    overwrite: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_path).resolve()
    all_indices = meta.episode_indices(
        dataset_root, fallback_glob="episode_*.sync.json"
    )
    target_indices = episode_indices or all_indices

    # 第一遍：读全部 sync，收集所有由 VAD 推导出来的窗口作为参考。
    sync_by_index: dict[int, dict[str, Any]] = {}
    reference_windows: list[dict[str, Any]] = []
    for index in all_indices:
        sync = meta.read_json_or(meta.sync_json_path(dataset_root, int(index)), {})
        sync_by_index[int(index)] = sync
        window = build_vad_instruction_audio_window(
            sync.get("vad_segments"),
            audio_duration_sec=_audio_duration(sync),
            pre_margin_sec=pre_margin_sec,
            post_margin_sec=post_margin_sec,
        )
        if window is not None:
            reference_windows.append(window)

    # 第二遍：给目标 episode 写窗口（没有 VAD 就用参考窗口的中位数补）。
    updated = 0
    vad_windows = 0
    imputed_windows = 0
    pending_windows = 0
    missing_sync = 0
    for index in target_indices:
        index = int(index)
        path = meta.sync_json_path(dataset_root, index)
        sync = sync_by_index.get(index) or meta.read_json_or(path, {})
        if not path.is_file() and not sync:
            missing_sync += 1
            continue
        if not overwrite and isinstance(sync.get("instruction_audio_window"), dict):
            continue

        duration = _audio_duration(sync)
        window = build_vad_instruction_audio_window(
            sync.get("vad_segments"),
            audio_duration_sec=duration,
            pre_margin_sec=pre_margin_sec,
            post_margin_sec=post_margin_sec,
        )
        if window is not None:
            vad_windows += 1
        else:
            window = build_imputed_instruction_audio_window(
                reference_windows,
                audio_duration_sec=duration,
                pre_margin_sec=pre_margin_sec,
                post_margin_sec=post_margin_sec,
            )
            if window is not None:
                imputed_windows += 1
            else:
                window = make_pending_instruction_audio_window(
                    pre_margin_sec=pre_margin_sec,
                    post_margin_sec=post_margin_sec,
                )
                pending_windows += 1

        if sync.get("instruction_audio_window") == window:
            continue
        sync["instruction_audio_window"] = window
        updated += 1
        if not dry_run:
            meta.write_json(path, sync, indent=meta.SYNC_JSON_INDENT)

    return {
        "dataset_path": str(dataset_root),
        "episodes": len(target_indices),
        "updated": updated,
        "missing_sync": missing_sync,
        "vad_windows": vad_windows,
        "imputed_windows": imputed_windows,
        "pending_windows": pending_windows,
        "reference_window_count": len(reference_windows),
        "dry_run": bool(dry_run),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh instruction_audio_window from VAD metadata."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, action="append", default=None)
    parser.add_argument("--pre-margin-sec", type=float, default=DEFAULT_PRE_MARGIN_SEC)
    parser.add_argument(
        "--post-margin-sec", type=float, default=DEFAULT_POST_MARGIN_SEC
    )
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = refresh_dataset_instruction_audio_windows(
        args.dataset_path,
        episode_indices=args.episode_index,
        pre_margin_sec=float(args.pre_margin_sec),
        post_margin_sec=float(args.post_margin_sec),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
