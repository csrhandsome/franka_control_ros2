"""删除 LeRobot 数据集里的 episode，并把后续编号前移。

合并了原先的两个脚本：

- `delete_lastest_episode_by_id.py`（按 episode_index 删除，带重编号）
- `delete_latest_episode.py`（删除编号最大的那个）

前者本来就是后者的严格超集，这里把「删最新」表达成 `--latest`，共用同一套
删除 + 重编号逻辑。

默认**只做 dry-run**，确认预览无误后加 `--yes` 才真正落盘。

会同步处理：
- `data/chunk-*/episode_*.parquet`（改写 episode_index / index 并重命名）
- `videos/chunk-*/*/episode_*.mp4`（如果存在）
- `audio/episode_*.wav`、`.audio.json`、`.sync.json`
- `audio/vad_segments/episode_*`（如果存在）
- `meta/episodes.jsonl`、`meta/episodes_stats.jsonl`、`meta/info.json`

用法示例：

  # 预览删除 episode 12
  uv run -m data_analysis.episode_edit \\
    --dataset data/openpi/franka_lerobot_4_9_audio --episode-index 12

  # 确认后执行
  uv run -m data_analysis.episode_edit \\
    --dataset data/openpi/franka_lerobot_4_9_audio --episode-index 12 --yes

  # 采集中断后丢弃最后一次(编号最大)的 episode
  uv run -m data_analysis.episode_edit --dataset data/openpi/xxx --latest --yes
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_analysis import lerobot_meta as meta
from data_analysis.lerobot_meta import EpisodeInfo


@dataclass(frozen=True)
class EpisodeMove:
    old_index: int
    new_index: int
    length: int
    global_start: int


def _assert_contiguous_episode_infos(episode_infos: list[EpisodeInfo]) -> None:
    expected = list(range(len(episode_infos)))
    actual = [info.episode_index for info in episode_infos]
    if actual != expected:
        raise RuntimeError(
            "episodes.jsonl 里的 episode_index 不是连续的 0..N-1，拒绝自动重排。\n"
            f"Actual: {actual[:20]}{'...' if len(actual) > 20 else ''}"
        )


def _episode_video_destination(
    info: dict[str, Any], dataset_dir: Path, src_video: Path, new_index: int
) -> Path:
    """重编号后视频应落到的位置：沿用原 video_key 子目录名，只换 chunk 与文件名。"""
    new_chunk = meta.episode_chunk(info, new_index)
    video_key = src_video.parent.name
    return (
        dataset_dir
        / "videos"
        / f"chunk-{new_chunk:03d}"
        / video_key
        / f"{meta.episode_stem(new_index)}.mp4"
    )


def _build_moves(
    episode_infos: list[EpisodeInfo], delete_index: int
) -> list[EpisodeMove]:
    """算出删除后每个幸存 episode 的新编号与全局起始帧。"""
    moves: list[EpisodeMove] = []
    global_start = 0
    new_index = 0
    for info in episode_infos:
        if info.episode_index == delete_index:
            continue
        moves.append(
            EpisodeMove(
                old_index=info.episode_index,
                new_index=new_index,
                length=info.length,
                global_start=global_start,
            )
        )
        global_start += info.length
        new_index += 1
    return moves


def _rewrite_metadata(
    *,
    dataset_dir: Path,
    info: dict[str, Any],
    moves: list[EpisodeMove],
    delete_index: int,
    old_total: int,
) -> None:
    """重写 meta/episodes.jsonl、episodes_stats.jsonl、info.json。"""
    meta_dir = dataset_dir / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"
    info_path = meta_dir / "info.json"

    move_by_old = {move.old_index: move for move in moves}

    episodes_rows = meta.read_jsonl(episodes_path)
    new_episodes_rows: list[dict[str, Any]] = []
    for row in episodes_rows:
        old_idx = int(row.get(meta.COL_EPISODE_INDEX, -1))
        if old_idx == delete_index:
            continue
        move = move_by_old[old_idx]
        row = dict(row)
        row[meta.COL_EPISODE_INDEX] = move.new_index
        new_episodes_rows.append(row)
    new_episodes_rows.sort(key=lambda row: int(row[meta.COL_EPISODE_INDEX]))
    meta.write_jsonl(episodes_path, new_episodes_rows)

    if stats_path.exists():
        stats_rows = meta.read_jsonl(stats_path)
        new_stats_rows: list[dict[str, Any]] = []
        for row in stats_rows:
            old_idx = int(row.get(meta.COL_EPISODE_INDEX, -1))
            if old_idx == delete_index:
                continue
            move = move_by_old[old_idx]
            new_stats_rows.append(
                meta.patch_stats_row(
                    row,
                    episode_index=move.new_index,
                    global_start=move.global_start,
                    length=move.length,
                )
            )
        new_stats_rows.sort(key=lambda row: int(row[meta.COL_EPISODE_INDEX]))
        meta.write_jsonl(stats_path, new_stats_rows)

    new_total = len(moves)
    chunks_size = int(info.get("chunks_size", 1000))
    info["total_episodes"] = new_total
    info["total_frames"] = sum(move.length for move in moves)
    info["total_chunks"] = (
        (new_total + chunks_size - 1) // chunks_size if new_total > 0 else 0
    )
    meta.update_splits_in_place(info, old_total=old_total, new_total=new_total)
    meta.write_json(info_path, info)


def _print_plan(
    *,
    dataset_dir: Path,
    info: dict[str, Any],
    episode_infos: list[EpisodeInfo],
    delete_index: int,
    moves: list[EpisodeMove],
) -> None:
    delete_info = next(
        item for item in episode_infos if item.episode_index == delete_index
    )
    old_total = len(episode_infos)
    new_total = len(moves)
    old_frames = sum(item.length for item in episode_infos)
    new_frames = sum(move.length for move in moves)

    print(f"Dataset: {dataset_dir}")
    print(f"Delete episode_index: {delete_index}")
    print(f"Deleted frames: {delete_info.length}")
    print(f"Total episodes: {old_total} -> {new_total}")
    print(f"Total frames: {old_frames} -> {new_frames}")
    print(
        f"Remove parquet: {meta.episode_parquet_path(info, dataset_dir, delete_index)}"
    )

    video_paths = meta.episode_video_paths(info, dataset_dir, delete_index)
    audio_paths = [
        path
        for path in meta.episode_audio_paths(dataset_dir, delete_index)
        if path.exists()
    ]
    print(f"Remove videos: {len(video_paths)}")
    print(f"Remove audio/sidecars: {len(audio_paths)}")

    renumbered = [move for move in moves if move.old_index != move.new_index]
    if renumbered:
        first = renumbered[0]
        last = renumbered[-1]
        print(
            "Renumber episodes: "
            f"{first.old_index:06d}->{first.new_index:06d} ... "
            f"{last.old_index:06d}->{last.new_index:06d}"
        )
    else:
        print("Renumber episodes: none (deleting latest episode)")


def delete_episode_by_id(dataset_dir: Path, episode_index: int, *, yes: bool) -> int:
    """删除指定 episode_index 并重编号后续 episode。`yes=False` 时只预览。"""
    dataset_dir = Path(dataset_dir).resolve()
    if not meta.is_lerobot_dataset(dataset_dir):
        print(f"[ERROR] 不是有效的 LeRobot 数据集目录: {dataset_dir}", file=sys.stderr)
        return 2

    if episode_index < 0:
        print("[ERROR] episode_index 必须 >= 0", file=sys.stderr)
        return 2

    info = meta.read_json(dataset_dir / "meta" / "info.json")
    episode_infos = meta.load_episode_infos(dataset_dir)
    if not episode_infos:
        print("[WARN] episodes.jsonl 为空，没有可删除的 episode。")
        return 0

    try:
        _assert_contiguous_episode_infos(episode_infos)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if episode_index not in {item.episode_index for item in episode_infos}:
        print(f"[ERROR] episode_index 不存在: {episode_index}", file=sys.stderr)
        return 2

    for episode in episode_infos:
        parquet_path = meta.episode_parquet_path(
            info, dataset_dir, episode.episode_index
        )
        if not parquet_path.exists():
            print(f"[ERROR] 缺少 parquet 文件: {parquet_path}", file=sys.stderr)
            return 2

    old_total = len(episode_infos)
    moves = _build_moves(episode_infos, episode_index)
    _print_plan(
        dataset_dir=dataset_dir,
        info=info,
        episode_infos=episode_infos,
        delete_index=episode_index,
        moves=moves,
    )

    if not yes:
        print("[DRY-RUN] 未修改任何文件。确认无误后加 --yes 执行硬删除。")
        return 0

    # 1) 删除目标 episode 的数据文件。
    meta.remove_path(meta.episode_parquet_path(info, dataset_dir, episode_index))
    for path in meta.episode_video_paths(info, dataset_dir, episode_index):
        meta.remove_path(path)
    for path in meta.episode_audio_paths(dataset_dir, episode_index):
        meta.remove_path(path)

    # 2) 重写并前移后续 episode 的 parquet，同时修正内部 episode_index/index。
    for move in moves:
        if move.old_index == move.new_index:
            continue
        meta.rewrite_parquet_episode(
            src=meta.episode_parquet_path(info, dataset_dir, move.old_index),
            dst=meta.episode_parquet_path(info, dataset_dir, move.new_index),
            new_episode_index=move.new_index,
            global_start=move.global_start,
            expected_length=move.length,
            remove_src=True,
        )

    # 3) 前移视频和音频 sidecar，并修正 JSON 里的路径 / index。
    for move in moves:
        if move.old_index == move.new_index:
            continue

        for src_video in meta.episode_video_paths(info, dataset_dir, move.old_index):
            meta.move_path(
                src_video,
                _episode_video_destination(
                    info, dataset_dir, src_video, move.new_index
                ),
            )

        for src_audio, dst_audio in meta.episode_audio_path_map(
            dataset_dir, move.old_index, move.new_index
        ).items():
            meta.move_path(src_audio, dst_audio)

        meta.patch_audio_json(
            meta.audio_json_path(dataset_dir, move.new_index),
            dataset_dir,
            move.new_index,
        )
        meta.patch_sync_json(
            meta.sync_json_path(dataset_dir, move.new_index), move.new_index
        )

    # 4) 更新 meta。
    _rewrite_metadata(
        dataset_dir=dataset_dir,
        info=info,
        moves=moves,
        delete_index=episode_index,
        old_total=old_total,
    )
    meta.cleanup_empty_dirs(dataset_dir)

    print("[OK] 已删除指定 episode，并完成后续 episode 重编号。")
    return 0


def delete_latest_episode(dataset_dir: Path, *, yes: bool = True) -> int:
    """删除编号最大的那个 episode。

    默认 `yes=True` 以保持原 `delete_latest_episode.py` 的「直接硬删除」语义
    （采集中断后的清理路径依赖它）。删最新不会触发任何重编号，等价于
    `delete_episode_by_id(dir, max_index, yes=yes)`。
    """
    episode_infos = meta.load_episode_infos(dataset_dir)
    if not episode_infos:
        if not meta.is_lerobot_dataset(dataset_dir):
            print(
                f"[ERROR] 不是有效的 LeRobot 数据集目录: {Path(dataset_dir).resolve()}",
                file=sys.stderr,
            )
            return 2
        print("[WARN] episodes.jsonl 为空，没有可删除的 episode。")
        return 0
    latest_index = max(item.episode_index for item in episode_infos)
    return delete_episode_by_id(dataset_dir, latest_index, yes=yes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="删除 LeRobot episode（按 episode_index 或删最新），并重排后续编号"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="数据集目录（包含 meta/info.json）。不填则自动在 data/openpi 下找最近修改的数据集。",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="要删除的 episode_index，例如 12。",
    )
    target.add_argument(
        "--latest",
        action="store_true",
        help="删除编号最大的 episode（采集中断后丢弃最后一次）。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="真正执行硬删除。默认不加时只 dry-run 预览。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_dir: Path | None = args.dataset
    if dataset_dir is None:
        auto_root = Path("data") / "openpi"
        dataset_dir = meta.find_latest_dataset_dir(auto_root)
        if dataset_dir is None:
            print(
                f"[ERROR] 未找到数据集目录（在 {auto_root} 下没有 meta/info.json + episodes.jsonl）。",
                file=sys.stderr,
            )
            return 2
        print(f"[AUTO] 选择最近修改的数据集: {dataset_dir}")

    if args.latest:
        return delete_latest_episode(dataset_dir, yes=bool(args.yes))
    return delete_episode_by_id(
        dataset_dir, int(args.episode_index), yes=bool(args.yes)
    )


if __name__ == "__main__":
    raise SystemExit(main())
