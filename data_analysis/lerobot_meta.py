"""LeRobot 数据集元数据的共享读写工具。

这个模块集中了原先散落在 `data_analysis/` 各脚本里被重复实现 2～8 遍的
LeRobot 磁盘格式逻辑：JSON/JSONL 读写、episode 文件路径解析、数据集发现、
以及 `meta/` 与 parquet 的「重编号手术」。

设计约束（改动时请保持）：

- 模块顶层**只依赖标准库**。`pyarrow` 在 parquet 函数内部延迟导入，
  `numpy` 在统计函数内部延迟导入。这样 `audio/window.py`、`route_labels.py`
  这类只碰 sync.json 的模块导入本模块时依然是零成本、零重依赖。
- **不引入 import 期的全局副作用**（不改 `os.environ`、不动 `sys.path`）。
- 同一个功能如果存在**语义不同**的历史实现（`_read_json` 抛异常 vs 返回 `{}`，
  `write_json` 的 indent 2 vs 4），一律用显式参数或两个函数名保留差异，
  不做静默统一 —— 否则会改变写回磁盘的字节。
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JSON = dict[str, Any]

# LeRobot parquet 里的列名。重编号工具会改写这些列，散落的魔法字符串统一在这里。
COL_EPISODE_INDEX = "episode_index"
COL_FRAME_INDEX = "frame_index"
COL_INDEX = "index"
COL_TASK_INDEX = "task_index"
COL_TIMESTAMP = "timestamp"

# LeRobot 数据集默认的 episode 路径模板（info.json 可覆盖）。
DEFAULT_DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)

# 本地数据集的根目录，相对于仓库根。
DEFAULT_DATA_ROOT = ("data", "openpi")

# sync.json 写入方（vad / window / route_labels / asr_convert）统一用 indent=2，
# meta/ 的写入方（episode_edit / dataset_merge）统一用 indent=4。
SYNC_JSON_INDENT = 2
META_JSON_INDENT = 4


# --------------------------------------------------------------------------
# JSON / JSONL
# --------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    """读取 JSON。文件不存在或内容非法时抛异常。

    与 `read_json_or` 的区别是刻意的：多数调用方（ASR/merge/标注）希望缺失
    就是错误，而 vad/window 希望缺失等价于空配置。
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_or(path: Path, default: Any) -> Any:
    """读取 JSON；文件不存在时返回 `default`。

    注意：只有「文件不是常规文件」才回退。文件存在但内容非法时依然抛
    `json.JSONDecodeError` —— 这与重构前 vad/window 的行为一致，避免把
    真正损坏的元数据静默当成空配置。
    """
    path = Path(path)
    if not path.is_file():
        return default
    return read_json(path)


def write_json(
    path: Path,
    obj: Any,
    *,
    indent: int = META_JSON_INDENT,
    atomic: bool = True,
    mkdir: bool = False,
) -> None:
    """写 JSON。

    `atomic=True` 走 `临时文件 + os.replace`，避免中途失败留下半截文件；
    产物字节与直接写完全一致。`indent` 必须由调用方按上面的常量传，否则
    会改变磁盘上已有文件的格式。
    """
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, ensure_ascii=False) + "\n"
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    else:
        path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[JSON]:
    """读取 JSONL；文件不存在返回空列表，忽略空行与纯空白行。"""
    path = Path(path)
    if not path.exists():
        return []
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return [json.loads(line) for line in lines]


def write_jsonl(
    path: Path,
    rows: Iterable[JSON],
    *,
    atomic: bool = True,
    mkdir: bool = False,
) -> None:
    """写 JSONL，每行一个对象，末尾补换行。"""
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    if atomic:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    else:
        path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# 数值工具
# --------------------------------------------------------------------------


def safe_float(value: Any) -> float | None:
    """把值转成有限浮点数；None / 不可转 / NaN / ±inf 一律返回 None。

    原先 `instruction_audio_window` 和 `annotate_route_pairs` 各写了一份
    （一份用 `out == out and abs(out) != inf`，一份用 `np.isfinite`），
    两者语义完全一致，这里统一成不依赖 numpy 的版本。

    `OverflowError` 必须一起接住：两个原实现都是裸 `except Exception`，
    而 `float(10 ** 400)` 抛的正是 `OverflowError`（不是 `ValueError`）。
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


# --------------------------------------------------------------------------
# 数据集发现与路径解析
# --------------------------------------------------------------------------


def repo_root() -> Path:
    """仓库根目录（data_analysis/ 的上一级）。"""
    return Path(__file__).resolve().parent.parent


def default_data_root() -> Path:
    """默认的数据集根目录：<repo>/data/openpi。"""
    return repo_root().joinpath(*DEFAULT_DATA_ROOT)


def is_lerobot_dataset(path: Path) -> bool:
    """判定一个目录是否是 LeRobot 数据集根（有 meta/info.json 和 episodes.jsonl）。"""
    path = Path(path)
    return (path / "meta" / "info.json").is_file() and (
        path / "meta" / "episodes.jsonl"
    ).is_file()


def resolve_dataset(name_or_path: str, data_root: Path) -> Path:
    """把一个数据集「名字或路径」解析成实际存在的 LeRobot 数据集目录。

    依次尝试：绝对/相对路径本身 → `data_root/<name>` → `data_root` 的上一级
    `<name>`（即 `data/openpi` 的父目录，兼容 `openpi/xxx` 写法）→ CWD 相对。
    """
    raw = Path(name_or_path).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute() or raw.parts[:1] in ((".",), ("..",)):
        candidates.append(raw)
    else:
        candidates.append(data_root / raw)
        if raw.parts[:1] == ("openpi",):
            candidates.append(data_root.parent / raw)
        candidates.append(raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if is_lerobot_dataset(resolved):
            return resolved

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"找不到有效 LeRobot 数据集: {name_or_path} (tried: {tried})"
    )


def find_latest_dataset_dir(root: Path) -> Path | None:
    """在 root 下递归寻找 LeRobot 数据集目录，返回最近修改的那个。"""
    candidates: list[tuple[float, Path]] = []
    root = Path(root)
    if not root.exists():
        return None
    for info_json in root.rglob("meta/info.json"):
        dataset_dir = info_json.parent.parent
        if not is_lerobot_dataset(dataset_dir):
            continue
        try:
            mtime = info_json.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, dataset_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def episode_stem(episode_index: int) -> str:
    """`episode_000012` —— 所有 sidecar 文件共用的命名约定。"""
    return f"episode_{int(episode_index):06d}"


def episode_chunk(info: JSON, episode_index: int) -> int:
    chunks_size = int(info.get("chunks_size", 1000))
    return int(episode_index) // chunks_size


def episode_parquet_path(info: JSON, root: Path, episode_index: int) -> Path:
    template = info.get("data_path", DEFAULT_DATA_PATH_TEMPLATE)
    return Path(root) / template.format(
        episode_chunk=episode_chunk(info, episode_index),
        episode_index=episode_index,
    )


def episode_video_paths(info: JSON, root: Path, episode_index: int) -> list[Path]:
    """该 episode 对应的视频文件（可能分布在多个 video_key 子目录下）。"""
    chunk = episode_chunk(info, episode_index)
    videos_dir = Path(root) / "videos" / f"chunk-{chunk:03d}"
    if not videos_dir.is_dir():
        return []
    return sorted(videos_dir.rglob(f"{episode_stem(episode_index)}.mp4"))


def episode_audio_paths(root: Path, episode_index: int) -> list[Path]:
    """该 episode 的音频主文件与 sidecar（无论是否存在都返回，由调用方判断）。"""
    audio_dir = Path(root) / "audio"
    stem = episode_stem(episode_index)
    return [
        audio_dir / f"{stem}.wav",
        audio_dir / f"{stem}.audio.json",
        audio_dir / f"{stem}.sync.json",
        audio_dir / "vad_segments" / stem,
    ]


def episode_audio_path_map(
    root: Path, old_index: int, new_index: int
) -> dict[Path, Path]:
    """重编号时音频 sidecar 的 old→new 路径映射。"""
    audio_dir = Path(root) / "audio"
    old_stem = episode_stem(old_index)
    new_stem = episode_stem(new_index)
    return {
        audio_dir / f"{old_stem}.wav": audio_dir / f"{new_stem}.wav",
        audio_dir / f"{old_stem}.audio.json": audio_dir / f"{new_stem}.audio.json",
        audio_dir / f"{old_stem}.sync.json": audio_dir / f"{new_stem}.sync.json",
        audio_dir / "vad_segments" / old_stem: audio_dir / "vad_segments" / new_stem,
    }


def sync_json_path(root: Path, episode_index: int) -> Path:
    return Path(root) / "audio" / f"{episode_stem(episode_index)}.sync.json"


def audio_json_path(root: Path, episode_index: int) -> Path:
    return Path(root) / "audio" / f"{episode_stem(episode_index)}.audio.json"


def wav_path(root: Path, episode_index: int) -> Path:
    return Path(root) / "audio" / f"{episode_stem(episode_index)}.wav"


def parse_episode_filename(name: str) -> int | None:
    """从 `episode_000012.sync.json` / `.wav` 之类的文件名里取出 index。"""
    stem = name.removeprefix("episode_").split(".")[0]
    try:
        return int(stem)
    except ValueError:
        return None


def iter_sync_paths(root: Path) -> list[Path]:
    """数据集里全部 episode 的 sync.json，按文件名排序。"""
    return sorted((Path(root) / "audio").glob("episode_*.sync.json"))


def episode_indices(
    root: Path, *, fallback_glob: str = "episode_*.sync.json"
) -> list[int]:
    """数据集里全部 episode_index。

    优先信任 `meta/info.json` 的 `total_episodes`（假定编号是 0..N-1 连续的），
    否则回退到扫描 `audio/` 下的文件名。解析失败的文件名会被跳过。
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        try:
            total = int(read_json(info_path).get("total_episodes", 0))
        except Exception:
            total = 0
        if total > 0:
            return list(range(total))

    indices: list[int] = []
    for path in sorted((root / "audio").glob(fallback_glob)):
        parsed = parse_episode_filename(path.name)
        if parsed is not None:
            indices.append(parsed)
    return indices


# --------------------------------------------------------------------------
# 元数据手术
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int


def load_episode_infos(dataset_dir: Path) -> list[EpisodeInfo]:
    """从 meta/episodes.jsonl 读出 (episode_index, length)，按 index 排序。"""
    rows = read_jsonl(Path(dataset_dir) / "meta" / "episodes.jsonl")
    out: list[EpisodeInfo] = []
    for row in rows:
        if "episode_index" not in row or "length" not in row:
            continue
        try:
            out.append(EpisodeInfo(int(row["episode_index"]), int(row["length"])))
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda item: item.episode_index)


def update_splits_in_place(info: JSON, *, old_total: int, new_total: int) -> None:
    """把 info["splits"] 里以 old_total 结尾的区间改成 new_total。

    常见格式是 `"0:total_episodes"`；只改 end 恰好等于旧总数的项。
    """
    splits = info.get("splits")
    if not isinstance(splits, dict):
        return

    for key, value in list(splits.items()):
        if not isinstance(value, str) or ":" not in value:
            continue
        start_s, end_s = value.split(":", 1)
        try:
            start_i = int(start_s)
            end_i = int(end_s)
        except ValueError:
            continue
        if end_i == old_total:
            splits[key] = f"{start_i}:{new_total}"


def replace_column(table: Any, column_name: str, values: Any) -> Any:
    """替换一个**必须已存在**的 parquet 列；不存在就报错。"""
    field_index = table.schema.get_field_index(column_name)
    if field_index < 0:
        raise RuntimeError(f"parquet 缺少列: {column_name}")
    return table.set_column(field_index, table.schema.field(field_index), values)


def replace_or_append_column(
    table: Any, name: str, values: Any, fallback_type: Any
) -> Any:
    """替换列；列不存在时用 `fallback_type` 追加一列。"""
    import pyarrow as pa

    field_index = table.schema.get_field_index(name)
    field = (
        table.schema.field(field_index)
        if field_index >= 0
        else pa.field(name, fallback_type)
    )
    array = pa.array(values, type=field.type)
    if field_index >= 0:
        return table.set_column(field_index, field, array)
    return table.append_column(field, array)


def rewrite_parquet_episode(
    *,
    src: Path,
    dst: Path,
    new_episode_index: int,
    global_start: int,
    expected_length: int,
    remove_src: bool = False,
) -> None:
    """把一个 episode 的 parquet 重写成新的 episode_index 与全局 index。

    用于「删除后把后续 episode 前移」：只改 `episode_index` 和 `index` 两列。
    dataset_merge 的拷贝/合并路径改的列更多（还含 frame_index/task_index/
    timestamp），因此走自己的内联实现，不要强行合并到这里。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise RuntimeError(f"缺少 parquet 文件: {src}")

    table = pq.read_table(src)
    if table.num_rows != expected_length:
        raise RuntimeError(
            f"parquet 行数和 episodes.jsonl length 不一致: {src} "
            f"rows={table.num_rows}, length={expected_length}"
        )

    episode_field = table.schema.field(COL_EPISODE_INDEX)
    index_field = table.schema.field(COL_INDEX)
    table = replace_column(
        table,
        COL_EPISODE_INDEX,
        pa.array([new_episode_index] * table.num_rows, type=episode_field.type),
    )
    table = replace_column(
        table,
        COL_INDEX,
        pa.array(
            range(global_start, global_start + table.num_rows), type=index_field.type
        ),
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, dst)
    if remove_src and src != dst and src.exists():
        src.unlink()


def patch_stats_row(
    row: JSON, *, episode_index: int, global_start: int, length: int
) -> JSON:
    """重编号时定点修补 episodes_stats.jsonl 里的一行。

    只改 `episode_index` / `index` 两项统计，保留其余字段原样。
    dataset_merge 的 `fix_reindexed_stats` 会**重建**更多列，两者语义不同，
    刻意都保留。
    """
    row = dict(row)
    row[COL_EPISODE_INDEX] = episode_index

    stats = row.get("stats")
    if not isinstance(stats, dict):
        return row

    ep_stats = stats.get(COL_EPISODE_INDEX)
    if isinstance(ep_stats, dict):
        ep_stats["min"] = [episode_index]
        ep_stats["max"] = [episode_index]
        ep_stats["mean"] = [float(episode_index)]
        ep_stats["std"] = [0.0]
        ep_stats["count"] = [length]

    index_stats = stats.get(COL_INDEX)
    if isinstance(index_stats, dict):
        end = global_start + length - 1
        frame_stats = stats.get(COL_FRAME_INDEX)
        frame_std = None
        if isinstance(frame_stats, dict):
            std = frame_stats.get("std")
            if isinstance(std, list) and std:
                frame_std = std[0]
        index_stats["min"] = [global_start]
        index_stats["max"] = [end]
        index_stats["mean"] = [float(global_start + end) / 2.0]
        index_stats["std"] = [frame_std if frame_std is not None else 0.0]
        index_stats["count"] = [length]

    return row


def _stat(values: Any) -> JSON:
    if values.size == 0:
        return {"min": [0], "max": [0], "mean": [0.0], "std": [0.0], "count": [0]}
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
    }


def fix_reindexed_stats(
    row: JSON,
    *,
    new_episode_index: int,
    frame_start: int,
    length: int,
    fps: float,
    task_indices: Any,
) -> JSON:
    """合并数据集时重建一行 episodes_stats —— 比 `patch_stats_row` 覆盖更多列。

    这里重建 `episode_index`/`frame_index`/`index`/`task_index`/`timestamp`。
    """
    import numpy as np

    fixed = json.loads(json.dumps(row))
    fixed[COL_EPISODE_INDEX] = new_episode_index
    stats = fixed.get("stats")
    if not isinstance(stats, dict):
        return fixed

    frame_index = np.arange(length, dtype=np.int64)
    global_index = frame_start + frame_index
    timestamps = frame_index.astype(np.float64) / float(fps)

    stats[COL_EPISODE_INDEX] = _stat(np.full(length, new_episode_index, dtype=np.int64))
    stats[COL_FRAME_INDEX] = _stat(frame_index)
    stats[COL_INDEX] = _stat(global_index)
    stats[COL_TASK_INDEX] = _stat(task_indices.astype(np.int64, copy=False))
    stats[COL_TIMESTAMP] = {
        "min": [float(timestamps.min()) if length else 0.0],
        "max": [float(timestamps.max()) if length else 0.0],
        "mean": [float(timestamps.mean()) if length else 0.0],
        "std": [float(timestamps.std()) if length else 0.0],
        "count": [length],
    }
    return fixed


def patch_audio_json(path: Path, dataset_dir: Path, episode_index: int) -> None:
    """把 .audio.json 里的 audio_path 指向重编号后的 wav（绝对路径）。

    注意这里的 `indent` 用的是 `META_JSON_INDENT`(4)，与 vad/window 写这些
    sidecar 时的 2 不一致 —— 这是**刻意保留**的历史行为：重构前的删除脚本
    就是这样重排格式的，改掉会让同样的操作产出不同的字节。要统一请单独提
    一个改动，别顺手改。
    """
    path = Path(path)
    if not path.exists():
        return
    payload = read_json(path)
    payload["audio_path"] = str(
        (Path(dataset_dir) / "audio" / f"{episode_stem(episode_index)}.wav").resolve()
    )
    write_json(path, payload, indent=META_JSON_INDENT)


def patch_sync_json(path: Path, episode_index: int) -> None:
    """把 .sync.json 里的 episode_index 与两个音频路径改成重编号后的值。

    `indent` 同 `patch_audio_json`，为保持历史行为用 4。
    """
    path = Path(path)
    if not path.exists():
        return
    payload = read_json(path)
    stem = episode_stem(episode_index)
    payload[COL_EPISODE_INDEX] = episode_index
    payload["audio_path"] = f"audio/{stem}.wav"
    payload["audio_metadata_path"] = f"audio/{stem}.audio.json"
    write_json(path, payload, indent=META_JSON_INDENT)


# --------------------------------------------------------------------------
# 文件系统
# --------------------------------------------------------------------------


def remove_path(path: Path) -> None:
    """删除文件或目录；不存在则什么都不做。"""
    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def move_path(src: Path, dst: Path) -> None:
    """移动文件/目录；src 不存在则跳过，dst 已存在则拒绝覆盖。"""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"目标路径已存在，拒绝覆盖: {dst}")
    shutil.move(str(src), str(dst))


def cleanup_empty_dirs(root: Path) -> None:
    """删除后清理 data/ 与 videos/ 里空掉的目录。"""
    root = Path(root)
    for dirname in ("data", "videos"):
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
