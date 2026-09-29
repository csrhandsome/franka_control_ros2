#!/usr/bin/env python3
"""把音频数据集转成 ASR 提示词数据集，两个后端：whisper / qwen3-asr。

非破坏性：默认把输入数据集拷贝成 ``<name>_asr`` / ``<name>_qwen_asr``，只改副本。

用法：
  uv run -m data_analysis.audio.asr_convert whisper --overwrite-output
  uv run -m data_analysis.audio.asr_convert qwen \
      --input-dir data/openpi/franka_lerobot_6_19_audio \
      --output-dir data/openpi/franka_lerobot_6_19_audio_asr \
      --overwrite-output --asr-model Qwen/Qwen3-ASR-0.6B

拷贝、逐段转写、重写 meta/ 与 parquet 的流程只有一份；两个后端的差异（模型、
language 归一化、写进 sync.json 的 `asr_metadata` 字段、横幅文案）都收在
`WhisperBackend` / `QwenBackend` 里。用**子命令**而不是 `--backend` 开关，是为了
让两边的默认值（模型名、language、max_new_tokens、输出目录后缀）互不污染。

两个后端的 `asr_metadata` 结构本来就不同（qwen 多 backend/dtype/context/
detected_languages 等），而且这些字典是**按键序**原样写进 sync.json 的，所以
各自维护自己那一份，不做「公共字段 + 补丁」式的拼装。

**刻意保留的历史行为**：`meta/info.json` 在这里也用 ``indent=2`` 重写，
而删除（`episode_edit`）和合并（`dataset_merge`）用 4。改动会改变写盘字节。
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from control.util.audio_util import read_wav_pcm, transcribe_whisper_asr
from data_analysis import lerobot_meta as meta

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_INPUT_DIR = Path("data/openpi/franka_lerobot_4_9_audio")
DEFAULT_QWEN3_ASR_REPO = meta.repo_root() / "third_party" / "Qwen3-ASR"


@dataclass(frozen=True)
class Transcription:
    """后端无关的转写结果。

    whisper 只填 ``text``；qwen 还会带上它自己判定的 ``language``。
    """

    text: str
    language: str | None = None


@dataclass(frozen=True)
class PromptContext:
    """`_collect_episode_prompt` 算出来的、各后端都要写进 metadata 的公共部分。"""

    source: str
    transcript_text: str
    fallback_task: str
    used_asr: bool
    num_vad_segments: int
    errors: list[dict[str, str]]
    detected_languages: list[str]


def _use_hf_mirror() -> None:
    """默认走 hf-mirror，已经设过就不动。

    必须在 ``transformers`` / ``huggingface_hub`` **被 import 之前**调用 ——
    `huggingface_hub.HF_ENDPOINT` 是在 import 期读环境变量定下的常量，之后再设
    就没用了。所以这段从原来的模块顶层挪进了后端构造函数（`control/util/
    audio_util.py` 里的 torch/transformers 都是函数内延迟 import，构造后端必然
    早于第一次转写），既能生效，又不会让 `import asr_convert` 污染进程环境。
    """
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    os.environ.setdefault("HF_HUB_ENDPOINT", DEFAULT_HF_ENDPOINT)


def _resolve_existing_path(path: Path) -> Path:
    """相对路径先按 CWD 找，找不到再按仓库根找。"""
    if path.is_absolute():
        return path.resolve()

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    project_path = (meta.repo_root() / path).resolve()
    if project_path.exists():
        return project_path

    return cwd_path


def _default_output_dir(input_dir: Path, suffix: str) -> Path:
    return input_dir.with_name(f"{input_dir.name}{suffix}")


def _resolve_dataset_relative_path(dataset_dir: Path, raw_path: object) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return dataset_dir / path


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fallback_task(episode_row: dict[str, Any], sync_data: dict[str, Any]) -> str:
    """没有转写文本时用什么当提示词：episodes.jsonl 的 tasks → sync 的 task → 空串。"""
    tasks = episode_row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    task = sync_data.get("task")
    if task:
        return str(task)
    return ""


# --------------------------------------------------------------------------
# 后端
# --------------------------------------------------------------------------


class WhisperBackend:
    """openai/whisper-* 模型，走 `control.util.audio_util.transcribe_whisper_asr`。"""

    name = "whisper"
    output_suffix = "_asr"
    summary_title = "ASR conversion summary"
    tqdm_desc = "Transcribing VAD clips"

    def __init__(self, args: argparse.Namespace) -> None:
        _use_hf_mirror()
        self.model_name = str(args.asr_model)
        self.device = str(args.asr_device)
        self.language = args.asr_language
        self.max_new_tokens = int(args.asr_max_new_tokens)

    def prepare(self, args: argparse.Namespace) -> None:
        pass

    def transcribe(self, wav_path: Path) -> Transcription:
        audio, sample_rate = read_wav_pcm(wav_path, require_nonempty=True)
        text = transcribe_whisper_asr(
            audio,
            sample_rate=int(sample_rate),
            model_name=self.model_name,
            device=self.device,
            language=self.language,
            max_new_tokens=int(self.max_new_tokens),
        )
        return Transcription(text=text)

    def segment_fields(self, transcription: Transcription) -> dict[str, Any]:
        return {}

    def build_metadata(self, ctx: PromptContext) -> dict[str, Any]:
        return {
            "enabled": True,
            "model": self.model_name,
            "device": self.device,
            "language": self.language,
            "source": ctx.source,
            "transcript": ctx.transcript_text,
            "fallback_task": ctx.fallback_task,
            "used_for_task": ctx.used_asr,
            "num_vad_segments": ctx.num_vad_segments,
            "errors": ctx.errors,
            "processed_at": datetime.now(UTC).isoformat(),
        }

    def banner_lines(self, input_dir: Path, output_dir: Path) -> list[str]:
        return [
            f"Input dataset:  {input_dir}",
            f"Output dataset: {output_dir}",
            f"ASR model:      {self.model_name}",
            f"ASR device:     {self.device}",
            f"ASR language:   {self.language or 'auto'}",
        ]


class QwenBackend:
    """Qwen3-ASR。没装包时可以回退到本地 checkout（`QWEN3_ASR_REPO`）。"""

    name = "qwen"
    output_suffix = "_qwen_asr"
    summary_title = "Qwen3-ASR conversion summary"
    tqdm_desc = "Transcribing VAD clips with Qwen3-ASR"

    def __init__(self, args: argparse.Namespace) -> None:
        _use_hf_mirror()
        self.model_name = str(args.asr_model)
        self.device = str(args.asr_device)
        self.language = args.asr_language
        self.dtype = str(args.asr_dtype)
        self.max_inference_batch_size = int(args.asr_max_inference_batch_size)
        self.max_new_tokens = int(args.asr_max_new_tokens)
        self.attn_implementation = args.asr_attn_implementation
        self.context = str(args.asr_context)

    def prepare(self, args: argparse.Namespace) -> None:
        if args.qwen3_asr_repo is not None:
            os.environ["QWEN3_ASR_REPO"] = str(args.qwen3_asr_repo)

    def transcribe(self, wav_path: Path) -> Transcription:
        audio, sample_rate = read_wav_pcm(wav_path, require_nonempty=True)
        model, _, _ = _load_qwen3_asr_model(
            self.model_name,
            self.device,
            self.dtype,
            int(self.max_inference_batch_size),
            int(self.max_new_tokens),
            self.attn_implementation,
        )
        results = model.transcribe(
            audio=(audio, int(sample_rate)),
            context=self.context,
            language=_normalize_qwen_language(self.language),
            return_time_stamps=False,
        )
        if not results:
            return Transcription(text="", language="")
        result = results[0]
        return Transcription(
            text=_extract_result_field(result, "text").strip(),
            language=_extract_result_field(result, "language").strip(),
        )

    def segment_fields(self, transcription: Transcription) -> dict[str, Any]:
        return {"asr_language": transcription.language or None}

    def build_metadata(self, ctx: PromptContext) -> dict[str, Any]:
        return {
            "enabled": True,
            "backend": "qwen3-asr",
            "model": self.model_name,
            "device": self.device,
            "language": _normalize_qwen_language(self.language),
            "dtype": self.dtype,
            "max_inference_batch_size": int(self.max_inference_batch_size),
            "max_new_tokens": int(self.max_new_tokens),
            "attn_implementation": self.attn_implementation,
            "context": self.context,
            "source": ctx.source,
            "transcript": ctx.transcript_text,
            "detected_languages": ctx.detected_languages,
            "fallback_task": ctx.fallback_task,
            "used_for_task": ctx.used_asr,
            "num_vad_segments": ctx.num_vad_segments,
            "errors": ctx.errors,
            "processed_at": datetime.now(UTC).isoformat(),
        }

    def banner_lines(self, input_dir: Path, output_dir: Path) -> list[str]:
        return [
            f"Input dataset:       {input_dir}",
            f"Output dataset:      {output_dir}",
            "ASR backend:         Qwen3-ASR",
            f"ASR model:           {self.model_name}",
            f"ASR device:          {self.device}",
            f"ASR dtype:           {self.dtype}",
            f"ASR language:        {self.language or 'auto'}",
            f"ASR max new tokens:  {self.max_new_tokens}",
        ]


def _maybe_add_qwen3_asr_repo_to_path() -> None:
    candidates = []
    env_repo = os.environ.get("QWEN3_ASR_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.append(DEFAULT_QWEN3_ASR_REPO)

    for repo_dir in candidates:
        package_init = repo_dir / "qwen_asr" / "__init__.py"
        if package_init.is_file():
            resolved = str(repo_dir.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            return


def _import_qwen3_asr_model():
    try:
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel
    except ModuleNotFoundError as exc:
        if exc.name != "qwen_asr":
            raise

    _maybe_add_qwen3_asr_repo_to_path()
    try:
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel
    except ModuleNotFoundError as exc:
        if exc.name == "qwen_asr":
            raise RuntimeError(
                "Qwen3-ASR requires `qwen-asr`. Install it, or set "
                "QWEN3_ASR_REPO to a local Qwen3-ASR checkout."
            ) from exc
        raise


def _resolve_qwen_device_map(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(dtype: str, resolved_device: str):
    import torch

    normalized = dtype.strip().lower()
    if normalized == "auto":
        normalized = "bfloat16" if resolved_device.startswith("cuda") else "float32"

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in dtype_by_name:
        raise ValueError(
            f"Unsupported --asr-dtype {dtype!r}; use auto, bfloat16, float16, or float32."
        )
    return dtype_by_name[normalized], normalized


@functools.lru_cache(maxsize=4)
def _load_qwen3_asr_model(
    model_name: str,
    device: str,
    dtype: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
    attn_implementation: str | None,
):
    Qwen3ASRModel = _import_qwen3_asr_model()
    resolved_device = _resolve_qwen_device_map(device)
    torch_dtype, resolved_dtype = _resolve_torch_dtype(dtype, resolved_device)

    kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": resolved_device,
        "max_inference_batch_size": int(max_inference_batch_size),
        "max_new_tokens": int(max_new_tokens),
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = Qwen3ASRModel.from_pretrained(model_name, **kwargs)
    return model, resolved_device, resolved_dtype


def _normalize_qwen_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    if not value or value.lower() in {"auto", "none", "null"}:
        return None

    aliases = {
        "zh": "Chinese",
        "cn": "Chinese",
        "chs": "Chinese",
        "chinese": "Chinese",
        "mandarin": "Chinese",
        "en": "English",
        "eng": "English",
        "english": "English",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
    }
    return aliases.get(value.lower(), value[:1].upper() + value[1:].lower())


def _extract_result_field(result: object, field: str) -> str:
    if isinstance(result, dict):
        return str(result.get(field) or "")
    return str(getattr(result, field, "") or "")


# --------------------------------------------------------------------------
# 共享流程
# --------------------------------------------------------------------------


def _collect_episode_prompt(
    *,
    dataset_dir: Path,
    episode_row: dict[str, Any],
    backend: Any,
    transcribe_full_audio_when_no_vad: bool,
    dry_run: bool,
) -> tuple[str, dict[str, Any]]:
    episode_index = int(episode_row["episode_index"])
    stem = meta.episode_stem(episode_index)
    sync_path = meta.sync_json_path(dataset_dir, episode_index)
    sync_data = meta.read_json_or(sync_path, {})
    fallback_task = _fallback_task(episode_row, sync_data)

    vad_segments = sync_data.get("vad_segments")
    if not isinstance(vad_segments, list):
        vad_segments = []

    transcript_parts: list[str] = []
    detected_languages: list[str] = []
    errors: list[dict[str, str]] = []
    source = "vad_segments"

    for segment in vad_segments:
        if not isinstance(segment, dict):
            continue
        wav_path = _resolve_dataset_relative_path(
            dataset_dir, segment.get("audio_wav_path")
        )
        if wav_path is None or not wav_path.is_file():
            continue
        try:
            transcription = backend.transcribe(wav_path)
            transcript = _normalize_prompt(transcription.text)
            if transcript:
                transcript_parts.append(transcript)
            if transcription.language:
                detected_languages.append(transcription.language)
            if not dry_run:
                segment["asr_transcript"] = transcript
                segment.update(backend.segment_fields(transcription))
                segment["asr_error"] = None
        except Exception as exc:
            errors.append(
                {
                    "source": str(wav_path),
                    "error": str(exc),
                }
            )
            if not dry_run:
                segment["asr_error"] = str(exc)

    if not transcript_parts and transcribe_full_audio_when_no_vad and not vad_segments:
        wav_path = _resolve_dataset_relative_path(
            dataset_dir, sync_data.get("audio_path")
        )
        if wav_path is None:
            wav_path = dataset_dir / "audio" / f"{stem}.wav"
        if wav_path.is_file():
            source = "full_episode_audio"
            try:
                transcription = backend.transcribe(wav_path)
                transcript = _normalize_prompt(transcription.text)
                if transcript:
                    transcript_parts.append(transcript)
                if transcription.language:
                    detected_languages.append(transcription.language)
            except Exception as exc:
                errors.append(
                    {
                        "source": str(wav_path),
                        "error": str(exc),
                    }
                )

    transcript_text = _normalize_prompt(" ".join(transcript_parts))
    prompt = transcript_text or fallback_task
    used_asr = bool(transcript_text)

    metadata = backend.build_metadata(
        PromptContext(
            source=source,
            transcript_text=transcript_text,
            fallback_task=fallback_task,
            used_asr=used_asr,
            num_vad_segments=len(vad_segments),
            errors=errors,
            detected_languages=detected_languages,
        )
    )

    if not dry_run:
        sync_data["episode_index"] = episode_index
        sync_data["task"] = prompt
        sync_data["fallback_task"] = fallback_task
        sync_data["asr_metadata"] = metadata
        sync_data["vad_segments"] = vad_segments
        meta.write_json(sync_path, sync_data, indent=meta.SYNC_JSON_INDENT)

    return prompt, metadata


def _rewrite_parquet_task_index(
    *,
    path: Path,
    task_index: int,
    expected_length: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing parquet file: {path}")

    table = pq.read_table(path)
    if table.num_rows != expected_length:
        raise RuntimeError(
            f"Parquet row count does not match episodes.jsonl length: {path} "
            f"rows={table.num_rows}, length={expected_length}"
        )

    field = table.schema.field("task_index")
    values = pa.array([int(task_index)] * table.num_rows, type=field.type)
    table = meta.replace_column(table, "task_index", values)

    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _patch_task_index_stats(
    row: dict[str, Any], *, task_index: int, length: int
) -> dict[str, Any]:
    row = dict(row)
    stats = row.get("stats")
    if not isinstance(stats, dict):
        return row

    task_stats = stats.get("task_index")
    if isinstance(task_stats, dict):
        task_stats["min"] = [int(task_index)]
        task_stats["max"] = [int(task_index)]
        task_stats["mean"] = [float(task_index)]
        task_stats["std"] = [0.0]
        task_stats["count"] = [int(length)]
    return row


def _rewrite_metadata_and_parquet(
    *,
    dataset_dir: Path,
    episode_rows: list[dict[str, Any]],
    prompts_by_episode: dict[int, str],
    dry_run: bool,
) -> dict[str, Any]:
    """把每个 episode 的提示词去重成 task 表，重写 tasks/episodes/info 与 parquet。

    注意 `meta/info.json` 这里也是 indent=2（与删除/合并脚本的 4 不同），
    这是转写脚本一直以来的行为，别顺手统一。
    """
    meta_dir = dataset_dir / "meta"
    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"

    info = meta.read_json(info_path)

    task_to_index: dict[str, int] = {}
    tasks_rows: list[dict[str, Any]] = []
    episode_task_indices: dict[int, int] = {}

    new_episode_rows: list[dict[str, Any]] = []
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        prompt = prompts_by_episode[episode_index]
        if prompt not in task_to_index:
            task_to_index[prompt] = len(task_to_index)
            tasks_rows.append(
                {
                    "task_index": task_to_index[prompt],
                    "task": prompt,
                }
            )
        task_index = task_to_index[prompt]
        episode_task_indices[episode_index] = task_index

        new_row = dict(row)
        new_row["tasks"] = [prompt]
        new_episode_rows.append(new_row)

    if dry_run:
        return {
            "num_tasks": len(tasks_rows),
            "episode_task_indices": episode_task_indices,
            "tasks": tasks_rows,
        }

    meta.write_jsonl(tasks_path, tasks_rows)
    meta.write_jsonl(episodes_path, new_episode_rows)

    for row in tqdm(new_episode_rows, desc="Rewriting parquet task_index"):
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        parquet_path = meta.episode_parquet_path(info, dataset_dir, episode_index)
        _rewrite_parquet_task_index(
            path=parquet_path,
            task_index=episode_task_indices[episode_index],
            expected_length=length,
        )

    if stats_path.exists():
        stats_rows = meta.read_jsonl(stats_path)
        length_by_episode = {
            int(row["episode_index"]): int(row["length"]) for row in new_episode_rows
        }
        patched_stats = []
        for row in stats_rows:
            episode_index = int(row.get("episode_index", -1))
            patched_stats.append(
                _patch_task_index_stats(
                    row,
                    task_index=episode_task_indices[episode_index],
                    length=length_by_episode[episode_index],
                )
            )
        meta.write_jsonl(stats_path, patched_stats)

    info["total_tasks"] = len(tasks_rows)
    meta.write_json(info_path, info, indent=meta.SYNC_JSON_INDENT)

    return {
        "num_tasks": len(tasks_rows),
        "episode_task_indices": episode_task_indices,
        "tasks": tasks_rows,
    }


def convert_dataset(args: argparse.Namespace) -> int:
    backend = args.backend_factory(args)
    backend.prepare(args)

    input_dir = _resolve_existing_path(args.input_dir)
    if not meta.is_lerobot_dataset(input_dir):
        print(f"[ERROR] Not a LeRobot dataset: {input_dir}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else _default_output_dir(input_dir, backend.output_suffix)
    )

    if input_dir == output_dir and not args.dry_run:
        print(
            "[ERROR] Refusing to overwrite the input dataset in place.", file=sys.stderr
        )
        return 2

    for line in backend.banner_lines(input_dir, output_dir):
        print(line)

    if not args.dry_run:
        if output_dir.exists():
            if not args.overwrite_output:
                print(
                    "[ERROR] Output directory already exists. "
                    "Use --overwrite-output to replace it.",
                    file=sys.stderr,
                )
                return 2
            shutil.rmtree(output_dir)
        print("[Copy] Copying dataset...")
        shutil.copytree(input_dir, output_dir)
        work_dir = output_dir
    else:
        print("[Dry-run] No files will be written.")
        work_dir = input_dir

    episodes_path = work_dir / "meta" / "episodes.jsonl"
    episode_rows = meta.read_jsonl(episodes_path)
    if not episode_rows:
        print(f"[ERROR] No episodes found in {episodes_path}", file=sys.stderr)
        return 2

    prompts_by_episode: dict[int, str] = {}
    asr_used = 0
    errors = 0

    for row in tqdm(episode_rows, desc=backend.tqdm_desc):
        episode_index = int(row["episode_index"])
        prompt, metadata = _collect_episode_prompt(
            dataset_dir=work_dir,
            episode_row=row,
            backend=backend,
            transcribe_full_audio_when_no_vad=args.transcribe_full_audio_when_no_vad,
            dry_run=args.dry_run,
        )
        prompts_by_episode[episode_index] = prompt
        asr_used += int(bool(metadata["used_for_task"]))
        errors += len(metadata["errors"])

    rewrite_summary = _rewrite_metadata_and_parquet(
        dataset_dir=work_dir,
        episode_rows=episode_rows,
        prompts_by_episode=prompts_by_episode,
        dry_run=args.dry_run,
    )

    print("=" * 70)
    print(backend.summary_title)
    print(f"Episodes:        {len(episode_rows)}")
    print(f"ASR-used prompts:{asr_used}")
    print(f"Fallback prompts:{len(episode_rows) - asr_used}")
    print(f"Unique tasks:    {rewrite_summary['num_tasks']}")
    print(f"ASR errors:      {errors}")
    print(f"Output dataset:  {output_dir if not args.dry_run else '(dry-run)'}")
    print("=" * 70)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser, *, output_suffix: str) -> None:
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input LeRobot dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <input_dir>{output_suffix}.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete and recreate --output-dir if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run ASR and print a summary without copying or modifying files.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="auto",
        help="ASR device, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--transcribe-full-audio-when-no-vad",
        action="store_true",
        help="If an episode has no VAD segments, transcribe the full episode WAV.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a LeRobot audio dataset and replace prompts with ASR text."
    )
    subparsers = parser.add_subparsers(dest="backend", required=True)

    whisper = subparsers.add_parser(
        "whisper",
        description="Copy a LeRobot audio dataset and replace prompts with Whisper ASR text.",
        help="Rewrite prompts with Whisper ASR (openai/whisper-*).",
    )
    _add_common_args(whisper, output_suffix=WhisperBackend.output_suffix)
    whisper.add_argument(
        "--asr-model",
        type=str,
        default="openai/whisper-base",
        help="Whisper ASR model name or local model path.",
    )
    whisper.add_argument(
        "--asr-language",
        type=str,
        default="chinese",
        help="Optional Whisper language, e.g. english or chinese.",
    )
    whisper.add_argument(
        "--asr-max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens generated by Whisper.",
    )
    whisper.set_defaults(backend_factory=WhisperBackend)

    qwen = subparsers.add_parser(
        "qwen",
        description="Copy a LeRobot audio dataset and replace prompts with Qwen3-ASR text.",
        help="Rewrite prompts with Qwen3-ASR (Qwen/Qwen3-ASR-*).",
    )
    _add_common_args(qwen, output_suffix=QwenBackend.output_suffix)
    qwen.add_argument(
        "--qwen3-asr-repo",
        type=Path,
        default=None,
        help=(
            "Optional local Qwen3-ASR checkout. Used only if `qwen_asr` is not "
            "already importable."
        ),
    )
    qwen.add_argument(
        "--asr-model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen3-ASR model name or local model path.",
    )
    qwen.add_argument(
        "--asr-dtype",
        type=str,
        default="auto",
        help="Qwen3-ASR torch dtype: auto, bfloat16, float16, or float32.",
    )
    qwen.add_argument(
        "--asr-language",
        type=str,
        default="Chinese",
        help="Optional Qwen language, e.g. Chinese, English, Cantonese, or auto.",
    )
    qwen.add_argument(
        "--asr-context",
        type=str,
        default="",
        help="Optional context text passed to Qwen3-ASR.",
    )
    qwen.add_argument(
        "--asr-max-inference-batch-size",
        type=int,
        default=32,
        help="Qwen3-ASR inference batch size limit. Use -1 for unlimited.",
    )
    qwen.add_argument(
        "--asr-max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens generated by Qwen3-ASR.",
    )
    qwen.add_argument(
        "--asr-attn-implementation",
        type=str,
        default=None,
        help="Optional attention implementation, e.g. flash_attention_2.",
    )
    qwen.set_defaults(backend_factory=QwenBackend)

    return parser


def main(argv: list[str] | None = None) -> int:
    return convert_dataset(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
