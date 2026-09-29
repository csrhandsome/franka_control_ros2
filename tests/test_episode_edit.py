"""data_analysis.episode_edit 的删除 / 重编号测试。

全部在 tmpdir 里的合成数据集上跑（见 tests/lerobot_fixture.py）—— 绝不碰仓库
里真实的 `data/`，那部分在 .gitignore 中，改坏了无法恢复。

运行：
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
（必须带那个环境变量：ROS jazzy 的 pytest 插件会因为缺 `lark` 让 collection 直接失败。）
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lerobot_fixture import build_lerobot_dataset

from data_analysis import lerobot_meta as meta
from data_analysis.episode_edit import delete_episode_by_id, delete_latest_episode


class _DatasetTestCase(unittest.TestCase):
    """公共的 fixture 构造与读取辅助。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dataset_dir = Path(self._tmp.name) / "dataset"

    def build(self, **kwargs) -> Path:
        build_lerobot_dataset(self.dataset_dir, **kwargs)
        return self.dataset_dir

    def read_json(self, *parts: str):
        return json.loads(
            (self.dataset_dir.joinpath(*parts)).read_text(encoding="utf-8")
        )

    def read_jsonl(self, *parts: str) -> list[dict]:
        text = self.dataset_dir.joinpath(*parts).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]


class DeleteLatestEpisodeTest(_DatasetTestCase):
    def test_delete_latest_episode_removes_audio_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            meta_dir = dataset_dir / "meta"
            chunk_dir = dataset_dir / "data" / "chunk-000"
            audio_dir = dataset_dir / "audio"

            meta_dir.mkdir(parents=True)
            chunk_dir.mkdir(parents=True)
            audio_dir.mkdir(parents=True)

            def write_json(path: Path, payload: dict) -> None:
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            def write_jsonl(path: Path, rows: list[dict]) -> None:
                text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
                if text:
                    text += "\n"
                path.write_text(text, encoding="utf-8")

            write_json(
                meta_dir / "info.json",
                {
                    "chunks_size": 1000,
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "total_episodes": 2,
                    "total_frames": 12,
                    "total_chunks": 1,
                    "splits": {"train": "0:2"},
                },
            )
            write_jsonl(
                meta_dir / "episodes.jsonl",
                [
                    {"episode_index": 0, "length": 5},
                    {"episode_index": 1, "length": 7},
                ],
            )
            write_jsonl(
                meta_dir / "episodes_stats.jsonl",
                [
                    {"episode_index": 0, "mean": 0.0},
                    {"episode_index": 1, "mean": 1.0},
                ],
            )

            older_episode = chunk_dir / "episode_000000.parquet"
            latest_episode = chunk_dir / "episode_000001.parquet"
            older_episode.write_bytes(b"older")
            latest_episode.write_bytes(b"latest")

            older_audio = audio_dir / "episode_000000.wav"
            latest_wav = audio_dir / "episode_000001.wav"
            latest_meta = audio_dir / "episode_000001.audio.json"
            latest_sync = audio_dir / "episode_000001.sync.json"
            older_audio.write_bytes(b"older audio")
            latest_wav.write_bytes(b"latest audio")
            latest_meta.write_text("{}", encoding="utf-8")
            latest_sync.write_text("{}", encoding="utf-8")

            exit_code = delete_latest_episode(dataset_dir)

            self.assertEqual(exit_code, 0)
            self.assertTrue(older_episode.exists())
            self.assertFalse(latest_episode.exists())
            self.assertTrue(older_audio.exists())
            self.assertFalse(latest_wav.exists())
            self.assertFalse(latest_meta.exists())
            self.assertFalse(latest_sync.exists())

            episodes_rows = [
                json.loads(line)
                for line in (meta_dir / "episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            stats_rows = [
                json.loads(line)
                for line in (meta_dir / "episodes_stats.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))

            self.assertEqual(episodes_rows, [{"episode_index": 0, "length": 5}])
            self.assertEqual(stats_rows, [{"episode_index": 0, "mean": 0.0}])
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], 5)
            self.assertEqual(info["total_chunks"], 1)
            self.assertEqual(info["splits"]["train"], "0:1")


class DeleteEpisodeByIdTest(_DatasetTestCase):
    def test_renumbers_subsequent_episodes(self) -> None:
        self.build()  # lengths 5, 7, 3, 6
        self.assertEqual(delete_episode_by_id(self.dataset_dir, 1, yes=True), 0)

        infos = meta.load_episode_infos(self.dataset_dir)
        self.assertEqual(
            [(i.episode_index, i.length) for i in infos], [(0, 5), (1, 3), (2, 6)]
        )

        info = self.read_json("meta", "info.json")
        self.assertEqual(info["total_episodes"], 3)
        self.assertEqual(info["total_frames"], 14)
        self.assertEqual(info["splits"]["train"], "0:3")

        # 后续 episode 的 parquet 文件与内部 index 列都应前移。
        self.assertFalse(
            (self.dataset_dir / "data/chunk-000/episode_000003.parquet").exists()
        )
        self.assertTrue(
            (self.dataset_dir / "data/chunk-000/episode_000002.parquet").exists()
        )

        import pyarrow.parquet as pq

        table = pq.read_table(
            self.dataset_dir / "data/chunk-000/episode_000002.parquet"
        )
        self.assertEqual(table.num_rows, 6)
        self.assertEqual(set(table.column("episode_index").to_pylist()), {2})
        # 原 episode 3 的全局起始帧 = 5 + 7 + 3 = 15，删掉 7 帧后变成 5 + 3 = 8。
        self.assertEqual(table.column("index").to_pylist()[0], 8)

        # sidecar 也应跟着改名并改内部路径。
        sync = self.read_json("audio", "episode_000002.sync.json")
        self.assertEqual(sync["episode_index"], 2)
        self.assertEqual(sync["audio_path"], "audio/episode_000002.wav")
        self.assertFalse((self.dataset_dir / "audio/episode_000003.sync.json").exists())

    def test_dry_run_changes_nothing(self) -> None:
        self.build()
        before = sorted(p.name for p in (self.dataset_dir / "data/chunk-000").iterdir())
        self.assertEqual(delete_episode_by_id(self.dataset_dir, 1, yes=False), 0)
        after = sorted(p.name for p in (self.dataset_dir / "data/chunk-000").iterdir())
        self.assertEqual(before, after)
        self.assertEqual(len(meta.load_episode_infos(self.dataset_dir)), 4)

    def test_rejects_out_of_range_index(self) -> None:
        self.build()
        self.assertEqual(delete_episode_by_id(self.dataset_dir, 99, yes=True), 2)

    def test_rejects_non_contiguous_indices(self) -> None:
        self.build()
        rows = self.read_jsonl("meta", "episodes.jsonl")
        rows[2]["episode_index"] = 7
        (self.dataset_dir / "meta/episodes.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        self.assertEqual(delete_episode_by_id(self.dataset_dir, 1, yes=True), 2)


class DeleteLatestOnFixtureTest(_DatasetTestCase):
    def test_latest_removes_max_index_and_keeps_the_rest(self) -> None:
        self.build()
        self.assertEqual(delete_latest_episode(self.dataset_dir), 0)

        infos = meta.load_episode_infos(self.dataset_dir)
        self.assertEqual([i.episode_index for i in infos], [0, 1, 2])
        self.assertEqual(self.read_json("meta", "info.json")["total_episodes"], 3)
        self.assertFalse((self.dataset_dir / "audio/episode_000003.wav").exists())
        self.assertTrue((self.dataset_dir / "audio/episode_000002.wav").exists())

    def test_empty_episodes_jsonl_is_a_no_op(self) -> None:
        self.build()
        (self.dataset_dir / "meta/episodes.jsonl").write_text("", encoding="utf-8")
        self.assertEqual(delete_latest_episode(self.dataset_dir), 0)


class LerobotMetaHelpersTest(_DatasetTestCase):
    def test_is_lerobot_dataset(self) -> None:
        self.build()
        self.assertTrue(meta.is_lerobot_dataset(self.dataset_dir))
        self.assertFalse(meta.is_lerobot_dataset(self.dataset_dir / "audio"))

    def test_episode_paths_follow_info_templates(self) -> None:
        self.build()
        info = self.read_json("meta", "info.json")
        path = meta.episode_parquet_path(info, self.dataset_dir, 3)
        self.assertEqual(
            path, self.dataset_dir / "data/chunk-000/episode_000003.parquet"
        )
        self.assertTrue(path.exists())

    def test_episode_indices_prefers_total_episodes(self) -> None:
        self.build()
        self.assertEqual(meta.episode_indices(self.dataset_dir), [0, 1, 2, 3])

    def test_episode_indices_falls_back_to_glob(self) -> None:
        self.build()
        (self.dataset_dir / "meta/info.json").unlink()
        self.assertEqual(meta.episode_indices(self.dataset_dir), [0, 1, 2, 3])

    def test_read_json_or_only_swallows_missing_files(self) -> None:
        self.build()
        missing = self.dataset_dir / "meta/nope.json"
        self.assertEqual(meta.read_json_or(missing, {"a": 1}), {"a": 1})

        # 文件存在但内容损坏时必须抛出，不能静默当成空配置。
        broken = self.dataset_dir / "meta" / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            meta.read_json_or(broken, {})

    def test_episode_stem_and_filename_roundtrip(self) -> None:
        self.assertEqual(meta.episode_stem(12), "episode_000012")
        self.assertEqual(meta.parse_episode_filename("episode_000012.sync.json"), 12)
        self.assertIsNone(meta.parse_episode_filename("not_an_episode.json"))

    def test_update_splits_in_place_only_touches_matching_end(self) -> None:
        info = {"splits": {"train": "0:4", "val": "4:4", "other": "nonsense"}}
        meta.update_splits_in_place(info, old_total=4, new_total=3)
        self.assertEqual(info["splits"]["train"], "0:3")
        self.assertEqual(info["splits"]["val"], "4:3")
        self.assertEqual(info["splits"]["other"], "nonsense")


if __name__ == "__main__":
    unittest.main()
