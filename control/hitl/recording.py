"""Asynchronous, local recording of complete HITL trajectories.

Each episode is a stream of NumPy-aware MessagePack records. A stream can be
read with ``msgpack_numpy.Unpacker(open(path, 'rb'), raw=False)``.
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from control.util import msgpack_numpy


class LocalEpisodeRecorder:
    def __init__(self, output_dir: str | Path, max_pending: int = 300) -> None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(output_dir) / f"{run_name}_{uuid4().hex[:8]}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(max_pending)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._write, name="hitl-recorder", daemon=True
        )
        self._thread.start()

    def _write(self) -> None:
        stream = None
        current_episode = None
        packer = msgpack_numpy.Packer()
        events = None
        try:
            events = (self.run_dir / "events.jsonl").open("w", encoding="utf-8")
            while True:
                record = self._queue.get()
                try:
                    if record is None:
                        return
                    episode = int(record["episode_index"])
                    if episode != current_episode:
                        if stream is not None:
                            stream.close()
                        stream = (self.run_dir / f"episode_{episode:06d}.msgpack").open(
                            "wb"
                        )
                        current_episode = episode
                    stream.write(packer.pack(record))
                    stream.flush()
                    if record["type"] in {
                        "episode_start",
                        "branch_start",
                        "branch_release",
                        "episode_stats",
                    }:
                        summary = {
                            key: value.tolist()
                            if isinstance(value, np.ndarray)
                            else value
                            for key, value in record.items()
                            if key not in {"fork_state", "resume_state"}
                        }
                        events.write(json.dumps(summary, ensure_ascii=False) + "\n")
                        events.flush()
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._error = exc
            # Unblock a producer waiting for the queue to drain.
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
        finally:
            if stream is not None:
                stream.close()
            if events is not None:
                events.close()

    def record(self, event: dict[str, Any]) -> None:
        self._raise_if_failed()
        try:
            self._queue.put_nowait(event)
        except queue.Full as exc:
            raise RuntimeError("HITL recorder fell behind the robot loop") from exc

    def drain(self) -> None:
        self._queue.join()
        self._raise_if_failed()

    def close(self) -> None:
        self.drain()
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("HITL recorder did not stop")
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("HITL recorder failed") from self._error
