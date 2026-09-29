"""Send transitions / episode stats to a GPU-side replay sink."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import websockets.exceptions
import websockets.sync.client

from control.util import msgpack_numpy


class TransitionClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8002) -> None:
        if host.startswith(("ws://", "wss://")):
            self._uri = host
        else:
            self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self.server_metadata = self._wait_for_server()

    def _connect(
        self,
    ) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        try:
            conn = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=10,
                ping_timeout=300,
            )
        except TypeError:
            conn = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None
            )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def _wait_for_server(
        self, timeout: float | None = None
    ) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        logging.info("[HITL] Waiting for transition sink at %s", self._uri)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                conn, metadata = self._connect()
                logging.info("[HITL] Connected to transition sink %s", self._uri)
                return conn, metadata
            except (ConnectionRefusedError, OSError, TimeoutError):
                if deadline is not None and time.monotonic() >= deadline:
                    raise ConnectionRefusedError(
                        f"Timed out waiting for transition sink at {self._uri}"
                    ) from None
                logging.info("[HITL] Still waiting for %s", self._uri)
                time.sleep(1.0)

    def send(self, message: dict[str, Any]) -> dict[str, Any]:
        data = self._packer.pack(message)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosed:
            self._ws, self.server_metadata = self._wait_for_server(timeout=120)
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Transition sink error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


class AsyncTransitionSink:
    """Serialize replay traffic without waiting for the server on a servo tick."""

    def __init__(self, client: TransitionClient, max_pending: int = 300) -> None:
        self._client = client
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="transition-sink", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            kind, payload = self._queue.get()
            try:
                if kind == "stop":
                    return
                if kind == "message" and self._error is None:
                    self._client.send(payload)
                if kind == "barrier":
                    payload.set()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
                if kind == "barrier":
                    payload.set()
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Transition sink failed") from self._error

    def send(self, message: dict[str, Any]) -> None:
        self._raise_if_failed()
        try:
            self._queue.put_nowait(("message", message))
        except queue.Full as exc:
            raise RuntimeError("Transition sink fell behind the robot loop") from exc

    def drain(self) -> None:
        event = threading.Event()
        self._queue.put(("barrier", event))
        event.wait()
        self._raise_if_failed()

    def close(self) -> None:
        try:
            self._queue.put(("stop", None), timeout=1.0)
        except queue.Full:
            logging.error("[HITL] transition queue did not drain before shutdown")
        self._thread.join(timeout=2.0)
        self._client.close()
        if self._thread.is_alive():
            raise RuntimeError("Transition sink did not stop within 2 seconds")
        self._raise_if_failed()
