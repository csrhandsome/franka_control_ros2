"""Reactive Desk websocket publisher for robot EE positions."""

from __future__ import annotations

import json
import threading

import numpy as np
import websockets
import websockets.sync.client


class ReactiveDeskVlaClient:
    """Minimal Reactive Desk websocket client compatible with the openpi sender."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/ws/VlaIngest",
        *,
        enabled: bool = True,
    ) -> None:
        self._uri = self._build_ws_uri(host, port, path)
        self._enabled = enabled
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest_xy: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._prompt = ""

    def start_publishing(self, prompt: str) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._prompt = prompt
        self._thread = threading.Thread(
            target=self._publish_worker, name="reactive-desk", daemon=True
        )
        self._thread.start()

    def publish_xy(self, ee_position: np.ndarray) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._latest_xy = np.asarray(ee_position[:2], dtype=np.float32).copy()

    def _publish_worker(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                point = self._latest_xy
            if point is not None:
                self.send_predictions(
                    point.reshape(1, 2),
                    np.ones(1, dtype=np.float32),
                    prompt=self._prompt,
                    is_executing=True,
                )
            self._stop.wait(1.0 / 30.0)

    def connect(self) -> None:
        if not self._enabled or self._ws is not None:
            return

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
        )
        self._ws.recv()

    def send_predictions(
        self,
        xyz: np.ndarray,
        probabilities: np.ndarray,
        *,
        prompt: str = "",
        is_executing: bool = True,
    ) -> bool:
        if not self._enabled:
            return False

        xyz = np.asarray(xyz, dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        predictions = [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if point.shape[0] > 2 else 0.0,
                "probability": float(probabilities[rank])
                if rank < probabilities.shape[0]
                else 1.0,
                "rank": int(rank),
            }
            for rank, point in enumerate(xyz)
        ]
        payload = {
            "type": "vla_predictions",
            "predictions": predictions,
            "is_executing": is_executing,
            "current_prompt": prompt,
        }

        try:
            self.connect()
            if self._ws is None:
                return False
            self._ws.send(json.dumps(payload))
            self._ws.recv()
            return True
        except websockets.ConnectionClosed:
            self._ws = None
            return False
        except OSError:
            self._ws = None
            return False

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._ws is None:
            return
        self._ws.close()
        self._ws = None

    @staticmethod
    def _build_ws_uri(host: str, port: int, path: str) -> str:
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        if path:
            path = path if path.startswith("/") else f"/{path}"
            if not uri.endswith(path):
                uri = f"{uri.rstrip('/')}{path}"
        return uri
