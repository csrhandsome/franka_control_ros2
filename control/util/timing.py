"""Monotonic loop pacing shared by workflow code; no ROS dependency."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


class control_loop:
    """Pace iterations while keep_running() is true."""

    def __init__(self, frequency: float, *, keep_running: Callable[[], bool]) -> None:
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency must be positive and finite")
        self.period = 1.0 / frequency
        self.keep_running = keep_running
        self.running = False

    def __enter__(self):
        self.running = True
        self.last = time.monotonic()
        return self

    def ok(self) -> bool:
        if not self.running or not self.keep_running():
            return False
        remaining = self.period - (time.monotonic() - self.last)
        if remaining > 0:
            time.sleep(remaining)
        self.last = time.monotonic()
        return self.running and self.keep_running()

    def __exit__(self, *_exc) -> None:
        self.running = False
