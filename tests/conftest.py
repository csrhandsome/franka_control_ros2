"""Shared pytest configuration for the repository tests.

The dataset tooling tests import `lerobot`, which only the host Python 3.11
environment installs. The Humble container runs the ROS and policy entry points
from a smaller venv, so those modules are skipped there instead of failing
collection.
"""

from __future__ import annotations

import importlib.util

_LEROBOT_TESTS = ("test_episode_edit.py", "test_multirate.py")

if importlib.util.find_spec("lerobot") is None:  # pragma: no cover - container only
    collect_ignore = list(_LEROBOT_TESTS)
