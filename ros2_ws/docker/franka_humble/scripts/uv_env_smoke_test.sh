#!/usr/bin/env bash
# Confirm the container uses the baked uv venv, not data_collect/.venv.
set -euo pipefail

python_bin="$(command -v python)"
uv_bin="$(command -v uv)"

echo "python=${python_bin}"
echo "uv=${uv_bin}"
python - <<'PY'
import os
import sys
from pathlib import Path

print(f"sys.executable={sys.executable}")
print(f"sys.prefix={sys.prefix}")
print(f"sys.base_prefix={sys.base_prefix}")
print(f"sys.version={sys.version.split()[0]}")
print(f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '')}")
print(f"UV_PROJECT={os.environ.get('UV_PROJECT', '')}")
print(f"UV_PROJECT_ENVIRONMENT={os.environ.get('UV_PROJECT_ENVIRONMENT', '')}")

if sys.prefix != "/opt/uv/venv":
    raise SystemExit(f"expected sys.prefix /opt/uv/venv, got {sys.prefix}")
if sys.prefix == sys.base_prefix:
    raise SystemExit("python is not running inside a virtualenv")
if "data_collect" in sys.prefix:
    raise SystemExit("python is coming from the mounted repository")

cfg = Path("/opt/uv/venv/pyvenv.cfg").read_text(encoding="utf-8")
if "include-system-site-packages = true" not in cfg:
    raise SystemExit("venv is missing system site-packages; rclpy would be hidden")

import rclpy
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

print(f"rclpy={rclpy.__file__}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"LeRobotDataset={LeRobotDataset.__module__}")
if torch.cuda.is_available() or "+cpu" not in torch.__version__:
    raise SystemExit("expected CPU-only PyTorch in the collection container")
print("PASS: uv venv can import Humble rclpy and CPU-only LeRobot")
PY

uv --version
uv pip show rclpy >/dev/null 2>&1 || true
echo "PASS: uv environment smoke test completed."
