#!/usr/bin/env python3
"""Run policy inference with optional VR intervention and branch recording.

Runs in the Humble container for real/fake hardware, or ``--dry-run`` with a
compatible policy WebSocket server.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from control.hitl.collector import HumanInTheLoopCollector
from control.robot_config import config_path, load_mapping


def run_loop(
    *,
    config: dict,
    dry_run: bool,
    max_episodes: int = 0,
    max_steps: int = 0,
) -> None:
    loop = HumanInTheLoopCollector.from_config(
        config,
        dry_run=dry_run,
        max_episodes=max_episodes,
        max_steps=max_steps,
    )
    loop.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Human in the Loop inference and recording"
    )
    parser.add_argument(
        "--robot",
        default="franka",
        help="Robot yaml name under config/hitl/, default franka",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override config path. Default: config/hitl/<robot>.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dummy arm/cameras; still talks to the GPU websocket servers.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_file = config_path(stage="hitl", robot=args.robot, explicit=args.config)
    config = load_mapping(config_file)
    logging.info("[HIL] config=%s", config_file)
    run_loop(
        config=config,
        dry_run=bool(args.dry_run),
        max_episodes=int(args.max_episodes),
        max_steps=int(args.max_steps),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
