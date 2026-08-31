"""LiMOSAT command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .run import LiMOSATRun
from .store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="limosat")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="process all catalogue components")
    run.add_argument("config", type=Path)
    status = commands.add_parser("status", help="show durable local run state")
    status.add_argument("config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config)
    if arguments.command == "run":
        result = LiMOSATRun(config).execute(["limosat", "run", str(arguments.config)])
    else:
        result = RunStore(config).status()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0
