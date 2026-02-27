"""KOA CLI command modules.

Each command module exposes:
    register_parser(subparsers) -> argparse.ArgumentParser
    handle(args, config) -> int
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from typing import Any

from ..config import Config

# ---------------------------------------------------------------------------
# Shared helpers available to all command modules
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach --config, --backend, and --format flags shared by every command."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the KOA config file.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Slurm backend name to use.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        dest="output_format",
        help="Output format (default: table).",
    )


def emit_json(data: Any) -> None:
    """Write *data* as pretty-printed JSON to stdout."""
    _json.dump(data, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Auto-discovery: import all command modules so their register/handle are
# accessible via ``commands.<name>.register_parser / handle``.
# ---------------------------------------------------------------------------

from . import (  # noqa: E402
    optimize,
    audit,
    why,
    diagnose,
    limits,
    spy,
    priority,
    efficiency,
    resubmit,
    notify,
    sweep,
    watch,
    chain,
    distributed,
    validate,
    anywhere,
    env as env_cmd,
    budget,
    jupyter,
)

ALL_COMMANDS = [
    optimize,
    audit,
    why,
    diagnose,
    validate,
    limits,
    spy,
    priority,
    efficiency,
    resubmit,
    notify,
    sweep,
    watch,
    chain,
    distributed,
    anywhere,
    env_cmd,
    budget,
    jupyter,
]
