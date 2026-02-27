"""KOA CLI command modules.

Each command module exposes:
    register_parser(subparsers) -> argparse.ArgumentParser
    handle(args, config) -> int
"""
from __future__ import annotations

import argparse
import json as _json
import sys
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table

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


def print_gpu_selection(
    config: Config,
    gpu_type: str,
    gpus: int,
    partition: Optional[str] = None,
    *,
    console: Optional[Console] = None,
) -> None:
    """Print the selected GPU with queue info and an alternatives table.

    Shared by ``koa submit``, ``koa jupyter``, and other allocation commands.
    """
    from ..slurm import (
        GPU_PRIORITY,
        GPU_VRAM_GB,
        get_max_gpus_per_node,
        get_pending_gpu_counts,
    )

    console = console or Console()
    pending = get_pending_gpu_counts(config, partition)
    max_per_node = get_max_gpus_per_node(config, partition)
    pending_count = pending.get(gpu_type, 0)
    vram = GPU_VRAM_GB.get(gpu_type, 0)
    queue_tag = (
        "[green](no queue)[/green]"
        if pending_count == 0
        else f"[yellow]({pending_count} pending "
        f"job{'s' if pending_count != 1 else ''})[/yellow]"
    )
    console.print(
        f"Selected GPU: [bold cyan]{gpu_type}[/bold cyan] "
        f"({vram}GB VRAM) {queue_tag}"
    )

    # Build alternatives table
    all_types = set(pending.keys()) | set(max_per_node.keys())
    if not all_types:
        return

    table = Table(title="Available GPUs", show_lines=False, pad_edge=False)
    table.add_column("GPU", style="cyan")
    table.add_column("VRAM", justify="right")
    table.add_column("Max/Node", justify="right")
    table.add_column("Queue", justify="right")
    table.add_column("Score", justify="right")
    for g in sorted(
        all_types,
        key=lambda g: GPU_PRIORITY.get(g, 0) / (1 + pending.get(g, 0)),
        reverse=True,
    ):
        g_vram = GPU_VRAM_GB.get(g, 0)
        g_max = max_per_node.get(g, 0)
        g_pending = pending.get(g, 0)
        g_score = GPU_PRIORITY.get(g, 0) / (1 + g_pending)
        marker = " [bold green]*[/bold green]" if g == gpu_type else ""
        fit_warn = "" if g_max >= gpus else " [red](< --gpus)[/red]"
        table.add_row(
            f"{g}{marker}",
            f"{g_vram}GB",
            f"{g_max}{fit_warn}",
            f"{g_pending}" if g_pending else "[green]0[/green]",
            f"{g_score:.0f}",
        )
    console.print(table)


def gpu_selection_json(
    config: Config,
    gpu_type: str,
    gpus: int,
    partition: Optional[str] = None,
) -> Dict[str, Any]:
    """Return GPU selection details as a dict for JSON output (MCP / --format json)."""
    from ..slurm import (
        GPU_PRIORITY,
        GPU_VRAM_GB,
        get_max_gpus_per_node,
        get_pending_gpu_counts,
    )

    pending = get_pending_gpu_counts(config, partition)
    max_per_node = get_max_gpus_per_node(config, partition)

    alternatives = []
    for g in sorted(
        set(pending.keys()) | set(max_per_node.keys()),
        key=lambda g: GPU_PRIORITY.get(g, 0) / (1 + pending.get(g, 0)),
        reverse=True,
    ):
        alternatives.append({
            "gpu_type": g,
            "vram_gb": GPU_VRAM_GB.get(g, 0),
            "max_per_node": max_per_node.get(g, 0),
            "pending_jobs": pending.get(g, 0),
            "score": round(GPU_PRIORITY.get(g, 0) / (1 + pending.get(g, 0)), 1),
            "fits_request": max_per_node.get(g, 0) >= gpus,
        })

    return {
        "selected_gpu": gpu_type,
        "vram_gb": GPU_VRAM_GB.get(gpu_type, 0),
        "pending_jobs": pending.get(gpu_type, 0),
        "alternatives": alternatives,
    }


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
