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
        get_available_gpus,
        get_free_gpu_counts,
        get_max_gpus_per_node,
        get_pending_gpu_counts,
    )

    console = console or Console()
    pending = get_pending_gpu_counts(config, partition)
    max_per_node = get_max_gpus_per_node(config, partition)
    total_gpus = get_available_gpus(config, partition)
    free_gpus = get_free_gpu_counts(config, partition)

    vram = GPU_VRAM_GB.get(gpu_type, 0)
    free_count = free_gpus.get(gpu_type, 0)
    pending_count = pending.get(gpu_type, 0)

    if free_count > 0:
        status_tag = f"[green]({free_count} free)[/green]"
    elif pending_count > 0:
        status_tag = f"[yellow](busy, {pending_count} waiting)[/yellow]"
    else:
        status_tag = "[yellow](busy)[/yellow]"
    console.print(
        f"Selected GPU: [bold cyan]{gpu_type}[/bold cyan] "
        f"({vram}GB VRAM) {status_tag}"
    )

    # Build alternatives table
    all_types = set(pending.keys()) | set(max_per_node.keys())
    if not all_types:
        return

    def _contention(g: str) -> float:
        total = total_gpus.get(g, 1)
        return (pending.get(g, 0) + 1) / total

    table = Table(title="Available GPUs", show_lines=False, pad_edge=False)
    table.add_column("GPU", style="cyan")
    table.add_column("VRAM", justify="right")
    table.add_column("Free", justify="right")
    table.add_column("Waiting", justify="right")
    table.add_column("Max/Node", justify="right")
    # Sort: has free GPUs first, then lowest contention, then strongest chip
    for g in sorted(
        all_types,
        key=lambda g: (free_gpus.get(g, 0) > 0, -_contention(g), GPU_PRIORITY.get(g, 0)),
        reverse=True,
    ):
        g_vram = GPU_VRAM_GB.get(g, 0)
        g_max = max_per_node.get(g, 0)
        g_free = free_gpus.get(g, 0)
        g_pending = pending.get(g, 0)
        marker = " [bold green]*[/bold green]" if g == gpu_type else ""
        fit_warn = "" if g_max >= gpus else " [red](< --gpus)[/red]"
        free_str = (
            f"[green]{g_free}[/green]" if g_free > 0
            else "[red]0[/red]"
        )
        wait_str = (
            f"[yellow]{g_pending}[/yellow]" if g_pending > 0
            else "[dim]0[/dim]"
        )
        table.add_row(
            f"{g}{marker}",
            f"{g_vram}GB",
            free_str,
            wait_str,
            f"{g_max}{fit_warn}",
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
        get_available_gpus,
        get_free_gpu_counts,
        get_max_gpus_per_node,
        get_pending_gpu_counts,
    )

    pending = get_pending_gpu_counts(config, partition)
    max_per_node = get_max_gpus_per_node(config, partition)
    total_gpus = get_available_gpus(config, partition)
    free_gpus = get_free_gpu_counts(config, partition)

    def _contention(g: str) -> float:
        total = total_gpus.get(g, 1)
        return (pending.get(g, 0) + 1) / total

    alternatives = []
    for g in sorted(
        set(pending.keys()) | set(max_per_node.keys()),
        key=lambda g: (free_gpus.get(g, 0) > 0, -_contention(g), GPU_PRIORITY.get(g, 0)),
        reverse=True,
    ):
        alternatives.append({
            "gpu_type": g,
            "vram_gb": GPU_VRAM_GB.get(g, 0),
            "max_per_node": max_per_node.get(g, 0),
            "free_gpus": free_gpus.get(g, 0),
            "pending_jobs": pending.get(g, 0),
            "total_gpus": total_gpus.get(g, 0),
            "contention": round(_contention(g), 2),
            "fits_request": max_per_node.get(g, 0) >= gpus,
        })

    return {
        "selected_gpu": gpu_type,
        "vram_gb": GPU_VRAM_GB.get(gpu_type, 0),
        "free_gpus": free_gpus.get(gpu_type, 0),
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
