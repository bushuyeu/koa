"""koa watch -- Background GPU availability monitor.

Polls ``sinfo``, diffs state changes, and alerts when GPUs matching the
user's criteria become idle.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()


def _parse_gpu_inventory(
    raw: str,
    gpu_type_filter: Optional[str] = None,
    partition_filter: Optional[str] = None,
) -> list[dict]:
    """Parse sinfo -N output into a list of node GPU records.

    Each record: {"node": str, "partition": str, "gpu_type": str,
                  "gpu_count": int, "state": str}
    """
    results: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue

        node = parts[0].strip()
        partition = parts[1].strip().rstrip("*")
        gres_field = parts[2].strip()
        state = parts[3].strip().lower().rstrip("*")

        if partition_filter and partition.lower() != partition_filter.lower():
            continue

        for gres_entry in gres_field.split(","):
            gres_entry = gres_entry.strip()
            if not gres_entry.startswith("gpu:"):
                continue
            gres_parts = gres_entry.split(":")
            if len(gres_parts) < 3:
                continue
            gpu_name = gres_parts[1].lower()
            count_str = gres_parts[2].split("(")[0]
            try:
                count = int(count_str)
            except ValueError:
                continue

            if gpu_type_filter and gpu_type_filter.lower() not in gpu_name:
                continue

            results.append({
                "node": node,
                "partition": partition,
                "gpu_type": gpu_name,
                "gpu_count": count,
                "state": state,
            })

    return results


def _filter_idle(records: list[dict]) -> list[dict]:
    """Return only records where nodes are idle or mix."""
    return [r for r in records if r["state"] in ("idle", "mix", "mixed")]


def _build_status_table(
    records: list[dict],
    gpu_type: Optional[str],
    partition: Optional[str],
    new_nodes: Optional[set[str]] = None,
) -> Table:
    """Build a Rich table showing current matching GPU inventory."""
    title_parts = ["GPU Watch"]
    if gpu_type:
        title_parts.append(f"type={gpu_type}")
    if partition:
        title_parts.append(f"partition={partition}")
    title = " | ".join(title_parts)

    table = Table(title=title, show_lines=False)
    table.add_column("Node", style="bold")
    table.add_column("Partition")
    table.add_column("GPU Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("State")

    idle_records = _filter_idle(records)

    for rec in sorted(idle_records, key=lambda r: (r["gpu_type"], r["node"])):
        is_new = new_nodes and rec["node"] in new_nodes
        state_style = "green" if rec["state"] == "idle" else "yellow"
        row_style = "bold green" if is_new else ""
        marker = " NEW" if is_new else ""
        table.add_row(
            rec["node"],
            rec["partition"],
            rec["gpu_type"],
            str(rec["gpu_count"]),
            f"[{state_style}]{rec['state']}{marker}[/{state_style}]",
            style=row_style,
        )

    total_idle = sum(r["gpu_count"] for r in idle_records)
    table.caption = f"Total idle/mix GPUs matching filter: {total_idle}"
    return table


def _send_notification(message: str) -> None:
    """Send webhook notification using the notify config if available."""
    try:
        from .notify import _load_notify_config, _send_webhook

        config = _load_notify_config()
        webhooks = config.get("webhooks", [])
        for wh in webhooks:
            _send_webhook(wh["url"], wh.get("type", "slack"), message)
    except Exception:
        pass


def _poll_sinfo(config: Config, partition: Optional[str] = None) -> str:
    """Query sinfo for GPU node inventory."""
    cmd: list[str] = [
        "sinfo", "-N",
        "-o", "%N|%P|%G|%T",
        "--noheader",
    ]
    if partition:
        cmd.extend(["-p", partition])

    result = run_ssh(config, cmd, capture_output=True)
    return result.stdout or ""


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "watch",
        help="Monitor GPU availability and alert when matching GPUs become idle.",
    )
    parser.add_argument(
        "--gpu-type",
        default=None,
        help="Filter for a specific GPU type (e.g. h100, a100).",
    )
    parser.add_argument(
        "--partition", "-p",
        default=None,
        help="Filter to a specific partition.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum number of idle GPUs to trigger an alert (default: 1).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Poll interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send webhook notification when matching GPUs are found.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (no continuous monitoring).",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    gpu_type = args.gpu_type
    partition = args.partition
    min_count = max(1, args.min_count)
    interval = max(5, args.interval)
    do_notify = args.notify
    once = args.once

    # --- Single check mode ---
    if once:
        try:
            raw = _poll_sinfo(config, partition)
        except SSHError as exc:
            console.print(f"[red]SSH error:[/red] {exc}", file=sys.stderr)
            return 1

        records = _parse_gpu_inventory(raw, gpu_type, partition)
        idle = _filter_idle(records)
        total_idle = sum(r["gpu_count"] for r in idle)

        if args.output_format == "json":
            emit_json({
                "gpu_type_filter": gpu_type,
                "partition_filter": partition,
                "min_count": min_count,
                "total_idle_gpus": total_idle,
                "matching_nodes": idle,
                "threshold_met": total_idle >= min_count,
            })
            return 0

        table = _build_status_table(records, gpu_type, partition)
        console.print(table)

        if total_idle >= min_count:
            console.print(
                f"\n[bold green]Threshold met:[/bold green] "
                f"{total_idle} idle GPU(s) >= {min_count} required."
            )
        else:
            console.print(
                f"\n[bold yellow]Below threshold:[/bold yellow] "
                f"{total_idle} idle GPU(s) < {min_count} required."
            )

        return 0

    # --- Continuous monitoring mode ---
    watch_desc_parts = []
    if gpu_type:
        watch_desc_parts.append(gpu_type)
    else:
        watch_desc_parts.append("any")
    watch_desc_parts.append("GPUs")
    if partition:
        watch_desc_parts.append(f"on {partition}")

    console.print(
        f"[bold]Watching for {' '.join(watch_desc_parts)}... "
        f"(Ctrl+C to stop, polling every {interval}s)[/bold]\n"
    )

    previous_idle_nodes: set[str] = set()

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                try:
                    raw = _poll_sinfo(config, partition)
                except SSHError:
                    live.update("[yellow]SSH poll failed, retrying...[/yellow]")
                    time.sleep(interval)
                    continue

                records = _parse_gpu_inventory(raw, gpu_type, partition)
                idle = _filter_idle(records)
                current_idle_nodes = {r["node"] for r in idle}
                total_idle = sum(r["gpu_count"] for r in idle)

                # Detect newly idle nodes
                new_nodes = current_idle_nodes - previous_idle_nodes

                if new_nodes:
                    for rec in idle:
                        if rec["node"] in new_nodes:
                            msg = (
                                f"NEW: {rec['node']} has "
                                f"{rec['gpu_count']}x {rec['gpu_type']} IDLE"
                            )
                            console.print(f"[bold green]{msg}[/bold green]")
                            if do_notify:
                                _send_notification(
                                    f"KOA GPU Alert: {rec['node']} has "
                                    f"{rec['gpu_count']}x {rec['gpu_type']} idle"
                                )

                if total_idle >= min_count and new_nodes:
                    console.print(
                        f"[bold green]Threshold met:[/bold green] "
                        f"{total_idle} idle GPU(s) available."
                    )

                table = _build_status_table(records, gpu_type, partition, new_nodes)
                live.update(table)

                previous_idle_nodes = current_idle_nodes
                time.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[bold]Stopped watching.[/bold]")

    return 0
