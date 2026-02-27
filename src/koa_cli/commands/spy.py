"""koa spy -- Queue intelligence.

Analyses queue depth per partition, identifies next GPUs to free, computes
historical wait times, and produces a partition comparison table.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh
from . import add_common_arguments, emit_json

console = Console()


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "spy",
        help="Queue intelligence: depth, next GPUs freeing, wait history.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--partition", "-p",
        default=None,
        help="Specific partition to analyse.",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of entries to show in ranked lists (default: 10).",
    )
    return parser


def _parse_pipe_rows(raw: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([c.strip() for c in line.split("|")])
    return rows


def _parse_slurm_time(t: str) -> timedelta | None:
    """Parse SLURM time strings: D-HH:MM:SS, HH:MM:SS, MM:SS, or UNLIMITED."""
    if not t or t in ("UNLIMITED", "INVALID", "Partition_Limit", "NONE"):
        return None
    t = t.strip()
    days = 0
    if "-" in t:
        d_part, t = t.split("-", 1)
        try:
            days = int(d_part)
        except ValueError:
            return None
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return timedelta(days=days, hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
        if len(parts) == 2:
            return timedelta(days=days, minutes=int(parts[0]), seconds=int(parts[1]))
        if len(parts) == 1:
            return timedelta(days=days, seconds=int(parts[0]))
    except ValueError:
        return None
    return None


def _format_timedelta(td: timedelta | None) -> str:
    if td is None:
        return "-"
    total_secs = int(td.total_seconds())
    if total_secs < 0:
        return "0:00:00"
    hours, remainder = divmod(total_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _time_left(end_str: str) -> str:
    """Compute how much time is left from an end-time string."""
    try:
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%S")
        delta = end_dt - datetime.now()
        if delta.total_seconds() < 0:
            return "ending"
        return _format_timedelta(delta)
    except (ValueError, TypeError):
        return "-"


def _parse_datetime(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def handle(args, config: Config) -> int:
    partition_filter = getattr(args, "partition", None)
    top_n = max(1, getattr(args, "top", 10))

    # ---- 1. Queue depth per partition ----
    queue_depth: dict[str, dict[str, int]] = {}
    try:
        result = run_ssh(
            config,
            ["squeue", "-h", "-o", "%P|%T"],
            capture_output=True,
        )
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 2:
                continue
            part = row[0].rstrip("*")
            state = row[1]
            if partition_filter and part != partition_filter:
                continue
            queue_depth.setdefault(part, {"RUNNING": 0, "PENDING": 0})
            if state == "RUNNING":
                queue_depth[part]["RUNNING"] += 1
            elif state == "PENDING":
                queue_depth[part]["PENDING"] += 1
    except SSHError as exc:
        print(f"Warning: could not query queue: {exc}", file=sys.stderr)

    # ---- 2. Next GPUs to free ----
    next_free: list[dict] = []
    try:
        result = run_ssh(
            config,
            ["squeue", "-t", "RUNNING", "--sort=e", "-o", "%N|%b|%e|%u|%j|%P", "-h"],
            capture_output=True,
        )
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 6:
                continue
            part = row[5].rstrip("*")
            if partition_filter and part != partition_filter:
                continue
            if not row[1] or "gpu" not in row[1].lower():
                continue
            next_free.append({
                "node": row[0],
                "gpus": row[1],
                "end_time": row[2],
                "time_left": _time_left(row[2]),
                "user": row[3],
                "job_name": row[4],
                "partition": part,
            })
    except SSHError as exc:
        print(f"Warning: could not query running jobs: {exc}", file=sys.stderr)

    # ---- 3. Historical wait times ----
    wait_stats: dict[str, list[float]] = {}
    try:
        result = run_ssh(
            config,
            [
                "sacct", "-u", config.user,
                "--format=JobID,Submit,Start,Partition",
                "-P", "-n",
                "--starttime=now-7days",
            ],
            capture_output=True,
        )
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 4:
                continue
            job_id = row[0]
            # Skip sub-steps like "12345.batch"
            if "." in job_id:
                continue
            submit_dt = _parse_datetime(row[1])
            start_dt = _parse_datetime(row[2])
            part = row[3].rstrip("*")
            if partition_filter and part != partition_filter:
                continue
            if submit_dt and start_dt and start_dt >= submit_dt:
                wait_secs = (start_dt - submit_dt).total_seconds()
                wait_stats.setdefault(part, []).append(wait_secs)
    except SSHError as exc:
        print(f"Warning: could not query job history: {exc}", file=sys.stderr)

    # ---- 4. Partition overview ----
    partition_info: list[dict] = []
    try:
        result = run_ssh(
            config,
            ["sinfo", "-o", "%P|%a|%l|%D|%G|%C", "--noheader"],
            capture_output=True,
        )
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 6:
                continue
            part = row[0].rstrip("*")
            if partition_filter and part != partition_filter:
                continue
            partition_info.append({
                "partition": part,
                "state": row[1],
                "max_wall": row[2],
                "nodes": row[3],
                "gpus": row[4],
                "cpus": row[5],
            })
    except SSHError as exc:
        print(f"Warning: could not query partition info: {exc}", file=sys.stderr)

    # ---- JSON output ----
    if args.output_format == "json":
        computed_waits: dict[str, dict] = {}
        for part, vals in wait_stats.items():
            vals_sorted = sorted(vals)
            computed_waits[part] = {
                "avg_seconds": sum(vals) / len(vals) if vals else 0,
                "median_seconds": _median(vals),
                "min_seconds": min(vals) if vals else 0,
                "max_seconds": max(vals) if vals else 0,
                "sample_size": len(vals),
            }
        data = {
            "queue_depth": queue_depth,
            "next_gpus_to_free": next_free[:top_n],
            "historical_wait_times": computed_waits,
            "partition_overview": partition_info,
        }
        emit_json(data)
        return 0

    # ---- Rich table output ----

    # 1. Queue Depth
    if queue_depth:
        tbl = Table(title="Queue Depth by Partition", show_lines=False)
        tbl.add_column("Partition", style="cyan")
        tbl.add_column("Pending", justify="right", style="yellow")
        tbl.add_column("Running", justify="right", style="green")
        tbl.add_column("Total", justify="right")
        for part in sorted(queue_depth):
            d = queue_depth[part]
            total = d["RUNNING"] + d["PENDING"]
            tbl.add_row(part, str(d["PENDING"]), str(d["RUNNING"]), str(total))
        console.print(tbl)
        console.print()

    # 2. Next GPUs to Free
    if next_free:
        tbl = Table(title=f"Next GPUs to Free (top {top_n})", show_lines=False)
        tbl.add_column("Node", style="cyan")
        tbl.add_column("GPUs")
        tbl.add_column("End Time")
        tbl.add_column("Time Left", justify="right")
        tbl.add_column("User")
        tbl.add_column("Job Name")
        tbl.add_column("Partition")
        for entry in next_free[:top_n]:
            tbl.add_row(
                entry["node"],
                entry["gpus"],
                entry["end_time"],
                entry["time_left"],
                entry["user"],
                entry["job_name"],
                entry["partition"],
            )
        console.print(tbl)
        console.print()

    # 3. Historical Wait Times
    if wait_stats:
        tbl = Table(title="Historical Wait Times (last 7 days)", show_lines=False)
        tbl.add_column("Partition", style="cyan")
        tbl.add_column("Avg Wait", justify="right")
        tbl.add_column("Median Wait", justify="right")
        tbl.add_column("Min", justify="right")
        tbl.add_column("Max", justify="right")
        tbl.add_column("Samples", justify="right")
        for part in sorted(wait_stats):
            vals = wait_stats[part]
            if not vals:
                continue
            avg_td = timedelta(seconds=sum(vals) / len(vals))
            med_td = timedelta(seconds=_median(vals))
            min_td = timedelta(seconds=min(vals))
            max_td = timedelta(seconds=max(vals))
            tbl.add_row(
                part,
                _format_timedelta(avg_td),
                _format_timedelta(med_td),
                _format_timedelta(min_td),
                _format_timedelta(max_td),
                str(len(vals)),
            )
        console.print(tbl)
        console.print()

    # 4. Partition Overview
    if partition_info:
        tbl = Table(title="Partition Overview", show_lines=False)
        tbl.add_column("Partition", style="cyan")
        tbl.add_column("State")
        tbl.add_column("Max Wall", justify="right")
        tbl.add_column("Nodes", justify="right")
        tbl.add_column("GPUs")
        tbl.add_column("CPUs (A/I/O/T)", justify="right")
        for entry in partition_info:
            tbl.add_row(
                entry["partition"],
                entry["state"],
                entry["max_wall"],
                entry["nodes"],
                entry["gpus"],
                entry["cpus"],
            )
        console.print(tbl)
        console.print()

    if not queue_depth and not next_free and not partition_info:
        console.print("[dim]No queue data available.[/dim]")

    return 0
