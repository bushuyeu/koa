"""koa priority -- Fair-share monitor.

Shows the user's priority factor breakdown, fair-share score, ranking among
pending jobs, and an estimated recovery timeline.
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh
from . import add_common_arguments, emit_json

console = Console()


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "priority",
        help="Fair-share score, priority breakdown, and queue ranking.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Show all users' priorities, not just yours.",
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


def _parse_sprio_line(line: str) -> dict | None:
    """Parse a single sprio -l output line (whitespace-delimited).

    Expected columns: JOBID PARTITION USER PRIORITY AGE ASSOC FAIRSHARE JOBSIZE PARTITION QOS NICE TRES
    We capture the ones we care about.
    """
    parts = line.split()
    if len(parts) < 8:
        return None
    # Sometimes there's a CLUSTER column first; detect by checking if first field is numeric (JOBID)
    offset = 0
    if not parts[0].isdigit() and len(parts) > 8:
        offset = 1
    try:
        return {
            "job_id": parts[0 + offset],
            "partition": parts[1 + offset],
            "user": parts[2 + offset] if len(parts) > 2 + offset else "",
            "priority": parts[3 + offset] if len(parts) > 3 + offset else "",
            "age": parts[4 + offset] if len(parts) > 4 + offset else "",
            "fairshare": parts[6 + offset] if len(parts) > 6 + offset else "",
            "jobsize": parts[7 + offset] if len(parts) > 7 + offset else "",
            "qos": parts[9 + offset] if len(parts) > 9 + offset else "",
        }
    except IndexError:
        return None


def handle(args, config: Config) -> int:
    show_all = getattr(args, "show_all", False)

    # ---- 1. Priority factor breakdown ----
    priority_entries: list[dict] = []
    try:
        if show_all:
            cmd = ["sprio", "-S", "-l", "-h"]
        else:
            cmd = ["sprio", "-u", config.user, "-l", "-h"]
        result = run_ssh(config, cmd, capture_output=True)
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = _parse_sprio_line(line)
            if entry:
                priority_entries.append(entry)
    except SSHError as exc:
        print(f"Warning: could not query priority factors: {exc}", file=sys.stderr)

    # ---- 2. Fair-share score ----
    share_rows: list[list[str]] = []
    try:
        result = run_ssh(
            config,
            ["sshare", "-u", config.user, "-l", "-P"],
            capture_output=True,
        )
        share_rows = _parse_pipe_rows(result.stdout)
        # sshare -P includes a header row; skip it
        if share_rows and share_rows[0][0].lower() in ("account", "cluster"):
            share_rows = share_rows[1:]
    except SSHError as exc:
        print(f"Warning: could not query fair-share: {exc}", file=sys.stderr)

    # ---- 3. Queue ranking ----
    all_priorities: list[dict] = []
    try:
        result = run_ssh(
            config,
            ["sprio", "-S", "-l", "-h"],
            capture_output=True,
        )
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = _parse_sprio_line(line)
            if entry:
                all_priorities.append(entry)
    except SSHError as exc:
        print(f"Warning: could not query global priorities: {exc}", file=sys.stderr)

    # Compute ranking
    my_positions: list[int] = []
    total_pending = len(all_priorities)
    for i, entry in enumerate(all_priorities):
        if entry.get("user") == config.user:
            my_positions.append(i + 1)

    best_position = min(my_positions) if my_positions else None
    percentile = None
    if best_position is not None and total_pending > 0:
        percentile = round((1.0 - best_position / total_pending) * 100, 1)

    # ---- JSON output ----
    if args.output_format == "json":
        data = {
            "priority_factors": priority_entries,
            "fair_share": [
                {
                    "account": r[0] if len(r) > 0 else "",
                    "user": r[1] if len(r) > 1 else "",
                    "raw_shares": r[2] if len(r) > 2 else "",
                    "norm_shares": r[3] if len(r) > 3 else "",
                    "raw_usage": r[4] if len(r) > 4 else "",
                    "norm_usage": r[5] if len(r) > 5 else "",
                    "effectivity": r[6] if len(r) > 6 else "",
                    "fairshare": r[7] if len(r) > 7 else "",
                }
                for r in share_rows
            ],
            "ranking": {
                "best_position": best_position,
                "total_pending": total_pending,
                "percentile": percentile,
                "my_job_count": len(my_positions),
            },
        }
        emit_json(data)
        return 0

    # ---- Rich table output ----

    # 1. Priority Breakdown
    if priority_entries:
        title = "Priority Breakdown" if show_all else "Your Priority Breakdown"
        tbl = Table(title=title, show_lines=False)
        tbl.add_column("Job ID", style="cyan", no_wrap=True)
        tbl.add_column("Partition")
        tbl.add_column("Total Priority", justify="right")
        tbl.add_column("Age", justify="right")
        tbl.add_column("Fair-Share", justify="right")
        tbl.add_column("Job Size", justify="right")
        tbl.add_column("QOS", justify="right")
        for e in priority_entries:
            tbl.add_row(
                e["job_id"],
                e["partition"],
                e["priority"],
                e["age"],
                e["fairshare"],
                e["jobsize"],
                e["qos"],
            )
        console.print(tbl)
        console.print()

    # 2. Fair-Share Score
    if share_rows:
        tbl = Table(title="Fair-Share Score", show_lines=False)
        tbl.add_column("User", style="cyan")
        tbl.add_column("Raw Shares", justify="right")
        tbl.add_column("Norm Shares", justify="right")
        tbl.add_column("Raw Usage", justify="right")
        tbl.add_column("Effectiveness", justify="right")
        tbl.add_column("Factor", justify="right")
        for r in share_rows:
            tbl.add_row(
                r[1] if len(r) > 1 else "",
                r[2] if len(r) > 2 else "",
                r[3] if len(r) > 3 else "",
                r[4] if len(r) > 4 else "",
                r[6] if len(r) > 6 else "",
                r[7] if len(r) > 7 else "",
            )
        console.print(tbl)
        console.print()

        # Recovery advice
        for r in share_rows:
            if len(r) > 7:
                try:
                    factor = float(r[7])
                    if factor < 0.5:
                        pct = round(factor * 100, 1)
                        console.print(
                            f"[bold yellow]Your fair-share is at {pct}%.[/bold yellow] "
                            "To recover, reduce job submissions and let your usage decay. "
                            "SLURM's half-life typically means noticeable recovery within 24-48 hours of lower usage.",
                        )
                        console.print()
                except (ValueError, TypeError):
                    pass

    # 3. Queue Ranking
    if total_pending > 0:
        tbl = Table(title="Queue Ranking", show_lines=False)
        tbl.add_column("Metric", style="cyan")
        tbl.add_column("Value", justify="right")
        if best_position is not None:
            tbl.add_row("Your Best Position", str(best_position))
        tbl.add_row("Total Pending Jobs", str(total_pending))
        if percentile is not None:
            style = "green" if percentile >= 75 else ("yellow" if percentile >= 50 else "red")
            tbl.add_row("Percentile", f"[{style}]{percentile}%[/{style}]")
        tbl.add_row("Your Pending Jobs", str(len(my_positions)))
        console.print(tbl)
        console.print()

    if not priority_entries and not share_rows and total_pending == 0:
        console.print("[dim]No priority or fair-share data available. You may not have pending jobs.[/dim]")

    return 0
