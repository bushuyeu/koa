"""koa budget -- GPU-hours allocation tracking.

Queries ``sacct`` for historical GPU-hour consumption and ``sacctmgr`` for
allocation limits.  Displays burn rate, waste breakdown, and projected
exhaustion so users can pace their GPU spending.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..config import Config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()


def _parse_slurm_time_seconds(time_str: str) -> Optional[float]:
    """Parse SLURM time formats: D-HH:MM:SS, HH:MM:SS, MM:SS."""
    if not time_str or time_str in ("", "UNLIMITED", "Partition_Limit"):
        return None
    time_str = time_str.strip()
    days = 0
    if "-" in time_str:
        day_part, time_str = time_str.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None

    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        elif len(parts) == 2:
            hours, minutes, seconds = 0, int(parts[0]), float(parts[1])
        elif len(parts) == 1:
            hours, minutes, seconds = 0, 0, float(parts[0])
        else:
            return None
    except ValueError:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_tres_gpu_count(tres_str: str) -> int:
    """Extract GPU count from AllocTRES like 'billing=8,cpu=8,gres/gpu=2,mem=64G'."""
    for entry in tres_str.split(","):
        entry = entry.strip()
        if entry.startswith("gres/gpu="):
            try:
                return int(entry.split("=")[1])
            except (ValueError, IndexError):
                pass
    return 0


def _parse_grp_tres_mins_gpu(tres_str: str) -> Optional[float]:
    """Extract GPU-minute limit from GrpTRESMins like 'cpu=100000,gres/gpu=60000'.

    Returns the limit in GPU-hours, or None if not set.
    """
    if not tres_str or not tres_str.strip():
        return None
    for entry in tres_str.split(","):
        entry = entry.strip()
        if "gpu" in entry.lower() and "=" in entry:
            try:
                minutes = float(entry.split("=")[1])
                return minutes / 60.0
            except (ValueError, IndexError):
                pass
    return None


def _state_bucket(state: str) -> str:
    """Normalize job state into a display bucket."""
    s = state.upper()
    if "COMPLETED" in s:
        return "COMPLETED"
    if "FAIL" in s or "TIMEOUT" in s:
        return "FAILED"
    if "CANCEL" in s:
        return "CANCELLED"
    if "RUNNING" in s:
        return "RUNNING"
    return "OTHER"


def _state_color(bucket: str) -> str:
    """Rich color for a state bucket."""
    return {
        "COMPLETED": "green",
        "FAILED": "red",
        "CANCELLED": "yellow",
        "RUNNING": "cyan",
    }.get(bucket, "dim")


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "budget",
        help="GPU-hours budget tracker -- burn rate, waste, and projected exhaustion.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback period in days (default: 30).",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        default=False,
        help="Show per-job breakdown table sorted by GPU-hours.",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    days = max(1, args.days)
    breakdown = args.breakdown

    # ---------------------------------------------------------------
    # 1. Query sacct for GPU-hours
    # ---------------------------------------------------------------
    sacct_cmd = (
        f"sacct -u {config.user} "
        f"--format=JobID,JobName%30,Partition,State%20,Elapsed,AllocTRES%60,Start,End "
        f"-P -n "
        f"--starttime=$(date -d '{days} days ago' +%Y-%m-%d 2>/dev/null || date -v-{days}d +%Y-%m-%d)"
    )

    try:
        result = run_ssh(
            config,
            ["bash", "-lc", sacct_cmd],
            capture_output=True,
        )
    except SSHError as exc:
        console.print(f"[red]Error querying sacct:[/red] {exc}")
        return 1

    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        console.print(f"[dim]No jobs found in the last {days} day(s).[/dim]")
        return 0

    # ---------------------------------------------------------------
    # 2. Parse jobs
    # ---------------------------------------------------------------
    jobs: list[dict] = []
    total_gpu_hours = 0.0
    by_state: dict[str, float] = {}
    by_partition: dict[str, float] = {}

    for line in lines:
        fields = line.split("|")
        if len(fields) < 8:
            continue

        job_id = fields[0].strip()
        if "." in job_id:
            continue

        job_name = fields[1].strip()
        partition = fields[2].strip()
        state_raw = fields[3].strip()
        elapsed_raw = fields[4].strip()
        alloc_tres_raw = fields[5].strip()
        start_raw = fields[6].strip()
        end_raw = fields[7].strip()

        elapsed_s = _parse_slurm_time_seconds(elapsed_raw)
        if elapsed_s is None or elapsed_s <= 0:
            continue

        gpu_count = _parse_tres_gpu_count(alloc_tres_raw)
        if gpu_count <= 0:
            continue

        gpu_hours = (elapsed_s * gpu_count) / 3600.0
        bucket = _state_bucket(state_raw)

        total_gpu_hours += gpu_hours
        by_state[bucket] = by_state.get(bucket, 0.0) + gpu_hours
        by_partition[partition] = by_partition.get(partition, 0.0) + gpu_hours

        jobs.append({
            "job_id": job_id,
            "job_name": job_name,
            "partition": partition,
            "state": bucket,
            "elapsed_s": elapsed_s,
            "gpu_count": gpu_count,
            "gpu_hours": round(gpu_hours, 2),
            "start": start_raw,
            "end": end_raw,
        })

    if not jobs:
        console.print(f"[dim]No GPU jobs found in the last {days} day(s).[/dim]")
        return 0

    # ---------------------------------------------------------------
    # 3. Query sacctmgr for allocation limit
    # ---------------------------------------------------------------
    allocation_limit: Optional[float] = None
    try:
        assoc_cmd = (
            f"sacctmgr show assoc user={config.user} "
            f"format=Account,GrpTRESMins --parsable2 --noheader"
        )
        assoc_result = run_ssh(
            config,
            ["bash", "-lc", assoc_cmd],
            capture_output=True,
        )
        for aline in assoc_result.stdout.strip().splitlines():
            aline = aline.strip()
            if not aline:
                continue
            parts = [p.strip() for p in aline.split("|")]
            if len(parts) >= 2:
                limit = _parse_grp_tres_mins_gpu(parts[1])
                if limit is not None and (allocation_limit is None or limit > allocation_limit):
                    allocation_limit = limit
    except SSHError:
        pass

    # ---------------------------------------------------------------
    # 4. Compute metrics
    # ---------------------------------------------------------------
    burn_rate = total_gpu_hours / days
    wasted_hours = by_state.get("FAILED", 0.0) + by_state.get("CANCELLED", 0.0)

    projected_exhaustion: Optional[str] = None
    days_remaining: Optional[float] = None
    if allocation_limit is not None and burn_rate > 0:
        remaining_hours = allocation_limit - total_gpu_hours
        if remaining_hours > 0:
            days_remaining = remaining_hours / burn_rate
            exhaust_date = datetime.now() + timedelta(days=days_remaining)
            projected_exhaustion = exhaust_date.strftime("%Y-%m-%d")
        else:
            days_remaining = 0.0
            projected_exhaustion = "EXHAUSTED"

    # ---------------------------------------------------------------
    # 5. JSON output
    # ---------------------------------------------------------------
    if args.output_format == "json":
        data = {
            "period_days": days,
            "total_gpu_hours": round(total_gpu_hours, 2),
            "by_state": {k: round(v, 2) for k, v in sorted(by_state.items())},
            "by_partition": {k: round(v, 2) for k, v in sorted(by_partition.items())},
            "burn_rate_per_day": round(burn_rate, 2),
            "wasted_gpu_hours": round(wasted_hours, 2),
            "allocation_limit": round(allocation_limit, 2) if allocation_limit is not None else None,
            "projected_exhaustion": projected_exhaustion,
            "days_remaining": round(days_remaining, 1) if days_remaining is not None else None,
            "jobs": jobs,
        }
        emit_json(data)
        return 0

    # ---------------------------------------------------------------
    # 6. Rich table output
    # ---------------------------------------------------------------

    # Summary panel
    summary_lines = [
        f"[bold]Period:[/bold]          last {days} day(s)",
        f"[bold]Total GPU-hours:[/bold] {total_gpu_hours:,.1f}",
        f"[bold]Burn rate:[/bold]       {burn_rate:,.1f} GPU-hours/day",
        f"[bold]Wasted:[/bold]          {wasted_hours:,.1f} GPU-hours (FAILED + CANCELLED)",
    ]
    if allocation_limit is not None:
        summary_lines.append(f"[bold]Allocation:[/bold]     {allocation_limit:,.1f} GPU-hours")
    console.print(Panel("\n".join(summary_lines), title="GPU Budget Summary", border_style="blue"))

    # Allocation progress bar (if limit exists)
    if allocation_limit is not None and allocation_limit > 0:
        usage_pct = min(total_gpu_hours / allocation_limit, 1.0)
        pct_display = usage_pct * 100

        if usage_pct >= 0.9:
            bar_color = "red"
        elif usage_pct >= 0.8:
            bar_color = "yellow"
        else:
            bar_color = "green"

        bar_text = Text()
        bar_text.append(f"  Usage: {total_gpu_hours:,.1f} / {allocation_limit:,.1f} GPU-hours ")
        bar_text.append(f"({pct_display:.1f}%)", style=f"bold {bar_color}")

        console.print(bar_text)
        bar = ProgressBar(total=allocation_limit, completed=min(total_gpu_hours, allocation_limit))
        console.print(f"  ", end="")
        console.print(bar, width=60)

        if projected_exhaustion == "EXHAUSTED":
            console.print("  [bold red]Allocation EXHAUSTED[/bold red]")
        elif projected_exhaustion:
            console.print(
                f"  Projected exhaustion: [bold]{projected_exhaustion}[/bold] "
                f"({days_remaining:.0f} days at current rate)"
            )

        if usage_pct >= 0.8:
            console.print("  [bold yellow]Warning: >80% of allocation consumed[/bold yellow]")
        console.print()

    # By-state table
    state_table = Table(title="GPU-Hours by State", show_lines=False)
    state_table.add_column("State", style="bold")
    state_table.add_column("GPU-Hours", justify="right")
    state_table.add_column("% of Total", justify="right")

    for bucket in ("COMPLETED", "RUNNING", "FAILED", "CANCELLED", "OTHER"):
        hours = by_state.get(bucket, 0.0)
        if hours <= 0:
            continue
        pct = (hours / total_gpu_hours * 100) if total_gpu_hours > 0 else 0
        color = _state_color(bucket)
        state_table.add_row(
            f"[{color}]{bucket}[/{color}]",
            f"{hours:,.1f}",
            f"{pct:.1f}%",
        )

    console.print(state_table)
    console.print()

    # By-partition table
    part_table = Table(title="GPU-Hours by Partition", show_lines=False)
    part_table.add_column("Partition", style="cyan")
    part_table.add_column("GPU-Hours", justify="right")
    part_table.add_column("% of Total", justify="right")

    for partition, hours in sorted(by_partition.items(), key=lambda x: -x[1]):
        pct = (hours / total_gpu_hours * 100) if total_gpu_hours > 0 else 0
        part_table.add_row(partition, f"{hours:,.1f}", f"{pct:.1f}%")

    console.print(part_table)

    # Per-job breakdown (--breakdown flag)
    if breakdown:
        console.print()
        job_table = Table(
            title=f"Per-Job Breakdown (top consumers)",
            show_lines=False,
        )
        job_table.add_column("JobID", style="cyan", no_wrap=True)
        job_table.add_column("Name", max_width=25)
        job_table.add_column("Partition")
        job_table.add_column("GPUs", justify="right")
        job_table.add_column("GPU-Hours", justify="right")
        job_table.add_column("State")

        sorted_jobs = sorted(jobs, key=lambda j: -j["gpu_hours"])
        for job in sorted_jobs:
            color = _state_color(job["state"])
            job_table.add_row(
                job["job_id"],
                job["job_name"],
                job["partition"],
                str(job["gpu_count"]),
                f"{job['gpu_hours']:,.1f}",
                f"[{color}]{job['state']}[/{color}]",
            )

        console.print(job_table)

    return 0
