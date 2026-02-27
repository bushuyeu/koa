"""koa audit -- Right-sizing advisor.

Queries ``sacct`` for historical resource usage and computes efficiency ratios
(memory, time, CPU).  Suggests tighter resource requests so jobs become
backfill-friendly.
"""
from __future__ import annotations

import argparse
import re
import shlex
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

# Multipliers for memory unit normalization (to MB)
_MEM_MULTIPLIERS = {
    "K": 1 / 1024,
    "M": 1,
    "G": 1024,
    "T": 1024 * 1024,
}


def _parse_mem_mb(value: str) -> Optional[float]:
    """Parse a SLURM memory string like '4096M', '16G', '512K', '4096' to MB."""
    if not value or value == "0" or value in ("", "0n"):
        return None
    value = value.strip()
    # Handle per-node/per-cpu suffixes like "4Gn" or "4Gc"
    value = value.rstrip("nc")
    if not value:
        return None
    suffix = value[-1].upper()
    if suffix in _MEM_MULTIPLIERS:
        try:
            return float(value[:-1]) * _MEM_MULTIPLIERS[suffix]
        except ValueError:
            return None
    # No suffix -- assume MB
    try:
        return float(value)
    except ValueError:
        return None


def _parse_slurm_time_seconds(time_str: str) -> Optional[float]:
    """Parse SLURM time formats: D-HH:MM:SS, HH:MM:SS, MM:SS, or MM:SS.mmm."""
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


def _format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or D-HH:MM:SS."""
    s = int(seconds)
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, s = divmod(s, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{s:02d}"
    return f"{hours:02d}:{minutes:02d}:{s:02d}"


def _format_mem(mb: float) -> str:
    """Format MB into a human-friendly string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f}G"
    return f"{mb:.0f}M"


def _efficiency_style(ratio: float) -> str:
    """Return a Rich color based on efficiency ratio."""
    if ratio >= 0.60:
        return "green"
    if ratio >= 0.30:
        return "yellow"
    return "red"


def _efficiency_pct(ratio: Optional[float]) -> str:
    """Format a ratio as a percentage string with color markup."""
    if ratio is None:
        return "[dim]N/A[/dim]"
    pct = ratio * 100
    style = _efficiency_style(ratio)
    return f"[{style}]{pct:.0f}%[/{style}]"


def _suggest_value(actual: float, multiplier: float = 1.3) -> float:
    """Suggest a right-sized value: 130% of actual peak."""
    return actual * multiplier


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


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "audit",
        help="Right-sizing advisor -- analyze job history for resource waste.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Lookback period in days (default: 7).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=20,
        help="Maximum number of jobs to analyze (default: 20).",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    days = int(max(1, args.days))
    max_jobs = int(max(1, args.jobs))

    # sacct query -- use a date expression compatible with GNU date on the cluster
    sacct_cmd = (
        f"sacct -u {shlex.quote(config.user)} "
        f"--format=JobID,JobName%30,MaxRSS,ReqMem,Elapsed,Timelimit,AllocCPUS,TotalCPU,AllocTRES%60,State%20 "
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
        console.print("[dim]No completed jobs found in the last {days} day(s).[/dim]")
        return 0

    # Parse jobs -- skip sub-steps (JobID contains '.')
    jobs: list[dict] = []
    for line in lines:
        fields = line.split("|")
        if len(fields) < 10:
            continue

        job_id = fields[0].strip()
        # Skip batch/extern sub-steps
        if "." in job_id:
            continue

        job_name = fields[1].strip()
        max_rss_raw = fields[2].strip()
        req_mem_raw = fields[3].strip()
        elapsed_raw = fields[4].strip()
        timelimit_raw = fields[5].strip()
        alloc_cpus_raw = fields[6].strip()
        total_cpu_raw = fields[7].strip()
        alloc_tres_raw = fields[8].strip()
        state = fields[9].strip()

        # Skip jobs that never started
        if state in ("PENDING", "CANCELLED+", "CANCELLED"):
            continue

        max_rss_mb = _parse_mem_mb(max_rss_raw)
        req_mem_mb = _parse_mem_mb(req_mem_raw)
        elapsed_s = _parse_slurm_time_seconds(elapsed_raw)
        timelimit_s = _parse_slurm_time_seconds(timelimit_raw)
        total_cpu_s = _parse_slurm_time_seconds(total_cpu_raw)

        try:
            alloc_cpus = int(alloc_cpus_raw)
        except ValueError:
            alloc_cpus = 0

        gpu_count = _parse_tres_gpu_count(alloc_tres_raw)

        # Compute efficiencies
        mem_eff: Optional[float] = None
        if max_rss_mb and req_mem_mb and req_mem_mb > 0:
            mem_eff = min(max_rss_mb / req_mem_mb, 1.0)

        time_eff: Optional[float] = None
        if elapsed_s is not None and timelimit_s and timelimit_s > 0:
            time_eff = min(elapsed_s / timelimit_s, 1.0)

        cpu_eff: Optional[float] = None
        if total_cpu_s is not None and elapsed_s and alloc_cpus and elapsed_s > 0:
            cpu_eff = min(total_cpu_s / (elapsed_s * alloc_cpus), 1.0)

        # Suggested right-sized values
        suggested_mem: Optional[str] = None
        if max_rss_mb and max_rss_mb > 0:
            suggested_mem = _format_mem(_suggest_value(max_rss_mb))

        suggested_time: Optional[str] = None
        if elapsed_s is not None and elapsed_s > 0:
            suggested_time = _format_time(_suggest_value(elapsed_s))

        jobs.append({
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "alloc_cpus": alloc_cpus,
            "gpu_count": gpu_count,
            "max_rss_mb": max_rss_mb,
            "req_mem_mb": req_mem_mb,
            "elapsed_s": elapsed_s,
            "timelimit_s": timelimit_s,
            "total_cpu_s": total_cpu_s,
            "mem_eff": mem_eff,
            "time_eff": time_eff,
            "cpu_eff": cpu_eff,
            "suggested_mem": suggested_mem,
            "suggested_time": suggested_time,
        })

        if len(jobs) >= max_jobs:
            break

    if not jobs:
        console.print(f"[dim]No analyzable jobs found in the last {days} day(s).[/dim]")
        return 0

    if args.output_format == "json":
        emit_json(jobs)
        return 0

    # Rich table output
    table = Table(
        title=f"Resource Audit ({len(jobs)} jobs, last {days} day(s))",
        show_lines=False,
    )
    table.add_column("JobID", style="cyan", no_wrap=True)
    table.add_column("Name", max_width=25)
    table.add_column("State")
    table.add_column("Mem Eff", justify="right")
    table.add_column("Time Eff", justify="right")
    table.add_column("CPU Eff", justify="right")
    table.add_column("Suggested Mem", justify="right")
    table.add_column("Suggested Time", justify="right")

    for job in jobs:
        state = job["state"]
        state_style = "green" if "COMPLETED" in state else ("red" if "FAIL" in state or "TIMEOUT" in state else "yellow")

        table.add_row(
            job["job_id"],
            job["job_name"],
            f"[{state_style}]{state}[/{state_style}]",
            _efficiency_pct(job["mem_eff"]),
            _efficiency_pct(job["time_eff"]),
            _efficiency_pct(job["cpu_eff"]),
            job["suggested_mem"] or "[dim]-[/dim]",
            job["suggested_time"] or "[dim]-[/dim]",
        )

    console.print(table)

    # Summary statistics
    mem_effs = [j["mem_eff"] for j in jobs if j["mem_eff"] is not None]
    time_effs = [j["time_eff"] for j in jobs if j["time_eff"] is not None]
    cpu_effs = [j["cpu_eff"] for j in jobs if j["cpu_eff"] is not None]

    console.print()
    if mem_effs:
        avg_mem = sum(mem_effs) / len(mem_effs)
        console.print(f"  Avg memory efficiency:  {_efficiency_pct(avg_mem)}")
    if time_effs:
        avg_time = sum(time_effs) / len(time_effs)
        console.print(f"  Avg time efficiency:    {_efficiency_pct(avg_time)}")
    if cpu_effs:
        avg_cpu = sum(cpu_effs) / len(cpu_effs)
        console.print(f"  Avg CPU efficiency:     {_efficiency_pct(avg_cpu)}")

    # Compute wasted compute-hours
    total_allocated_hours = 0.0
    total_used_hours = 0.0
    for job in jobs:
        if job["elapsed_s"] is not None and job["timelimit_s"] is not None:
            cpus = max(job["alloc_cpus"], 1)
            allocated_h = (job["timelimit_s"] * cpus) / 3600
            used_h = (job["elapsed_s"] * cpus) / 3600
            total_allocated_hours += allocated_h
            total_used_hours += used_h

    wasted_hours = total_allocated_hours - total_used_hours
    if wasted_hours > 0:
        console.print(
            f"\n  [bold yellow]Wasted compute-hours:[/bold yellow] "
            f"{wasted_hours:,.0f} CPU-hours "
            f"({total_used_hours:,.0f} used of {total_allocated_hours:,.0f} allocated)"
        )
        console.print(
            "  [dim]Tip: Right-sizing resource requests makes jobs backfill-eligible "
            "and reduces queue wait times.[/dim]"
        )

    return 0
