"""koa efficiency -- Live GPU waste detector.

Queries ``nvidia-smi`` over SSH on the allocated node and ``sstat`` to report
real-time GPU/CPU/memory utilization for a running job.
"""
from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh
from . import add_common_arguments, emit_json

console = Console()


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "efficiency",
        help="Live GPU utilization and waste detection for a running job.",
    )
    add_common_arguments(parser)
    parser.add_argument("job_id", help="Job ID to inspect.")
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Refresh every N seconds (default: off, run once).",
    )
    return parser


def _parse_slurm_time(t: str) -> str:
    """Pass through SLURM time strings as-is for display."""
    return t.strip() if t else "-"


def _parse_memory(s: str) -> str:
    """Normalise SLURM memory values (e.g. '1234K' -> '1.2M')."""
    s = s.strip()
    if not s or s == "0":
        return s
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    suffix = s[-1].upper()
    if suffix in multipliers:
        try:
            val = float(s[:-1])
        except ValueError:
            return s
        # Convert to most readable unit
        bytes_val = val * multipliers[suffix]
        if bytes_val >= 1024**3:
            return f"{bytes_val / 1024**3:.1f}G"
        if bytes_val >= 1024**2:
            return f"{bytes_val / 1024**2:.1f}M"
        return s
    return s


def _gpu_util_style(util_pct: float) -> str:
    if util_pct >= 60:
        return "green"
    if util_pct >= 20:
        return "yellow"
    return "red"


def _query_job_info(config: Config, job_id: str) -> dict | None:
    """Get job node, state, GRES, time used, time limit from squeue."""
    try:
        result = run_ssh(
            config,
            ["squeue", "-j", job_id, "-h", "-o", "%N|%T|%b|%M|%l"],
            capture_output=True,
        )
    except SSHError as exc:
        print(f"Error querying job {job_id}: {exc}", file=sys.stderr)
        return None

    line = result.stdout.strip()
    if not line:
        print(f"Job {job_id} not found in the queue.", file=sys.stderr)
        return None

    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 5:
        print(f"Unexpected squeue output for job {job_id}: {line}", file=sys.stderr)
        return None

    return {
        "node": parts[0],
        "state": parts[1],
        "gres": parts[2],
        "time_used": parts[3],
        "time_limit": parts[4],
    }


def _query_gpu_stats(config: Config, node: str) -> list[dict]:
    """Query nvidia-smi on the compute node via SSH hop."""
    query_fields = (
        "index,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw"
    )
    try:
        result = run_ssh(
            config,
            [
                "ssh", node,
                "nvidia-smi",
                f"--query-gpu={query_fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
        )
    except SSHError as exc:
        print(f"Warning: could not query nvidia-smi on {node}: {exc}", file=sys.stderr)
        return []

    gpus: list[dict] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpu_util = float(parts[1])
        except ValueError:
            gpu_util = 0.0
        try:
            mem_util = float(parts[2])
        except ValueError:
            mem_util = 0.0
        try:
            mem_used = float(parts[3])
        except ValueError:
            mem_used = 0.0
        try:
            mem_total = float(parts[4])
        except ValueError:
            mem_total = 0.0
        try:
            temp = float(parts[5])
        except ValueError:
            temp = 0.0
        try:
            power = float(parts[6])
        except ValueError:
            power = 0.0

        gpus.append({
            "index": parts[0],
            "gpu_util": gpu_util,
            "mem_util": mem_util,
            "mem_used_mib": mem_used,
            "mem_total_mib": mem_total,
            "temperature": temp,
            "power_draw": power,
        })
    return gpus


def _query_cpu_stats(config: Config, job_id: str) -> dict | None:
    """Query sstat for CPU/memory stats. May fail on some clusters."""
    try:
        result = run_ssh(
            config,
            [
                "sstat", "-j", f"{job_id}.batch",
                "--format=MaxRSS,AveCPU,MaxVMSize",
                "-P", "-n",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip()
        if not line:
            return None
        # Take the first data line
        first = line.splitlines()[0].strip()
        parts = [p.strip() for p in first.split("|")]
        if len(parts) < 3:
            return None
        return {
            "max_rss": _parse_memory(parts[0]),
            "avg_cpu": parts[1],
            "max_vm_size": _parse_memory(parts[2]),
        }
    except SSHError:
        return None


def _render(
    job_info: dict,
    gpu_stats: list[dict],
    cpu_stats: dict | None,
    job_id: str,
) -> list[str]:
    """Render tables and collect warnings. Returns warning messages."""
    warnings: list[str] = []

    # 1. Job Resources
    tbl = Table(title=f"Job {job_id} Resources", show_lines=False)
    tbl.add_column("Node", style="cyan")
    tbl.add_column("State")
    tbl.add_column("Time Used", justify="right")
    tbl.add_column("Time Limit", justify="right")
    tbl.add_column("GPUs Allocated")
    state_style = "green" if job_info["state"] == "RUNNING" else "yellow"
    tbl.add_row(
        job_info["node"],
        f"[{state_style}]{job_info['state']}[/{state_style}]",
        _parse_slurm_time(job_info["time_used"]),
        _parse_slurm_time(job_info["time_limit"]),
        job_info["gres"],
    )
    console.print(tbl)
    console.print()

    # 2. GPU Utilization
    if gpu_stats:
        tbl = Table(title="GPU Utilization", show_lines=False)
        tbl.add_column("GPU#", style="cyan", justify="right")
        tbl.add_column("GPU Util%", justify="right")
        tbl.add_column("Mem Util%", justify="right")
        tbl.add_column("Mem Used/Total", justify="right")
        tbl.add_column("Temp", justify="right")
        tbl.add_column("Power", justify="right")
        tbl.add_column("Status")
        for gpu in gpu_stats:
            style = _gpu_util_style(gpu["gpu_util"])
            status = ""
            if gpu["gpu_util"] < 20:
                status = "[red bold]IDLE GPU![/red bold]"
                warnings.append(
                    f"GPU {gpu['index']} is at {gpu['gpu_util']:.0f}% utilization "
                    "-- consider reducing GPU count"
                )
            elif gpu["gpu_util"] < 60:
                status = "[yellow]Underutilised[/yellow]"

            mem_str = f"{gpu['mem_used_mib']:.0f}/{gpu['mem_total_mib']:.0f} MiB"
            tbl.add_row(
                str(gpu["index"]),
                f"[{style}]{gpu['gpu_util']:.0f}%[/{style}]",
                f"{gpu['mem_util']:.0f}%",
                mem_str,
                f"{gpu['temperature']:.0f}C",
                f"{gpu['power_draw']:.0f}W",
                status,
            )
        console.print(tbl)
        console.print()
    else:
        console.print("[dim]No GPU data available (nvidia-smi may not be accessible).[/dim]")
        console.print()

    # 3. CPU/Memory (optional)
    if cpu_stats:
        tbl = Table(title="CPU / Memory", show_lines=False)
        tbl.add_column("Max RSS", justify="right")
        tbl.add_column("Avg CPU", justify="right")
        tbl.add_column("Max VM Size", justify="right")
        tbl.add_row(cpu_stats["max_rss"], cpu_stats["avg_cpu"], cpu_stats["max_vm_size"])
        console.print(tbl)
        console.print()

    # Warnings
    for w in warnings:
        console.print(f"[bold yellow]WARNING:[/bold yellow] {w}")
    if warnings:
        console.print()

    return warnings


def handle(args, config: Config) -> int:
    job_id = args.job_id
    watch_interval = max(0, getattr(args, "watch", 0))

    job_info = _query_job_info(config, job_id)
    if job_info is None:
        return 1

    if job_info["state"] != "RUNNING":
        print(
            f"Job {job_id} is {job_info['state']}, not RUNNING. "
            "GPU stats are only available for running jobs.",
            file=sys.stderr,
        )
        return 1

    node = job_info["node"]
    if not node or node in ("(None)", ""):
        print(f"Job {job_id} has no allocated node.", file=sys.stderr)
        return 1

    # Single-shot mode
    if watch_interval <= 0:
        gpu_stats = _query_gpu_stats(config, node)
        cpu_stats = _query_cpu_stats(config, job_id)

        if args.output_format == "json":
            warning_msgs: list[str] = []
            for gpu in gpu_stats:
                if gpu["gpu_util"] < 20:
                    warning_msgs.append(
                        f"GPU {gpu['index']} is at {gpu['gpu_util']:.0f}% utilization"
                    )
            data = {
                "job_info": job_info,
                "gpu_stats": gpu_stats,
                "cpu_stats": cpu_stats,
                "warnings": warning_msgs,
            }
            emit_json(data)
            return 0

        _render(job_info, gpu_stats, cpu_stats, job_id)
        return 0

    # Watch mode
    try:
        while True:
            # Re-query job info each iteration (may have finished)
            job_info = _query_job_info(config, job_id)
            if job_info is None or job_info["state"] != "RUNNING":
                console.print(f"[yellow]Job {job_id} is no longer running.[/yellow]")
                break

            gpu_stats = _query_gpu_stats(config, node)
            cpu_stats = _query_cpu_stats(config, job_id)
            console.clear()
            _render(job_info, gpu_stats, cpu_stats, job_id)
            console.print(f"[dim]Refreshing every {watch_interval}s. Press Ctrl+C to stop.[/dim]")
            time.sleep(watch_interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/dim]")

    return 0
