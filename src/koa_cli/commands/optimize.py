"""koa optimize -- Dry-run scheduling simulator.

Uses ``sbatch --test-only`` to simulate scheduling across GPU type/count,
walltime, and partition combinations.  Ranks results by estimated start time
so the user can pick the fastest path to execution.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

# sbatch --test-only output pattern:
# "Job <id> to start at <datetime> using <N> processors ..."
_START_RE = re.compile(
    r"to start at\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})",
)


def _parse_start_time(output: str) -> Optional[datetime]:
    """Extract estimated start time from sbatch --test-only output."""
    match = _START_RE.search(output)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    return None


def _detect_gpu_types(config: Config, partitions: list[str]) -> list[str]:
    """Auto-detect GPU types available across the given partitions via sinfo."""
    gpu_types: set[str] = set()
    for part in partitions:
        try:
            result = run_ssh(
                config,
                ["sinfo", "-p", part, "--Format=gres:40", "--noheader"],
                capture_output=True,
            )
        except SSHError:
            continue
        for line in result.stdout.splitlines():
            for entry in line.strip().split(","):
                entry = entry.strip()
                if not entry.startswith("gpu:"):
                    continue
                segments = entry.split(":")
                if len(segments) >= 2:
                    gpu_types.add(segments[1].lower())
    return sorted(gpu_types) if gpu_types else ["rtx2080ti"]


def _detect_partitions(config: Config) -> list[str]:
    """Return the default partition from config, or discover GPU partitions."""
    if config.default_partition:
        return [config.default_partition]
    try:
        result = run_ssh(
            config,
            ["sinfo", "--Format=partitionname", "--noheader"],
            capture_output=True,
        )
        parts = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        return parts if parts else ["kill-shared"]
    except SSHError:
        return ["kill-shared"]


def _parse_script_time(script_path: str, config: Config) -> Optional[str]:
    """Try to extract --time from the remote script's #SBATCH directives."""
    try:
        result = run_ssh(
            config,
            ["grep", "-oP", r"(?<=#SBATCH\s--time=)\S+", script_path],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except SSHError:
        pass
    return None


def _parse_script_gpu_count(script_path: str, config: Config) -> int:
    """Extract GPU count from the remote script."""
    try:
        result = run_ssh(
            config,
            ["grep", "-oP", r"(?<=#SBATCH\s--gres=gpu)(?::[\w]+)?:(\d+)", script_path],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            # The last capture group has the count
            line = result.stdout.strip().splitlines()[0]
            parts = line.strip(":").split(":")
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
    except SSHError:
        pass
    return 1


def _format_timedelta(start: datetime, now: datetime) -> str:
    """Format wait time as a human-readable string."""
    if start <= now:
        return "now"
    delta = start - now
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_mins = minutes % 60
    if hours < 24:
        return f"{hours}h {remaining_mins}m" if remaining_mins else f"{hours}h"
    days = hours // 24
    remaining_hours = hours % 24
    return f"{days}d {remaining_hours}h" if remaining_hours else f"{days}d"


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "optimize",
        help="Dry-run scheduling simulator -- find the fastest path to GPU execution.",
    )
    parser.add_argument(
        "job_script",
        help="Path to the job script (remote path on the cluster).",
    )
    parser.add_argument(
        "--gpu-types",
        default=None,
        help="Comma-separated GPU types to test (default: auto-detect from sinfo).",
    )
    parser.add_argument(
        "--partitions",
        default=None,
        help="Comma-separated partitions to test (default: config default).",
    )
    parser.add_argument(
        "--max-time",
        default=None,
        help="Max walltime to try (e.g. 04:00:00). Default: read from script or 02:00:00.",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs to request (default: read from script or 1).",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    script_path = args.job_script

    # Determine partitions to test
    if args.partitions:
        partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    else:
        partitions = _detect_partitions(config)

    # Determine GPU types to test
    if args.gpu_types:
        gpu_types = [g.strip() for g in args.gpu_types.split(",") if g.strip()]
    else:
        gpu_types = _detect_gpu_types(config, partitions)

    # Determine walltime
    walltime = args.max_time
    if not walltime:
        walltime = _parse_script_time(script_path, config) or "02:00:00"

    # Determine GPU count
    gpu_count = args.gpu_count
    if gpu_count is None:
        gpu_count = _parse_script_gpu_count(script_path, config)

    now = datetime.now()
    results: list[dict] = []

    console.print(
        f"[bold]Testing {len(gpu_types)} GPU type(s) x {len(partitions)} partition(s)...[/bold]"
    )

    for gpu_type in gpu_types:
        for partition in partitions:
            gres = f"gpu:{gpu_type}:{gpu_count}"
            cmd = [
                "sbatch", "--test-only",
                f"--gres={gres}",
                f"--partition={partition}",
                f"--time={walltime}",
                script_path,
            ]

            try:
                result = run_ssh(config, cmd, capture_output=True, check=False)
                combined = (result.stdout or "") + "\n" + (result.stderr or "")
            except SSHError as exc:
                combined = str(exc)

            start_time = _parse_start_time(combined)

            entry: dict = {
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "partition": partition,
                "walltime": walltime,
                "gres": gres,
            }

            if start_time:
                wait = _format_timedelta(start_time, now)
                entry["est_start"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
                entry["wait_time"] = wait
                entry["start_dt"] = start_time
                entry["error"] = None
            else:
                # Check for common error messages
                error_msg = combined.strip()
                if not error_msg:
                    error_msg = "No start time returned"
                # Truncate long error messages
                if len(error_msg) > 120:
                    error_msg = error_msg[:120] + "..."
                entry["est_start"] = None
                entry["wait_time"] = None
                entry["start_dt"] = None
                entry["error"] = error_msg

            results.append(entry)

    # Sort: entries with start times first (by start time), then errors
    results.sort(
        key=lambda r: (
            r["start_dt"] is None,
            r["start_dt"] or datetime.max,
        )
    )

    # Assign ranks
    for i, entry in enumerate(results, 1):
        entry["rank"] = i if entry["start_dt"] else "-"

    if args.output_format == "json":
        json_results = []
        for entry in results:
            out = dict(entry)
            out.pop("start_dt", None)
            json_results.append(out)
        emit_json(json_results)
        return 0

    # Rich table output
    table = Table(title="Scheduling Options (fastest first)", show_lines=False)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("GPU Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Partition")
    table.add_column("Walltime", justify="right")
    table.add_column("Est. Start")
    table.add_column("Wait Time", justify="right")

    for entry in results:
        if entry["est_start"]:
            table.add_row(
                str(entry["rank"]),
                entry["gpu_type"],
                str(entry["gpu_count"]),
                entry["partition"],
                entry["walltime"],
                entry["est_start"],
                entry["wait_time"],
                style="green" if entry["rank"] == 1 else "",
            )
        else:
            table.add_row(
                str(entry["rank"]),
                entry["gpu_type"],
                str(entry["gpu_count"]),
                entry["partition"],
                entry["walltime"],
                f"[red]{entry['error']}[/red]",
                "-",
                style="dim",
            )

    console.print(table)

    # Quick recommendation
    viable = [r for r in results if r["start_dt"]]
    if viable:
        best = viable[0]
        console.print(
            f"\n[bold green]Recommendation:[/bold green] "
            f"Use [cyan]{best['gpu_type']}[/cyan] x{best['gpu_count']} "
            f"on [cyan]{best['partition']}[/cyan] "
            f"(estimated start in {best['wait_time']})"
        )
    else:
        console.print(
            "\n[bold yellow]No viable configurations found.[/bold yellow] "
            "Try different GPU types, partitions, or walltime values."
        )

    return 0
