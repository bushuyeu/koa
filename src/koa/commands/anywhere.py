"""koa anywhere / koa submit --anywhere -- Cross-cluster smart routing.

Probes all configured backends in parallel via ``sbatch --test-only`` and
ranks them by estimated start time so the user can pick (or auto-submit to)
the fastest cluster.
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import yaml

from rich.console import Console
from rich.table import Table

from ..config import Config, DEFAULT_CONFIG_PATH, load_config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

# sbatch --test-only output pattern (same as optimize)
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


def _list_backend_names() -> list[str]:
    """Read the global config and return all backend cluster_name values."""
    if not DEFAULT_CONFIG_PATH.exists():
        return []
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    backends = data.get("backends")
    if not isinstance(backends, list):
        return []
    names: list[str] = []
    for entry in backends:
        name = entry.get("cluster_name")
        if name:
            names.append(name)
    return names


def _detect_partition(config: Config) -> str:
    """Return the default partition for a backend, or discover the first GPU partition."""
    if config.default_partition:
        return config.default_partition
    try:
        result = run_ssh(
            config,
            ["sinfo", "--Format=partitionname", "--noheader"],
            capture_output=True,
        )
        parts = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        return parts[0] if parts else "kill-shared"
    except SSHError:
        return "kill-shared"


def _detect_gpu_type(config: Config, partition: str) -> str:
    """Auto-detect the first available GPU type on the given partition."""
    try:
        result = run_ssh(
            config,
            ["sinfo", "-p", partition, "--Format=gres:40", "--noheader"],
            capture_output=True,
        )
    except SSHError:
        return "gpu"
    for line in result.stdout.splitlines():
        for entry in line.strip().split(","):
            entry = entry.strip()
            if entry.startswith("gpu:"):
                segments = entry.split(":")
                if len(segments) >= 2:
                    return segments[1].lower()
    return "gpu"


def _probe_backend(
    backend_name: str,
    script_path: str,
    walltime: str,
    config_path: Optional[str],
) -> dict:
    """Probe a single backend with sbatch --test-only. Returns a result dict."""
    now = datetime.now()
    entry: dict = {
        "backend": backend_name,
        "gpu_type": None,
        "partition": None,
        "est_start": None,
        "wait_time": None,
        "status": "unavailable",
        "error": None,
        "start_dt": None,
    }

    try:
        config = load_config(config_path, backend_name=backend_name)
    except (FileNotFoundError, ValueError) as exc:
        entry["error"] = f"Config error: {exc}"
        return entry

    partition = _detect_partition(config)
    entry["partition"] = partition

    gpu_type = _detect_gpu_type(config, partition)
    entry["gpu_type"] = gpu_type

    gres = f"gpu:{gpu_type}:1"
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
        entry["error"] = f"SSH error: {exc}"
        return entry

    start_time = _parse_start_time(combined)

    if start_time:
        entry["est_start"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
        entry["wait_time"] = _format_timedelta(start_time, now)
        entry["start_dt"] = start_time
        entry["status"] = "available"
    else:
        error_msg = combined.strip()
        if not error_msg:
            error_msg = "No start time returned"
        if len(error_msg) > 120:
            error_msg = error_msg[:120] + "..."
        entry["error"] = error_msg
        entry["status"] = "error"

    return entry


def _run_anywhere(
    script_path: str,
    walltime: str,
    config_path: Optional[str],
    output_format: str,
) -> tuple[list[dict], Optional[dict]]:
    """Probe all backends in parallel. Returns (results, recommendation)."""
    backend_names = _list_backend_names()
    if not backend_names:
        console.print(
            "[bold yellow]No backends configured.[/bold yellow] "
            "Add a 'backends' list to ~/.config/koa/config.yaml."
        )
        return [], None

    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(len(backend_names), 8)) as pool:
        futures = {
            pool.submit(
                _probe_backend, name, script_path, walltime, config_path,
            ): name
            for name in backend_names
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort: available backends first (by start time), then errors
    results.sort(
        key=lambda r: (
            r["start_dt"] is None,
            r["start_dt"] or datetime.max,
        )
    )

    recommendation = None
    viable = [r for r in results if r["start_dt"]]
    if viable:
        recommendation = viable[0]

    return results, recommendation


def _display_results(
    results: list[dict],
    recommendation: Optional[dict],
    output_format: str,
) -> None:
    """Render the comparison table or JSON output."""
    if output_format == "json":
        json_results = []
        for entry in results:
            out = dict(entry)
            out.pop("start_dt", None)
            json_results.append(out)
        emit_json({
            "backends": json_results,
            "recommendation": recommendation["backend"] if recommendation else None,
        })
        return

    table = Table(title="Cross-Cluster Comparison (fastest first)", show_lines=False)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Backend", style="cyan")
    table.add_column("GPU Type")
    table.add_column("Partition")
    table.add_column("Est. Start")
    table.add_column("Wait Time", justify="right")
    table.add_column("Status")

    for i, entry in enumerate(results):
        rank = str(i + 1) if entry["start_dt"] else "-"
        is_best = recommendation and entry["backend"] == recommendation["backend"]

        if entry["start_dt"]:
            table.add_row(
                rank,
                entry["backend"],
                entry["gpu_type"] or "-",
                entry["partition"] or "-",
                entry["est_start"],
                entry["wait_time"],
                "[green]available[/green]",
                style="green" if is_best else "",
            )
        else:
            table.add_row(
                rank,
                entry["backend"],
                entry["gpu_type"] or "-",
                entry["partition"] or "-",
                f"[red]{entry['error'] or 'unavailable'}[/red]",
                "-",
                f"[red]{entry['status']}[/red]",
                style="dim",
            )

    console.print(table)

    if recommendation:
        console.print(
            f"\n[bold green]Recommendation:[/bold green] "
            f"Submit to [cyan]{recommendation['backend']}[/cyan] "
            f"(estimated start in {recommendation['wait_time']})"
        )
    else:
        console.print(
            "\n[bold yellow]No backends returned a viable start time.[/bold yellow] "
            "Check cluster connectivity and job script validity."
        )


# ---------------------------------------------------------------------------
# register_anywhere_args -- add --anywhere flag to the submit parser
# ---------------------------------------------------------------------------


def register_anywhere_args(submit_parser: argparse.ArgumentParser) -> None:
    """Add --anywhere flag to the existing submit parser."""
    submit_parser.add_argument(
        "--anywhere",
        action="store_true",
        default=False,
        help="Auto-select the fastest backend across all configured clusters.",
    )


# ---------------------------------------------------------------------------
# handle_anywhere_submit -- called from _submit() when --anywhere is set
# ---------------------------------------------------------------------------


def handle_anywhere_submit(args, config_path: Optional[str]) -> int:
    """Probe all backends, pick the fastest, and submit there.

    Returns 0 on success, 1 on failure.
    """
    script_path = str(args.job_script)
    walltime = getattr(args, "time", None) or "04:00:00"
    output_format = getattr(args, "output_format", "table")

    console.print("[bold]Probing all backends for earliest start time...[/bold]")

    results, recommendation = _run_anywhere(
        script_path, walltime, config_path, output_format,
    )

    _display_results(results, recommendation, output_format)

    if not recommendation:
        return 1

    # Load the winning backend config and submit
    best_backend = recommendation["backend"]
    console.print(f"\n[bold]Submitting to [cyan]{best_backend}[/cyan]...[/bold]")

    try:
        config = load_config(config_path, backend_name=best_backend)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Failed to load config for {best_backend}:[/red] {exc}", file=sys.stderr)
        return 1

    from ..slurm import submit_job, parse_gpu_count_from_script

    # Build sbatch args
    sbatch_args: list[str] = []
    partition = recommendation.get("partition")
    if partition:
        sbatch_args.extend(["--partition", partition])
    if walltime:
        sbatch_args.extend(["--time", walltime])

    gpu_type = recommendation.get("gpu_type")
    if gpu_type:
        from pathlib import Path
        gpu_count = parse_gpu_count_from_script(Path(args.job_script))
        sbatch_args.append(f"--gres=gpu:{gpu_type}:{gpu_count}")

    if getattr(args, "account", None):
        sbatch_args.extend(["--account", args.account])
    elif config.default_account:
        sbatch_args.extend(["--account", config.default_account])

    sbatch_args.extend(getattr(args, "sbatch_arg", None) or [])

    try:
        job_id = submit_job(
            config,
            args.job_script,
            sbatch_args=sbatch_args,
            remote_name=getattr(args, "remote_name", None),
            job_desc=getattr(args, "desc", None),
        )
    except SSHError as exc:
        console.print(f"[red]Submission to {best_backend} failed:[/red] {exc}", file=sys.stderr)
        return 1

    console.print(f"[bold green]Submitted job {job_id} to {best_backend}[/bold green]")
    return 0


# ---------------------------------------------------------------------------
# register_parser / handle -- standalone `koa anywhere` command
# ---------------------------------------------------------------------------


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "anywhere",
        help="Compare estimated start times across all configured backends.",
    )
    parser.add_argument(
        "job_script",
        help="Path to the job script (remote path on each cluster).",
    )
    parser.add_argument(
        "--time",
        default="04:00:00",
        help="Walltime to simulate (default: 04:00:00).",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    """Handle standalone `koa anywhere <script>` -- compare only, no submit."""
    script_path = args.job_script
    walltime = args.time or "04:00:00"
    config_path = getattr(args, "config", None)
    output_format = getattr(args, "output_format", "table")

    console.print("[bold]Probing all backends for earliest start time...[/bold]")

    results, recommendation = _run_anywhere(
        script_path, walltime, config_path, output_format,
    )

    _display_results(results, recommendation, output_format)

    return 0
