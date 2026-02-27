"""koa validate -- Pre-flight resource validation for SLURM job scripts.

Parses #SBATCH directives from a local script, queries the cluster for
partition/GPU/QOS information, and runs a series of checks to catch common
misconfigurations before submission.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh
from . import add_common_arguments, emit_json

console = Console()

# ---------------------------------------------------------------------------
# SBATCH directive parsing
# ---------------------------------------------------------------------------

_SBATCH_RE = re.compile(r"#SBATCH\s+--(\S+?)(?:=|\s+)(\S+)")

# Heuristic memory recommendations per GPU type (in GB).
_GPU_MEMORY_HEURISTIC: dict[str, int] = {
    "h200": 80,
    "h100": 80,
    "a100": 64,
    "a40": 48,
    "l40": 48,
    "a30": 32,
    "v100": 32,
    "rtx_a6000": 48,
    "rtx_a5000": 32,
    "rtx2080ti": 16,
    "t4": 16,
}

# Patterns that indicate GPU usage inside a script body.
_GPU_CODE_PATTERNS = re.compile(
    r"\bcuda\b|\btorch\b|\btensorflow\b|\bgpu\b|\bnvidia-smi\b",
    re.IGNORECASE,
)


def _parse_sbatch_directives(text: str) -> dict[str, str]:
    """Extract #SBATCH key=value pairs from script text."""
    directives: dict[str, str] = {}
    for match in _SBATCH_RE.finditer(text):
        directives[match.group(1)] = match.group(2)
    return directives


def _parse_gres(gres: str) -> tuple[Optional[str], int]:
    """Parse a gres string like 'gpu:a100:2' -> ('a100', 2)."""
    parts = gres.split(":")
    if len(parts) == 3 and parts[0] == "gpu":
        try:
            return parts[1].lower(), int(parts[2])
        except ValueError:
            return parts[1].lower(), 1
    if len(parts) == 2 and parts[0] == "gpu":
        try:
            return None, int(parts[1])
        except ValueError:
            return parts[1].lower(), 1
    return None, 0


def _parse_mem_gb(mem_str: str) -> float:
    """Convert a Slurm memory string (e.g. '64G', '4096M') to GB."""
    mem_str = mem_str.strip().upper()
    if mem_str.endswith("T"):
        return float(mem_str[:-1]) * 1024
    if mem_str.endswith("G"):
        return float(mem_str[:-1])
    if mem_str.endswith("M"):
        return float(mem_str[:-1]) / 1024
    if mem_str.endswith("K"):
        return float(mem_str[:-1]) / (1024 * 1024)
    # Default: assume megabytes
    try:
        return float(mem_str) / 1024
    except ValueError:
        return 0.0


def _parse_walltime_minutes(time_str: str) -> Optional[int]:
    """Convert a Slurm time string to minutes.

    Supported formats: MM, MM:SS, HH:MM:SS, D-HH:MM:SS, D-HH.
    """
    time_str = time_str.strip()
    days = 0
    if "-" in time_str:
        day_part, rest = time_str.split("-", 1)
        days = int(day_part)
        time_str = rest

    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            return days * 1440 + int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 2:
            return days * 1440 + int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 1:
            return days * 1440 + int(parts[0])
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Cluster queries
# ---------------------------------------------------------------------------


def _query_partitions(config: Config) -> set[str]:
    """Return the set of partition names visible on the cluster."""
    try:
        result = run_ssh(
            config,
            ["sinfo", "--Format=partitionname", "--noheader"],
            capture_output=True,
        )
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}
    except SSHError:
        return set()


def _query_partition_gpu_info(
    config: Config, partition: str
) -> dict[str, int]:
    """Return {gpu_type: max_per_node} for a given partition."""
    gpu_info: dict[str, int] = {}
    try:
        result = run_ssh(
            config,
            ["sinfo", "-p", partition, "--Format=gres:40", "--noheader"],
            capture_output=True,
        )
    except SSHError:
        return gpu_info
    for line in result.stdout.splitlines():
        for entry in line.strip().split(","):
            entry = entry.strip()
            if not entry.startswith("gpu:"):
                continue
            segments = entry.split(":")
            if len(segments) >= 3:
                gpu_type = segments[1].lower()
                count_str = segments[2].split("(")[0]
                try:
                    count = int(count_str)
                except ValueError:
                    count = 1
                gpu_info[gpu_type] = max(gpu_info.get(gpu_type, 0), count)
    return gpu_info


def _query_qos_max_wall(config: Config) -> dict[str, Optional[int]]:
    """Return {qos_name: max_wall_minutes} from sacctmgr."""
    qos_walls: dict[str, Optional[int]] = {}
    try:
        result = run_ssh(
            config,
            [
                "sacctmgr", "show", "qos",
                "format=Name,MaxWall", "-P", "-n",
            ],
            capture_output=True,
        )
    except SSHError:
        return qos_walls
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        name = parts[0]
        wall = parts[1]
        if wall:
            minutes = _parse_walltime_minutes(wall)
            qos_walls[name] = minutes
        else:
            qos_walls[name] = None
    return qos_walls


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def _check_result(
    check: str,
    status: str,
    message: str,
    suggestion: str = "",
) -> dict:
    return {
        "check": check,
        "status": status,
        "message": message,
        "suggestion": suggestion,
    }


def run_checks(
    script_text: str,
    directives: dict[str, str],
    config: Config,
    *,
    partition_override: Optional[str] = None,
    gpus_override: Optional[int] = None,
) -> list[dict]:
    """Run all validation checks and return a list of results."""
    results: list[dict] = []

    # Determine effective values from directives + overrides.
    partition = partition_override or directives.get("partition")
    gres_str = directives.get("gres", "")
    gpu_type, gpu_count = _parse_gres(gres_str) if gres_str else (None, 0)
    if gpus_override is not None:
        gpu_count = gpus_override
    mem_str = directives.get("mem")
    time_str = directives.get("time")
    output_path = directives.get("output")
    job_name = directives.get("job-name")

    # Query cluster info.
    cluster_partitions = _query_partitions(config)
    partition_gpus: dict[str, int] = {}
    if partition:
        partition_gpus = _query_partition_gpu_info(config, partition)
    qos_walls = _query_qos_max_wall(config)

    # 1. partition_exists
    if partition:
        if cluster_partitions and partition in cluster_partitions:
            results.append(_check_result(
                "partition_exists", "pass",
                f"Partition '{partition}' exists and is accessible.",
            ))
        elif cluster_partitions:
            results.append(_check_result(
                "partition_exists", "fail",
                f"Partition '{partition}' not found on cluster.",
                f"Available partitions: {', '.join(sorted(cluster_partitions))}",
            ))
        else:
            results.append(_check_result(
                "partition_exists", "warn",
                "Could not query partition list from cluster.",
                "Check SSH connectivity.",
            ))
    else:
        results.append(_check_result(
            "partition_exists", "warn",
            "No partition specified in script or CLI flags.",
            "Add --partition=<name> or #SBATCH --partition=<name>.",
        ))

    # 2. gpu_type_available
    if gpu_type and partition:
        if partition_gpus:
            if gpu_type in partition_gpus:
                results.append(_check_result(
                    "gpu_type_available", "pass",
                    f"GPU type '{gpu_type}' is available on partition '{partition}'.",
                ))
            else:
                avail = ", ".join(sorted(partition_gpus.keys())) if partition_gpus else "none detected"
                results.append(_check_result(
                    "gpu_type_available", "fail",
                    f"GPU type '{gpu_type}' not found on partition '{partition}'.",
                    f"Available GPU types: {avail}",
                ))
        else:
            results.append(_check_result(
                "gpu_type_available", "warn",
                f"Could not query GPU types for partition '{partition}'.",
            ))
    elif gpu_type and not partition:
        results.append(_check_result(
            "gpu_type_available", "warn",
            f"GPU type '{gpu_type}' requested but no partition specified; cannot verify.",
        ))

    # 3. gpu_count_valid
    if gpu_count > 0 and partition:
        if partition_gpus:
            max_on_node = max(partition_gpus.values()) if partition_gpus else 0
            if gpu_type and gpu_type in partition_gpus:
                max_on_node = partition_gpus[gpu_type]
            if gpu_count <= max_on_node:
                results.append(_check_result(
                    "gpu_count_valid", "pass",
                    f"GPU count {gpu_count} is within node limits (max {max_on_node}).",
                ))
            else:
                results.append(_check_result(
                    "gpu_count_valid", "fail",
                    f"Requested {gpu_count} GPUs but max per node is {max_on_node}.",
                    f"Reduce --gres count or use multi-node with --nodes.",
                ))
        else:
            results.append(_check_result(
                "gpu_count_valid", "warn",
                "Could not determine max GPUs per node.",
            ))

    # 4. memory_adequate
    if mem_str and gpu_type:
        mem_gb = _parse_mem_gb(mem_str)
        recommended = _GPU_MEMORY_HEURISTIC.get(gpu_type)
        if recommended and mem_gb < recommended:
            results.append(_check_result(
                "memory_adequate", "warn",
                f"Requested {mem_str} ({mem_gb:.0f}G) may be low for {gpu_type} "
                f"(recommended >= {recommended}G).",
                f"Consider --mem={recommended}G for {gpu_type} workloads.",
            ))
        elif recommended:
            results.append(_check_result(
                "memory_adequate", "pass",
                f"Memory {mem_str} ({mem_gb:.0f}G) meets recommendation for {gpu_type}.",
            ))
    elif gpu_type and not mem_str:
        recommended = _GPU_MEMORY_HEURISTIC.get(gpu_type)
        if recommended:
            results.append(_check_result(
                "memory_adequate", "warn",
                f"No --mem specified; consider {recommended}G for {gpu_type}.",
                f"Add #SBATCH --mem={recommended}G.",
            ))

    # 5. time_within_limits
    if time_str:
        job_minutes = _parse_walltime_minutes(time_str)
        if job_minutes is not None and qos_walls:
            exceeded = []
            for qos_name, max_minutes in qos_walls.items():
                if max_minutes is not None and job_minutes > max_minutes:
                    exceeded.append(f"{qos_name} (max {max_minutes}min)")
            if exceeded:
                results.append(_check_result(
                    "time_within_limits", "warn",
                    f"Walltime {time_str} ({job_minutes}min) exceeds QOS limits: "
                    + ", ".join(exceeded) + ".",
                    "Reduce --time or use a different QOS.",
                ))
            else:
                results.append(_check_result(
                    "time_within_limits", "pass",
                    f"Walltime {time_str} is within QOS limits.",
                ))
        elif job_minutes is None:
            results.append(_check_result(
                "time_within_limits", "warn",
                f"Could not parse walltime '{time_str}'.",
            ))
    else:
        results.append(_check_result(
            "time_within_limits", "warn",
            "No --time specified; cluster default will apply.",
            "Add #SBATCH --time=HH:MM:SS for predictable scheduling.",
        ))

    # 6. gpu_code_present
    if gpu_count > 0 or gpu_type:
        body_lines = [
            line for line in script_text.splitlines()
            if not line.strip().startswith("#SBATCH")
        ]
        body = "\n".join(body_lines)
        if _GPU_CODE_PATTERNS.search(body):
            results.append(_check_result(
                "gpu_code_present", "pass",
                "Script references GPU-related code (cuda/torch/tensorflow/gpu/nvidia-smi).",
            ))
        else:
            results.append(_check_result(
                "gpu_code_present", "warn",
                "GPUs requested but no GPU-related code detected in script body.",
                "Verify the script actually uses GPUs to avoid wasting resources.",
            ))

    # 7. output_dir_format
    if output_path and "%j" in output_path and not job_name:
        results.append(_check_result(
            "output_dir_format", "warn",
            "Output path uses %j (job ID) but no --job-name is set.",
            "Add #SBATCH --job-name=<name> for easier identification.",
        ))
    elif output_path and "%j" in output_path and job_name:
        results.append(_check_result(
            "output_dir_format", "pass",
            "Output path uses %j and --job-name is set.",
        ))

    return results


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate",
        help="Pre-flight validation of a SLURM job script.",
    )
    parser.add_argument(
        "script",
        help="Path to the local job script to validate.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Override the partition to validate against.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override GPU count for validation.",
    )
    parser.add_argument(
        "--time",
        default=None,
        dest="time_override",
        help="Override walltime for validation.",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: script not found: {script_path}", file=sys.stderr)
        return 1

    script_text = script_path.read_text(encoding="utf-8")
    directives = _parse_sbatch_directives(script_text)

    # Apply CLI overrides into directives for unified handling.
    if args.time_override:
        directives["time"] = args.time_override

    results = run_checks(
        script_text,
        directives,
        config,
        partition_override=args.partition,
        gpus_override=args.gpus,
    )

    if args.output_format == "json":
        emit_json(results)
        return 0

    # Rich table output
    _STATUS_ICON = {"pass": "[green]\u2713[/green]", "warn": "[yellow]\u26a0[/yellow]", "fail": "[red]\u2717[/red]"}

    table = Table(title="Job Script Validation", show_lines=False)
    table.add_column("", width=3)
    table.add_column("Check", style="cyan")
    table.add_column("Message")
    table.add_column("Suggestion", style="dim")

    for r in results:
        icon = _STATUS_ICON.get(r["status"], "?")
        table.add_row(
            icon,
            r["check"],
            r["message"],
            r.get("suggestion", ""),
        )

    console.print(table)

    # Summary line
    fail_count = sum(1 for r in results if r["status"] == "fail")
    warn_count = sum(1 for r in results if r["status"] == "warn")
    pass_count = sum(1 for r in results if r["status"] == "pass")

    parts = []
    if pass_count:
        parts.append(f"[green]{pass_count} passed[/green]")
    if warn_count:
        parts.append(f"[yellow]{warn_count} warnings[/yellow]")
    if fail_count:
        parts.append(f"[red]{fail_count} failed[/red]")
    console.print("\n" + ", ".join(parts))

    if fail_count:
        console.print(
            "\n[bold red]Validation failed.[/bold red] "
            "Fix the issues above before submitting."
        )

    return 0


# ---------------------------------------------------------------------------
# Pre-submit helper (called from __main__._submit)
# ---------------------------------------------------------------------------


def validate_before_submit(script_path: Path, config: Config) -> None:
    """Run validation checks and print warnings. Does NOT block submission."""
    if not script_path.exists():
        return
    script_text = script_path.read_text(encoding="utf-8")
    directives = _parse_sbatch_directives(script_text)
    results = run_checks(script_text, directives, config)

    warnings = [r for r in results if r["status"] in ("warn", "fail")]
    if not warnings:
        return

    console.print("\n[bold yellow]Pre-submit validation warnings:[/bold yellow]")
    for r in warnings:
        icon = "[red]\u2717[/red]" if r["status"] == "fail" else "[yellow]\u26a0[/yellow]"
        msg = f"  {icon} {r['check']}: {r['message']}"
        if r.get("suggestion"):
            msg += f" ({r['suggestion']})"
        console.print(msg)
    console.print()
