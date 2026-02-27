"""koa limits -- QOS / quota / fair-share visibility.

Queries ``sacctmgr show qos``, ``sacctmgr show assoc``, and ``sshare`` to
display the user's current limits and usage at a glance.
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
        "limits",
        help="Show QOS limits, account associations, and fair-share scores.",
    )
    add_common_arguments(parser)
    return parser


def _parse_pipe_rows(raw: str) -> list[list[str]]:
    """Split pipe-delimited output into a list of row-lists."""
    rows: list[list[str]] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([c.strip() for c in line.split("|")])
    return rows


def _parse_current_usage(raw: str) -> dict:
    """Parse squeue output to compute running/pending counts and GPU usage."""
    running = 0
    pending = 0
    gpus_in_use = 0
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        state = parts[1]
        gres = parts[2]
        if state == "RUNNING":
            running += 1
            gpus_in_use += _count_gpus_from_gres(gres)
        elif state == "PENDING":
            pending += 1
    return {
        "running": running,
        "pending": pending,
        "total": running + pending,
        "gpus_in_use": gpus_in_use,
    }


def _count_gpus_from_gres(gres: str) -> int:
    """Extract GPU count from a GRES string like 'gpu:a100:2'."""
    if not gres:
        return 0
    total = 0
    for entry in gres.split(","):
        entry = entry.strip()
        if not entry.startswith("gpu"):
            continue
        segments = entry.split(":")
        try:
            total += int(segments[-1])
        except (ValueError, IndexError):
            total += 1
    return total


def _extract_max_gpus(tres_str: str) -> str:
    """Extract GPU limit from a TRES string like 'cpu=100,gres/gpu=4'."""
    if not tres_str:
        return ""
    for item in tres_str.split(","):
        item = item.strip()
        if "gpu" in item.lower():
            parts = item.split("=")
            if len(parts) == 2:
                return parts[1]
    return ""


def handle(args, config: Config) -> int:
    # --- Query QOS limits ---
    qos_rows: list[list[str]] = []
    try:
        result = run_ssh(
            config,
            [
                "sacctmgr", "show", "qos",
                "format=Name,MaxTRESPerUser,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxWall",
                "-P", "-n",
            ],
            capture_output=True,
        )
        qos_rows = _parse_pipe_rows(result.stdout)
    except SSHError as exc:
        print(
            f"Warning: could not query QOS limits (sacctmgr may require admin): {exc}",
            file=sys.stderr,
        )

    # --- Query account associations ---
    assoc_rows: list[list[str]] = []
    try:
        result = run_ssh(
            config,
            [
                "sacctmgr", "show", "assoc",
                f"where user={config.user}",
                "format=Account,User,QOS,MaxTRESPerUser,MaxJobs,MaxSubmit,GrpTRES",
                "-P", "-n",
            ],
            capture_output=True,
        )
        assoc_rows = _parse_pipe_rows(result.stdout)
    except SSHError as exc:
        print(
            f"Warning: could not query account associations: {exc}",
            file=sys.stderr,
        )

    # --- Query fair-share ---
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

    # --- Query current usage ---
    usage = {"running": 0, "pending": 0, "total": 0, "gpus_in_use": 0}
    try:
        result = run_ssh(
            config,
            ["squeue", "-u", config.user, "-h", "-o", "%i|%T|%b"],
            capture_output=True,
        )
        usage = _parse_current_usage(result.stdout)
    except SSHError:
        pass

    # --- JSON output ---
    if args.output_format == "json":
        data = {
            "qos": [
                {
                    "name": r[0] if len(r) > 0 else "",
                    "max_tres_per_user": r[1] if len(r) > 1 else "",
                    "max_jobs_per_user": r[2] if len(r) > 2 else "",
                    "max_submit_per_user": r[3] if len(r) > 3 else "",
                    "max_wall": r[4] if len(r) > 4 else "",
                }
                for r in qos_rows
            ],
            "associations": [
                {
                    "account": r[0] if len(r) > 0 else "",
                    "user": r[1] if len(r) > 1 else "",
                    "qos": r[2] if len(r) > 2 else "",
                    "max_tres_per_user": r[3] if len(r) > 3 else "",
                    "max_jobs": r[4] if len(r) > 4 else "",
                    "max_submit": r[5] if len(r) > 5 else "",
                    "grp_tres": r[6] if len(r) > 6 else "",
                }
                for r in assoc_rows
            ],
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
            "current_usage": usage,
        }
        emit_json(data)
        return 0

    # --- Rich table output ---

    # 1. QOS Limits
    if qos_rows:
        tbl = Table(title="QOS Limits", show_lines=False)
        tbl.add_column("Name", style="cyan")
        tbl.add_column("Max TRES/User")
        tbl.add_column("Max Jobs", justify="right")
        tbl.add_column("Max Submit", justify="right")
        tbl.add_column("Max Wall", justify="right")
        for r in qos_rows:
            tbl.add_row(
                r[0] if len(r) > 0 else "",
                r[1] if len(r) > 1 else "",
                r[2] if len(r) > 2 else "",
                r[3] if len(r) > 3 else "",
                r[4] if len(r) > 4 else "",
            )
        console.print(tbl)
        console.print()

    # 2. Account Associations (with cross-referenced current usage)
    if assoc_rows:
        tbl = Table(title="Account Associations", show_lines=False)
        tbl.add_column("Account", style="cyan")
        tbl.add_column("QOS")
        tbl.add_column("Max TRES/User")
        tbl.add_column("Max Jobs", justify="right")
        tbl.add_column("Current Jobs", justify="right")
        tbl.add_column("Current GPUs", justify="right")
        for r in assoc_rows:
            max_jobs = r[4] if len(r) > 4 else ""
            max_gpu_str = _extract_max_gpus(r[3] if len(r) > 3 else "")

            current_jobs_str = str(usage["total"])
            if max_jobs:
                current_jobs_str = f"{usage['total']} of {max_jobs}"

            current_gpus_str = str(usage["gpus_in_use"])
            if max_gpu_str:
                current_gpus_str = f"{usage['gpus_in_use']} of {max_gpu_str}"

            tbl.add_row(
                r[0] if len(r) > 0 else "",
                r[2] if len(r) > 2 else "",
                r[3] if len(r) > 3 else "",
                max_jobs if max_jobs else "-",
                current_jobs_str,
                current_gpus_str,
            )
        console.print(tbl)
        console.print()

    # 3. Fair-Share
    if share_rows:
        tbl = Table(title="Fair-Share", show_lines=False)
        tbl.add_column("User", style="cyan")
        tbl.add_column("Raw Shares", justify="right")
        tbl.add_column("Norm Shares", justify="right")
        tbl.add_column("Raw Usage", justify="right")
        tbl.add_column("Norm Usage", justify="right")
        tbl.add_column("Effectiveness", justify="right")
        tbl.add_column("Fair-Share Factor", justify="right")
        for r in share_rows:
            tbl.add_row(
                r[1] if len(r) > 1 else "",
                r[2] if len(r) > 2 else "",
                r[3] if len(r) > 3 else "",
                r[4] if len(r) > 4 else "",
                r[5] if len(r) > 5 else "",
                r[6] if len(r) > 6 else "",
                r[7] if len(r) > 7 else "",
            )
        console.print(tbl)
        console.print()

    if not qos_rows and not assoc_rows and not share_rows:
        console.print("[dim]No limit or fair-share data available.[/dim]")

    return 0
