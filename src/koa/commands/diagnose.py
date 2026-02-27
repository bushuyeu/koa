"""koa diagnose -- Automatic job failure diagnosis.

Queries ``sacct`` for job exit state and resource usage, reads stderr logs,
and pattern-matches common failure signatures to produce an actionable
diagnosis with fix suggestions.
"""
from __future__ import annotations

import argparse
import math
import re
import shlex
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import Config
from ..slurm import get_job_io_paths
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

# ---------------------------------------------------------------------------
# Memory parsing helpers (shared pattern from audit.py)
# ---------------------------------------------------------------------------

_MEM_MULTIPLIERS = {
    "K": 1 / 1024,
    "M": 1,
    "G": 1024,
    "T": 1024 * 1024,
}


def _parse_mem_mb(value: str) -> Optional[float]:
    """Parse a SLURM memory string like '4096M', '16G', '512K' to MB."""
    if not value or value in ("", "0", "0n"):
        return None
    value = value.strip().rstrip("nc")
    if not value:
        return None
    suffix = value[-1].upper()
    if suffix in _MEM_MULTIPLIERS:
        try:
            return float(value[:-1]) * _MEM_MULTIPLIERS[suffix]
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_mem(mb: float) -> str:
    """Format MB into a human-friendly string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f}G"
    return f"{mb:.0f}M"


# ---------------------------------------------------------------------------
# Sacct query and parsing
# ---------------------------------------------------------------------------

_SACCT_FORMAT = (
    "JobID,State,ExitCode,MaxRSS,MaxVMSize,Elapsed,Timelimit,"
    "NodeList,Reason,ReqMem,AllocCPUS,AllocTRES%60"
)


def _query_sacct(config: Config, job_id: str) -> Optional[dict]:
    """Run sacct for *job_id* and return parsed fields, or None on failure."""
    cmd = [
        "sacct", "-j", str(job_id),
        f"--format={_SACCT_FORMAT}",
        "--parsable2", "--noheader",
    ]
    try:
        result = run_ssh(config, cmd, capture_output=True)
    except SSHError:
        return None

    # Use the main job line (skip sub-steps with '.' in JobID)
    for line in result.stdout.strip().splitlines():
        fields = line.strip().split("|")
        if len(fields) < 12:
            continue
        jid = fields[0].strip()
        if "." in jid:
            continue
        return {
            "job_id": jid,
            "state": fields[1].strip(),
            "exit_code": fields[2].strip(),
            "max_rss_raw": fields[3].strip(),
            "max_vmsize_raw": fields[4].strip(),
            "elapsed": fields[5].strip(),
            "timelimit": fields[6].strip(),
            "node": fields[7].strip(),
            "reason": fields[8].strip(),
            "req_mem_raw": fields[9].strip(),
            "alloc_cpus": fields[10].strip(),
            "alloc_tres": fields[11].strip(),
        }

    # Fallback: try the batch sub-step if no main line found
    for line in result.stdout.strip().splitlines():
        fields = line.strip().split("|")
        if len(fields) < 12:
            continue
        return {
            "job_id": fields[0].strip(),
            "state": fields[1].strip(),
            "exit_code": fields[2].strip(),
            "max_rss_raw": fields[3].strip(),
            "max_vmsize_raw": fields[4].strip(),
            "elapsed": fields[5].strip(),
            "timelimit": fields[6].strip(),
            "node": fields[7].strip(),
            "reason": fields[8].strip(),
            "req_mem_raw": fields[9].strip(),
            "alloc_cpus": fields[10].strip(),
            "alloc_tres": fields[11].strip(),
        }
    return None


# ---------------------------------------------------------------------------
# Stderr log fetching
# ---------------------------------------------------------------------------


def _read_stderr(config: Config, job_id: str, tail_lines: int = 50) -> str:
    """Fetch the last *tail_lines* from the job's stderr log."""
    try:
        io_paths = get_job_io_paths(config, job_id)
    except SSHError:
        return ""
    if not io_paths.stderr:
        return ""
    quoted = shlex.quote(io_paths.stderr)
    try:
        result = run_ssh(
            config,
            ["bash", "-lc", f"tail -n {tail_lines} {quoted} 2>/dev/null"],
            capture_output=True,
            check=False,
        )
        return result.stdout or ""
    except SSHError:
        return ""


# ---------------------------------------------------------------------------
# Failure pattern matching
# ---------------------------------------------------------------------------


def _diagnose(sacct: dict, stderr: str) -> dict:
    """Produce a diagnosis dict from sacct fields and stderr content."""
    state = sacct["state"]
    exit_code = sacct["exit_code"]
    max_rss_mb = _parse_mem_mb(sacct["max_rss_raw"])
    req_mem_mb = _parse_mem_mb(sacct["req_mem_raw"])

    diag: dict = {
        "job_id": sacct["job_id"],
        "state": state,
        "exit_code": exit_code,
        "max_rss_mb": round(max_rss_mb, 1) if max_rss_mb else None,
        "elapsed": sacct["elapsed"],
        "timelimit": sacct["timelimit"],
        "node": sacct["node"],
        "diagnosis": "Unknown failure",
        "root_cause": None,
        "suggestion": None,
        "fix_command": None,
    }

    # --- Successful job ---
    if state == "COMPLETED" and exit_code == "0:0":
        diag["diagnosis"] = "Job completed successfully"
        diag["root_cause"] = "No failure detected"
        diag["suggestion"] = "No action needed."
        return diag

    # --- OOM (system kill) ---
    if state == "OUT_OF_MEMORY" or exit_code == "0:137":
        diag["diagnosis"] = "Out of Memory (OOM killed)"
        diag["root_cause"] = "Job exceeded allocated RAM"
        if max_rss_mb and max_rss_mb > 0:
            suggested = _format_mem(max_rss_mb * 1.3)
            diag["suggestion"] = (
                f"Peak RSS was {_format_mem(max_rss_mb)}. "
                f"Request at least {suggested} (130% of peak)."
            )
            suggested_val = math.ceil(max_rss_mb * 1.3 / 1024) if max_rss_mb * 1.3 >= 1024 else math.ceil(max_rss_mb * 1.3)
            unit = "G" if max_rss_mb * 1.3 >= 1024 else "M"
            diag["fix_command"] = f"--mem={suggested_val}{unit}"
        else:
            diag["suggestion"] = "Increase --mem. No peak RSS data available to size precisely."
        return diag

    # --- TIMEOUT ---
    if state == "TIMEOUT":
        diag["diagnosis"] = "Walltime exceeded"
        diag["root_cause"] = "Job did not finish within the allocated time limit"
        diag["suggestion"] = (
            f"Elapsed {sacct['elapsed']} hit limit {sacct['timelimit']}. "
            "Request more time with --time, or use 'koa submit --chain' to "
            "split into checkpointed segments."
        )
        diag["fix_command"] = "koa submit --chain"
        return diag

    # --- NODE_FAIL ---
    if state == "NODE_FAIL":
        node = sacct["node"]
        diag["diagnosis"] = "Node failure"
        diag["root_cause"] = f"Node {node} failed during job execution"
        diag["suggestion"] = (
            f"Exclude the faulty node and resubmit: --exclude={node}"
        )
        diag["fix_command"] = f"koa resubmit {sacct['job_id']} -- --exclude={node}"
        return diag

    # --- CANCELLED ---
    if "CANCELLED" in state:
        reason = sacct.get("reason", "")
        diag["diagnosis"] = "Job was cancelled"
        if "by" in state.lower():
            diag["root_cause"] = f"Cancelled ({state})"
        else:
            diag["root_cause"] = f"Cancelled (reason: {reason})" if reason else "Cancelled by user or admin"
        diag["suggestion"] = "Check if cancellation was intentional. Resubmit if needed."
        diag["fix_command"] = f"koa resubmit {sacct['job_id']}"
        return diag

    # --- Stderr pattern matching ---
    if stderr:
        # CUDA OOM
        if "CUDA out of memory" in stderr or "out of memory" in stderr.lower() and "cuda" in stderr.lower():
            # Try to extract allocation details
            alloc_match = re.search(
                r"Tried to allocate ([\d.]+\s*\w+)",
                stderr,
            )
            detail = ""
            if alloc_match:
                detail = f" (tried to allocate {alloc_match.group(1)})"
            diag["diagnosis"] = f"GPU VRAM exhausted{detail}"
            diag["root_cause"] = "Model or batch does not fit in GPU memory"
            diag["suggestion"] = (
                "Reduce batch size, enable gradient checkpointing, use mixed "
                "precision (fp16/bf16), or request a GPU with more VRAM."
            )
            return diag

        # NCCL errors
        if "NCCL" in stderr and ("error" in stderr.lower() or "warn" in stderr.lower()):
            diag["diagnosis"] = "NCCL communication error"
            diag["root_cause"] = "Distributed training communication failure"
            diag["suggestion"] = (
                "Set NCCL_DEBUG=INFO for details. Try: "
                "export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1"
            )
            return diag

        # ModuleNotFoundError
        mod_match = re.search(r"ModuleNotFoundError:\s*No module named '([^']+)'", stderr)
        if mod_match:
            module = mod_match.group(1)
            diag["diagnosis"] = f"Missing Python module: {module}"
            diag["root_cause"] = f"Module '{module}' is not installed in the runtime environment"
            diag["suggestion"] = f"Install the missing module: pip install {module}"
            diag["fix_command"] = f"pip install {module}"
            return diag

        # FileNotFoundError / No such file or directory
        path_match = re.search(r"No such file or directory:\s*['\"]?([^'\"\\n]+)['\"]?", stderr)
        if not path_match:
            path_match = re.search(r"FileNotFoundError:.*'([^']+)'", stderr)
        if path_match:
            missing = path_match.group(1).strip()
            diag["diagnosis"] = f"Missing file or directory: {missing}"
            diag["root_cause"] = f"Path does not exist: {missing}"
            diag["suggestion"] = (
                "Verify the path exists on the cluster. Check that data was "
                "staged before submission."
            )
            return diag

        # Permission denied
        perm_match = re.search(r"Permission denied:?\s*['\"]?([^'\"\\n]*)['\"]?", stderr, re.IGNORECASE)
        if perm_match:
            target = perm_match.group(1).strip() or "unknown path"
            diag["diagnosis"] = f"Permission denied: {target}"
            diag["root_cause"] = "Insufficient filesystem permissions"
            diag["suggestion"] = f"Check permissions on {target}. Run: ls -la {target}"
            return diag

    # --- Generic FAILED state ---
    if "FAILED" in state:
        diag["diagnosis"] = f"Job failed with exit code {exit_code}"
        diag["root_cause"] = "Non-zero exit code"
        if stderr:
            # Show last meaningful line from stderr
            last_lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
            if last_lines:
                diag["root_cause"] = last_lines[-1][:200]
        diag["suggestion"] = "Check the full stderr log for details."
        return diag

    # --- Catch-all ---
    diag["diagnosis"] = f"Job ended in state {state} (exit {exit_code})"
    diag["root_cause"] = sacct.get("reason") or "Unknown"
    diag["suggestion"] = "Review job logs for details."
    return diag


# ---------------------------------------------------------------------------
# CLI registration and handler
# ---------------------------------------------------------------------------


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "diagnose",
        help="Diagnose why a completed/failed job ended the way it did.",
    )
    parser.add_argument(
        "job_id",
        help="Slurm job ID to diagnose.",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    job_id = args.job_id

    # 1. Query sacct
    sacct = _query_sacct(config, job_id)
    if sacct is None:
        console.print(
            f"[red]Job {job_id} not found in sacct.[/red] "
            "It may not have started or has been purged."
        )
        return 1

    # 2. Read stderr
    stderr = _read_stderr(config, job_id)

    # 3. Diagnose
    diag = _diagnose(sacct, stderr)

    # 4. Output
    if args.output_format == "json":
        emit_json(diag)
        return 0

    # Color based on diagnosis severity
    state = diag["state"]
    if state == "COMPLETED" and diag["exit_code"] == "0:0":
        header_style = "green"
    elif state in ("TIMEOUT", "CANCELLED"):
        header_style = "yellow"
    else:
        header_style = "red"

    # Job details table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Field", style="bold")
    info_table.add_column("Value")

    info_table.add_row("Job ID", f"[cyan]{diag['job_id']}[/cyan]")
    info_table.add_row("State", f"[{header_style}]{state}[/{header_style}]")
    info_table.add_row("Exit Code", diag["exit_code"])
    if diag["max_rss_mb"]:
        info_table.add_row("Peak Memory", _format_mem(diag["max_rss_mb"]))
    info_table.add_row("Elapsed", diag["elapsed"])
    info_table.add_row("Time Limit", diag["timelimit"])
    if diag["node"]:
        info_table.add_row("Node", diag["node"])

    console.print(Panel(info_table, title=f"Job {job_id}", border_style="blue"))

    # Diagnosis panel
    body_parts = [f"[bold {header_style}]{diag['diagnosis']}[/bold {header_style}]"]
    if diag["root_cause"]:
        body_parts.append(f"\n[bold]Root cause:[/bold] {diag['root_cause']}")
    if diag["suggestion"]:
        body_parts.append(f"\n[bold]Suggestion:[/bold] {diag['suggestion']}")
    if diag["fix_command"]:
        body_parts.append(f"\n[bold]Fix:[/bold] [cyan]{diag['fix_command']}[/cyan]")

    console.print()
    console.print(Panel(
        "\n".join(body_parts),
        title="Diagnosis",
        border_style=header_style,
    ))

    return 0
