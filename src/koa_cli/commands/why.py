"""koa why -- Pending reason decoder.

Parses ``scontrol show job`` for the Reason field and maps Slurm reason codes
to plain-English explanations with actionable advice.
"""
from __future__ import annotations

import argparse
import re
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

# Map SLURM reason codes to (explanation, advice)
REASON_MAP: dict[str, tuple[str, str]] = {
    "Priority": (
        "Your job is waiting for higher-priority jobs to finish.",
        "Try smaller resource requests, shorter walltime, or use koa optimize to find faster configs.",
    ),
    "Resources": (
        "Requested resources are not currently available.",
        "Check koa availability for idle GPUs. Consider fewer GPUs or a different partition.",
    ),
    "QOSMaxGRESPerUser": (
        "You've hit the QOS limit for GPUs per user.",
        "Cancel or wait for a running job to finish. Check limits with koa limits.",
    ),
    "QOSMaxJobsPerUser": (
        "You've hit the max concurrent jobs limit.",
        "Wait for a running job to complete or cancel one.",
    ),
    "ReqNodeNotAvail": (
        "Requested node(s) are down or reserved.",
        "Remove node constraints or wait for maintenance to finish.",
    ),
    "Dependency": (
        "Waiting on a dependent job to complete.",
        "Check the dependency chain. The parent job may have failed.",
    ),
    "PartitionTimeLimit": (
        "Requested time exceeds partition's max walltime.",
        "Reduce --time or switch to a partition with a longer limit.",
    ),
    "AssocGrpCPUMinutesLimit": (
        "Account CPU-minutes budget exhausted.",
        "Contact your PI or admin to request more allocation.",
    ),
    "AssocMaxJobsLimit": (
        "Account-level job limit reached.",
        "Wait for other jobs in your account to finish.",
    ),
    "BeginTime": (
        "Job is held until its requested start time.",
        "This is expected for --begin jobs. Wait or cancel and resubmit without --begin.",
    ),
    "JobHeldUser": (
        "Job is held by the user.",
        "Release with: scontrol release <job_id>",
    ),
    "JobHeldAdmin": (
        "Job is held by an administrator.",
        "Contact your cluster admin to release the hold.",
    ),
    "InvalidQOS": (
        "The requested QOS is not valid for your account.",
        "Check available QOS values with sacctmgr show qos or koa limits.",
    ),
    "BadConstraints": (
        "Job constraints cannot be satisfied.",
        "Check that the requested features/constraints match available nodes.",
    ),
    "PartitionNodeLimit": (
        "Requested more nodes than the partition allows.",
        "Reduce the node count or use a different partition.",
    ),
    "PartitionDown": (
        "The target partition is currently down.",
        "Switch to a different partition or wait for the partition to come back up.",
    ),
    "NonZeroExitCode": (
        "A previous job in a dependency chain exited with non-zero status.",
        "Check the logs of the parent job. Fix the error and resubmit.",
    ),
    "None": (
        "No specific reason reported (job may be starting soon).",
        "The scheduler has not assigned a pending reason. This usually means the job is about to start.",
    ),
}


def _parse_scontrol_field(output: str, field: str) -> Optional[str]:
    """Extract a field value from scontrol show job output."""
    pattern = re.compile(rf"{field}=(\S+)")
    match = pattern.search(output)
    return match.group(1) if match else None


def _parse_scontrol_fields(output: str) -> dict[str, str]:
    """Parse all key=value pairs from scontrol output into a dict."""
    fields: dict[str, str] = {}
    for match in re.finditer(r"(\w+)=(\S+)", output):
        fields[match.group(1)] = match.group(2)
    return fields


def _get_queue_position(config: Config, job_id: str, partition: Optional[str]) -> Optional[int]:
    """Determine the job's position in the pending queue."""
    cmd = ["squeue", "-t", "PD", "--sort=p,i", "-o", "%i", "--noheader"]
    if partition:
        cmd.extend(["-p", partition])
    try:
        result = run_ssh(config, cmd, capture_output=True, check=False)
        if result.returncode != 0:
            return None
        pending_ids = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        for pos, pid in enumerate(pending_ids, 1):
            if pid == str(job_id):
                return pos
        return None
    except SSHError:
        return None


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "why",
        help="Explain why a job is pending and suggest fixes.",
    )
    parser.add_argument(
        "job_id",
        help="Slurm job ID to investigate.",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    job_id = args.job_id

    # Query scontrol for full job details
    try:
        scontrol_result = run_ssh(
            config,
            ["scontrol", "show", "job", str(job_id)],
            capture_output=True,
        )
    except SSHError as exc:
        console.print(f"[red]Error querying job {job_id}:[/red] {exc}")
        return 1

    scontrol_output = scontrol_result.stdout
    if not scontrol_output.strip() or "Invalid job id" in scontrol_output:
        console.print(f"[red]Job {job_id} not found.[/red] It may have completed or been purged from the scheduler.")
        return 1

    fields = _parse_scontrol_fields(scontrol_output)
    state = fields.get("JobState", "UNKNOWN")
    reason = fields.get("Reason", "None")
    partition = fields.get("Partition")
    num_nodes = fields.get("NumNodes", "?")
    num_cpus = fields.get("NumCPUs", "?")
    time_limit = fields.get("TimeLimit", "?")
    submit_time = fields.get("SubmitTime", "?")
    start_time = fields.get("StartTime", "Unknown")
    gres = fields.get("Gres", fields.get("TresPerNode", "none"))
    job_name = fields.get("JobName", "?")
    account = fields.get("Account", "?")
    qos = fields.get("QOS", "?")
    dependency = fields.get("Dependency")

    # Also query squeue for additional context
    squeue_start: Optional[str] = None
    try:
        sq_result = run_ssh(
            config,
            ["squeue", "-j", str(job_id), "-o", "%i|%r|%S|%V|%P|%l|%m|%b", "--noheader"],
            capture_output=True,
            check=False,
        )
        if sq_result.returncode == 0 and sq_result.stdout.strip():
            sq_parts = sq_result.stdout.strip().split("|")
            if len(sq_parts) >= 3 and sq_parts[2].strip() not in ("N/A", "n/a", ""):
                squeue_start = sq_parts[2].strip()
    except SSHError:
        pass

    est_start = squeue_start or (start_time if start_time not in ("Unknown", "N/A") else None)

    # Get queue position for pending jobs
    queue_pos: Optional[int] = None
    if state == "PENDING":
        queue_pos = _get_queue_position(config, str(job_id), partition)

    # Look up reason explanation
    explanation, advice = REASON_MAP.get(
        reason,
        (
            f"Reason code '{reason}' is not in the known list.",
            "Check SLURM documentation or contact your cluster admin for details.",
        ),
    )

    result_data = {
        "job_id": job_id,
        "job_name": job_name,
        "state": state,
        "reason": reason,
        "explanation": explanation,
        "advice": advice,
        "partition": partition,
        "account": account,
        "qos": qos,
        "num_nodes": num_nodes,
        "num_cpus": num_cpus,
        "time_limit": time_limit,
        "gres": gres,
        "submit_time": submit_time,
        "est_start": est_start,
        "queue_position": queue_pos,
        "dependency": dependency,
    }

    if args.output_format == "json":
        emit_json(result_data)
        return 0

    # Rich panel output
    if state == "PENDING":
        state_style = "yellow"
    elif state in ("RUNNING", "COMPLETING"):
        state_style = "green"
    elif state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"):
        state_style = "red"
    else:
        state_style = "white"

    # Reason color
    reason_style = "red"
    if reason in ("Priority", "Resources", "BeginTime", "None"):
        reason_style = "yellow"

    # Build info table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Field", style="bold")
    info_table.add_column("Value")

    info_table.add_row("Job ID", f"[cyan]{job_id}[/cyan]")
    info_table.add_row("Name", job_name)
    info_table.add_row("State", f"[{state_style}]{state}[/{state_style}]")
    info_table.add_row("Reason", f"[{reason_style}]{reason}[/{reason_style}]")
    info_table.add_row("Partition", partition or "?")
    info_table.add_row("Account / QOS", f"{account} / {qos}")
    info_table.add_row("Resources", f"{num_nodes} node(s), {num_cpus} CPU(s), GRES: {gres}")
    info_table.add_row("Time Limit", time_limit)
    info_table.add_row("Submitted", submit_time)

    if est_start:
        info_table.add_row("Est. Start", f"[green]{est_start}[/green]")
    if queue_pos is not None:
        info_table.add_row("Queue Position", f"#{queue_pos}")
    if dependency:
        info_table.add_row("Dependency", dependency)

    console.print(Panel(info_table, title=f"Job {job_id}", border_style="blue"))

    # Explanation panel
    console.print()
    console.print(Panel(
        f"[bold {reason_style}]{reason}[/bold {reason_style}]\n\n"
        f"{explanation}\n\n"
        f"[bold]Advice:[/bold] {advice}",
        title="Diagnosis",
        border_style=reason_style,
    ))

    # Additional context for running jobs
    if state == "RUNNING":
        console.print(
            "\n[green]This job is currently running.[/green] "
            "No scheduling issue to diagnose."
        )
    elif state not in ("PENDING",):
        console.print(
            f"\n[dim]This job is in state {state}. "
            f"The reason shown reflects the last known scheduler state.[/dim]"
        )

    return 0
