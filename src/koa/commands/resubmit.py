"""koa resubmit -- One-click job retry from stored manifest.

Reads the manifest recorded by a previous submission and replays it with the
same sbatch arguments and environment.
"""
from __future__ import annotations

import argparse
import re
import sys

from rich.console import Console

from ..config import Config
from ..runs import show_run
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "resubmit",
        help="Retry a previously submitted job using its stored manifest.",
    )
    parser.add_argument(
        "job_id",
        help="Job ID of the original submission to retry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the reconstructed sbatch command without submitting.",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    job_id = args.job_id

    entry = show_run(config, job_id)
    if not entry:
        console.print(
            f"[red]No run recorded with job ID {job_id}.[/red] "
            "Only jobs submitted through `koa submit` can be resubmitted.",
            file=sys.stderr,
        )
        return 1

    sbatch_args: list[str] = entry.get("sbatch_args") or []
    manifest = entry.get("manifest") or entry.get("git") or {}
    description = entry.get("description") or ""
    remote_job_dir = entry.get("remote_job_dir")

    # Try to recover the job script path.  The manifest stores it under
    # "job_script"; fall back to scanning sbatch_args for the trailing
    # positional argument (the script path).
    job_script = manifest.get("job_script") or entry.get("job_script")
    if not job_script and sbatch_args:
        # The last non-flag element is usually the script path
        for candidate in reversed(sbatch_args):
            if not candidate.startswith("-"):
                job_script = candidate
                break

    if not job_script:
        console.print(
            f"[red]Cannot determine job script for job {job_id}.[/red] "
            "The stored manifest is missing the script path.",
            file=sys.stderr,
        )
        return 1

    # Build the sbatch command
    sbatch_cmd = ["sbatch"] + list(sbatch_args) + [job_script]

    if args.dry_run:
        console.print("[bold]Dry-run:[/bold] would execute the following on the cluster:\n")
        console.print(f"  [cyan]{' '.join(sbatch_cmd)}[/cyan]\n")
        if description:
            console.print(f"  Description: {description}")
        if remote_job_dir:
            console.print(f"  Original run dir: {remote_job_dir}")
        if args.output_format == "json":
            emit_json({
                "dry_run": True,
                "old_job_id": job_id,
                "sbatch_command": sbatch_cmd,
                "sbatch_args": sbatch_args,
                "job_script": job_script,
                "description": description,
            })
        return 0

    # Submit via SSH
    try:
        result = run_ssh(config, sbatch_cmd, capture_output=True)
    except SSHError as exc:
        console.print(f"[red]Submission failed:[/red] {exc}", file=sys.stderr)
        return 1

    output = result.stdout.strip() if result.stdout else ""
    match = SBATCH_JOB_ID_PATTERN.search(output)
    if not match:
        console.print(
            f"[red]Unable to parse new job ID from sbatch output:[/red] {output}",
            file=sys.stderr,
        )
        return 1

    new_job_id = match.group(1)

    # Record the new run
    try:
        from ..runs import record_submission

        record_submission(
            config,
            job_id=new_job_id,
            sbatch_args=sbatch_args,
            manifest=manifest,
            local_job_dir=None,
            remote_job_dir=None,
            description=f"resubmit of {job_id}" + (f" ({description})" if description else ""),
        )
    except Exception:
        pass  # Non-critical; the job was already submitted

    console.print(
        f"[bold green]Resubmitted job {job_id} as {new_job_id}[/bold green]"
    )
    if description:
        console.print(f"  Description: {description}")
    console.print(f"  sbatch args: {' '.join(sbatch_args) if sbatch_args else '(none)'}")
    console.print(f"  Script: {job_script}")

    if args.output_format == "json":
        emit_json({
            "old_job_id": job_id,
            "new_job_id": new_job_id,
            "sbatch_args": sbatch_args,
            "job_script": job_script,
            "description": description,
        })

    return 0
