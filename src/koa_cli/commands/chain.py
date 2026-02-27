"""koa submit --chain -- Auto job chaining.

Submits N dependent jobs with ``--dependency=afterok:$PREV`` so long training
runs are broken into backfill-friendly segments.
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import Optional

from rich.console import Console

from ..config import Config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


def register_chain_args(submit_parser: argparse.ArgumentParser) -> None:
    """Add --chain, --chain-time, and --off-peak flags to the submit parser."""
    submit_parser.add_argument(
        "--chain",
        type=int,
        metavar="N",
        default=None,
        help="Submit N chained jobs with dependency links.",
    )
    submit_parser.add_argument(
        "--chain-time",
        default="04:00:00",
        help="Walltime per chain segment (default: 04:00:00).",
    )
    submit_parser.add_argument(
        "--off-peak",
        action="store_true",
        help="Schedule the first job to start at 23:00 (off-peak hours).",
    )


def handle_chain_submit(
    args,
    config: Config,
    base_sbatch_args: list[str],
    job_script: str,
) -> list[str]:
    """Submit N chained jobs. Returns list of job IDs.

    Each link after the first depends on the previous via --dependency=afterok.
    All links get --signal=B:TERM@120 so the job script can checkpoint before
    the walltime limit.

    Environment variables injected:
        SLURM_CHAIN_LINK  — 0-based index of this link in the chain
        SLURM_CHAIN_TOTAL — total number of links
    """
    n_links = args.chain
    chain_time = args.chain_time
    off_peak = getattr(args, "off_peak", False)

    job_ids: list[str] = []

    for i in range(n_links):
        link_args = list(base_sbatch_args)

        # Override walltime with the chain segment time
        link_args.extend(["--time", chain_time])

        # Signal the job 120 seconds before walltime so it can checkpoint
        link_args.append("--signal=B:TERM@120")

        # Inject chain metadata as environment variables
        _inject_chain_env(link_args, link_index=i, total_links=n_links)

        # First link: apply off-peak scheduling if requested
        if i == 0 and off_peak:
            link_args.append("--begin=23:00")

        # Links after the first depend on the previous job
        if i > 0:
            prev_id = job_ids[-1]
            link_args.append(f"--dependency=afterok:{prev_id}")

        # Build and execute the sbatch command
        sbatch_cmd = ["sbatch"] + link_args + [job_script]

        try:
            result = run_ssh(config, sbatch_cmd, capture_output=True)
        except SSHError as exc:
            console.print(
                f"[red]Chain link {i} submission failed:[/red] {exc}",
                file=sys.stderr,
            )
            if job_ids:
                console.print(
                    f"[yellow]Warning: {len(job_ids)} earlier link(s) were already submitted: "
                    f"{', '.join(job_ids)}[/yellow]"
                )
            raise

        output = result.stdout.strip() if result.stdout else ""
        match = SBATCH_JOB_ID_PATTERN.search(output)
        if not match:
            console.print(
                f"[red]Could not parse job ID for chain link {i}:[/red] {output}",
                file=sys.stderr,
            )
            raise SSHError(f"Unable to parse sbatch output for chain link {i}: {output}")

        job_ids.append(match.group(1))

    return job_ids


def _inject_chain_env(args: list[str], link_index: int, total_links: int) -> None:
    """Inject SLURM_CHAIN_LINK and SLURM_CHAIN_TOTAL into --export if present,
    or add them as a new --export flag."""
    chain_vars = f"SLURM_CHAIN_LINK={link_index},SLURM_CHAIN_TOTAL={total_links}"

    # Look for an existing --export= flag to append to
    for i, arg in enumerate(args):
        if arg.startswith("--export="):
            existing = arg.split("=", 1)[1]
            args[i] = f"--export={existing},{chain_vars}"
            return

    # No existing --export; add one that passes through ALL plus our vars
    args.append(f"--export=ALL,{chain_vars}")


def display_chain_result(job_ids: list[str], off_peak: bool, chain_time: str) -> None:
    """Print a summary of the submitted chain."""
    chain_display = " -> ".join(job_ids)
    console.print(f"\n[bold green]Chain submitted:[/bold green] {chain_display}")
    console.print(f"  Links: {len(job_ids)}")
    console.print(f"  Time per link: {chain_time}")
    if off_peak:
        console.print("  First link scheduled for off-peak (23:00)")
    console.print(
        "  Each link signals TERM 120s before walltime for checkpointing."
    )


def chain_to_json(job_ids: list[str]) -> list[dict]:
    """Build JSON-serializable chain output."""
    result: list[dict] = []
    for i, jid in enumerate(job_ids):
        result.append({
            "link": i,
            "job_id": jid,
            "depends_on": job_ids[i - 1] if i > 0 else None,
        })
    return result


# ---------------------------------------------------------------------------
# register_parser / handle — `koa chain` is a no-op entry point that points
# users toward `koa submit --chain N`.
# ---------------------------------------------------------------------------


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "chain",
        help="(Use `koa submit --chain N` instead.) Shows help for job chaining.",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    console.print(
        "[bold]Job chaining is accessed via the submit command:[/bold]\n\n"
        "  koa submit script.slurm --chain 4\n"
        "  koa submit script.slurm --chain 4 --chain-time 04:00:00\n"
        "  koa submit script.slurm --chain 4 --off-peak\n\n"
        "This splits a long training run into N dependent segments.\n"
        "Each segment gets --signal=B:TERM@120 for checkpointing.\n"
        "Use --off-peak to start the first link at 23:00."
    )
    return 0
