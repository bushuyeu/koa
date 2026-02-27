"""Rich-formatted tables for KOA job and queue output."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()

# State -> Rich style mapping
_STATE_STYLES: dict[str, str] = {
    "RUNNING": "green",
    "COMPLETING": "green",
    "PENDING": "yellow",
    "SUSPENDED": "yellow",
    "FAILED": "red",
    "TIMEOUT": "red",
    "CANCELLED": "red",
    "NODE_FAIL": "red",
    "PREEMPTED": "red",
    "OUT_OF_MEMORY": "red",
}


def format_jobs_table(raw_output: str, username: str) -> None:
    """Print the user's jobs as a Rich table.

    Expects pipe-delimited squeue output with columns:
    JOBID|NAME|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)
    """
    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    if not lines:
        console.print("[dim]No active jobs.[/dim]")
        return

    table = Table(title=f"Jobs for {username}", show_lines=False)
    table.add_column("JOBID", style="cyan", no_wrap=True)
    table.add_column("NAME")
    table.add_column("STATE")
    table.add_column("TIME", justify="right")
    table.add_column("TIME_LIMIT", justify="right")
    table.add_column("NODES", justify="right")
    table.add_column("NODELIST(REASON)")

    for i, line in enumerate(lines):
        if i == 0 and "JOBID" in line.upper():
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        state = parts[2].strip()
        style = _STATE_STYLES.get(state, "")
        table.add_row(*[p.strip() for p in parts[:7]], style=style)

    console.print(table)


def format_queue_table(raw_output: str, username: str, partition: str | None = None) -> None:
    """Print the full cluster queue as a Rich table, highlighting the current user's jobs.

    Expects pipe-delimited squeue output with columns:
    JOBID|USER|NAME|STATE|TIME|TIME_LIMIT|NODES|CPUS|MIN_MEMORY|NODELIST(REASON)
    """
    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    if not lines:
        console.print("[dim]Queue is empty.[/dim]")
        return

    title = "Cluster Queue"
    if partition:
        title += f" ({partition})"

    table = Table(title=title, show_lines=False)
    table.add_column("JOBID", style="cyan", no_wrap=True)
    table.add_column("USER")
    table.add_column("NAME")
    table.add_column("STATE")
    table.add_column("TIME", justify="right")
    table.add_column("TIME_LIMIT", justify="right")
    table.add_column("NODES", justify="right")
    table.add_column("CPUS", justify="right")
    table.add_column("MIN_MEMORY", justify="right")
    table.add_column("NODELIST(REASON)")

    user_job_count = 0
    for i, line in enumerate(lines):
        if i == 0 and "JOBID" in line.upper():
            continue
        parts = line.split("|")
        if len(parts) < 10:
            continue

        row_user = parts[1].strip()
        state = parts[3].strip()
        is_mine = row_user == username

        if is_mine:
            user_job_count += 1
            state_color = _STATE_STYLES.get(state, "white")
            style = f"bold {state_color}"
        else:
            style = "dim"

        table.add_row(*[p.strip() for p in parts[:10]], style=style)

    table.caption = f"{user_job_count} of your jobs in queue"
    console.print(table)
