"""Rich-formatted tables for KOA job and queue output."""
from __future__ import annotations

from collections import defaultdict

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


# Node state -> Rich style mapping for availability table
_NODE_STATE_STYLES: dict[str, str] = {
    "idle": "green",
    "mixed": "yellow",
    "mix": "yellow",
    "allocated": "yellow",
    "alloc": "yellow",
    "down": "red",
    "drain": "red",
    "drained": "red",
    "draining": "red",
    "down*": "red",
    "not_responding": "red",
}


def _friendly_gpu(gres: str) -> str:
    """Convert a raw GRES field like 'gpu:h100:4' to 'h100 x4'."""
    from .slurm import GPU_NAME_MAP

    parts_list: list[str] = []
    for entry in gres.split(","):
        entry = entry.strip()
        if not entry.startswith("gpu:"):
            parts_list.append(entry)
            continue
        segments = entry.split(":")
        if len(segments) < 3:
            parts_list.append(entry)
            continue
        gpu_name = segments[1].lower()
        count_str = segments[2].split("(")[0]
        friendly = GPU_NAME_MAP.get(gpu_name, gpu_name)
        parts_list.append(f"{friendly} x{count_str}")
    return ", ".join(parts_list) if parts_list else gres


def format_availability_table(raw_output: str, partition: str | None = None) -> None:
    """Print a Rich-formatted GPU/node availability table with a summary footer.

    Expects pipe-delimited sinfo output with columns:
    NODELIST|PARTITION|GRES|STATE|CPUS(A/I/O/T)|MEMORY
    """
    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    if not lines:
        console.print("[dim]No nodes found.[/dim]")
        return

    title = "GPU Node Availability"
    if partition:
        title += f" ({partition})"

    table = Table(title=title, show_lines=False)
    table.add_column("NODE", style="cyan", no_wrap=True)
    table.add_column("PARTITION")
    table.add_column("GPUs")
    table.add_column("STATE")
    table.add_column("CPUs (A/I/O/T)", justify="right")
    table.add_column("MEMORY", justify="right")

    # Track GPU summary: {gpu_type: {state_bucket: count}}
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for line in lines:
        parts = line.split("|")
        if len(parts) < 6:
            continue

        node = parts[0].strip()
        part = parts[1].strip().rstrip("*")
        gres_raw = parts[2].strip()
        state = parts[3].strip().rstrip("*")
        cpus = parts[4].strip()
        memory = parts[5].strip()

        state_lower = state.lower()
        style = _NODE_STATE_STYLES.get(state_lower, "")

        # Determine state bucket for summary
        if state_lower in ("idle",):
            bucket = "idle"
        elif state_lower in ("mixed", "mix", "allocated", "alloc"):
            bucket = "mixed"
        else:
            bucket = "down"

        # Accumulate GPU counts for summary
        for entry in gres_raw.split(","):
            entry = entry.strip()
            if not entry.startswith("gpu:"):
                continue
            segments = entry.split(":")
            if len(segments) < 3:
                continue
            from .slurm import GPU_NAME_MAP
            gpu_name = GPU_NAME_MAP.get(segments[1].lower(), segments[1].lower())
            count_str = segments[2].split("(")[0]
            try:
                count = int(count_str)
            except ValueError:
                continue
            summary[gpu_name][bucket] += count

        friendly_gres = _friendly_gpu(gres_raw)

        # Skip CPU-only nodes (no GPUs)
        if gres_raw in ("(null)", "(N/A)", "") or "gpu:" not in gres_raw:
            continue

        table.add_row(node, part, friendly_gres, state, cpus, memory, style=style)

    console.print(table)

    # Summary footer
    if summary:
        console.print()
        sum_table = Table(title="GPU Summary", show_lines=False)
        sum_table.add_column("GPU Type", style="bold")
        sum_table.add_column("Free", style="green", justify="right")
        sum_table.add_column("In Use", style="yellow", justify="right")
        sum_table.add_column("Offline", style="red", justify="right")
        sum_table.add_column("Total", justify="right")

        for gpu_type in sorted(summary, key=lambda g: (-sum(summary[g].values()), g)):
            buckets = summary[gpu_type]
            free = buckets.get("idle", 0)
            in_use = buckets.get("mixed", 0)
            offline = buckets.get("down", 0)
            total = free + in_use + offline
            sum_table.add_row(gpu_type, str(free), str(in_use), str(offline), str(total))

        console.print(sum_table)
