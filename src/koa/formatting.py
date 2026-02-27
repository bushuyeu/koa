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


def _format_memory(mem_mb_str: str) -> str:
    """Convert memory in MB (e.g. '257000') to human-readable (e.g. '257 GB')."""
    try:
        mb = int(mem_mb_str)
    except ValueError:
        return mem_mb_str
    if mb >= 1_000_000:
        return f"{mb / 1_000_000:.1f} TB"
    return f"{mb // 1000} GB"


def format_availability_table(
    raw_output: str,
    partition: str | None = None,
    pending: dict[str, int] | None = None,
) -> None:
    """Print a Rich-formatted GPU/node availability table with a summary footer.

    Expects pipe-delimited sinfo output with columns:
    NODELIST|PARTITION|GRES|STATE|CPUS(A/I/O/T)|MEMORY

    ``pending`` maps ``{gpu_type: pending_job_count}`` from squeue.
    """
    lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
    if not lines:
        console.print("[dim]No nodes found.[/dim]")
        return

    pending = pending or {}

    title = "GPU Availability"
    if partition:
        title += f" ({partition})"

    # Track GPU totals: {gpu_type: {"available": N, "offline": N}}
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Deduplicate: merge partitions per node
    node_info: dict[str, dict] = {}
    seen_gpu_summary: set[str] = set()

    for line in lines:
        parts = line.split("|")
        if len(parts) < 6:
            continue

        node = parts[0].strip()
        part = parts[1].strip().rstrip("*")
        gres_raw = parts[2].strip()
        state = parts[3].strip().rstrip("*")
        memory = parts[5].strip()

        # Skip CPU-only nodes
        if gres_raw in ("(null)", "(N/A)", "") or "gpu:" not in gres_raw:
            continue

        state_lower = state.lower()
        is_down = state_lower not in (
            "idle", "mixed", "mix", "allocated", "alloc",
        )

        # Count total GPUs per type on this node
        from .slurm import GPU_NAME_MAP
        total_by_type: dict[str, int] = {}
        for entry in gres_raw.split(","):
            entry = entry.strip()
            if not entry.startswith("gpu:"):
                continue
            segments = entry.split(":")
            if len(segments) < 3:
                continue
            gpu_name = GPU_NAME_MAP.get(segments[1].lower(), segments[1].lower())
            count_str = segments[2].split("(")[0]
            try:
                total_by_type[gpu_name] = total_by_type.get(gpu_name, 0) + int(count_str)
            except ValueError:
                pass

        # Accumulate summary (once per physical node)
        if node not in seen_gpu_summary:
            seen_gpu_summary.add(node)
            for gpu_type, count in total_by_type.items():
                if is_down:
                    summary[gpu_type]["offline"] += count
                else:
                    summary[gpu_type]["available"] += count

        # Build GPU label with queue info
        friendly = _friendly_gpu(gres_raw)
        # Get the queue for the first GPU type on this node
        node_gpu_types = list(total_by_type.keys())
        node_queue = max((pending.get(g, 0) for g in node_gpu_types), default=0)

        if is_down:
            gpu_label = f"{friendly} [red](offline)[/red]"
        elif node_queue == 0:
            gpu_label = f"{friendly} [green](no queue)[/green]"
        else:
            gpu_label = (
                f"{friendly} [yellow](queue: {node_queue})[/yellow]"
            )

        style = _NODE_STATE_STYLES.get(state_lower, "")

        # Merge partitions for the same node
        if node in node_info:
            node_info[node]["partitions"].append(part)
        else:
            node_info[node] = {
                "partitions": [part],
                "gpu_label": gpu_label,
                "memory": memory,
                "style": style,
            }

    table = Table(title=title, show_lines=False)
    table.add_column("NODE", style="cyan", no_wrap=True)
    table.add_column("PARTITIONS")
    table.add_column("GPUs")
    table.add_column("MEMORY", justify="right")

    for node, info in node_info.items():
        table.add_row(
            node,
            ", ".join(info["partitions"]),
            info["gpu_label"],
            _format_memory(info["memory"]),
            style=info["style"],
        )

    console.print(table)

    # Summary footer
    if summary:
        console.print()
        sum_table = Table(title="GPU Summary", show_lines=False)
        sum_table.add_column("GPU Type", style="bold")
        sum_table.add_column("Queue", justify="right")
        sum_table.add_column("GPUs", justify="right")
        sum_table.add_column("Offline", style="red", justify="right")

        for gpu_type in sorted(summary, key=lambda g: (pending.get(g, 0), g)):
            s = summary[gpu_type]
            available = s.get("available", 0)
            offline = s.get("offline", 0)
            q = pending.get(gpu_type, 0)
            queue_str = (
                "[green]no queue[/green]" if q == 0
                else f"[yellow]{q} pending[/yellow]"
            )
            offline_str = str(offline) if offline else "[dim]0[/dim]"
            sum_table.add_row(gpu_type, queue_str, str(available), offline_str)

        console.print(sum_table)
