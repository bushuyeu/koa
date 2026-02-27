"""koa submit --distributed -- Multi-node distributed training helper.

Adds distributed training flags to the submit parser and injects the
correct NCCL environment variables, sbatch node/task flags, and launcher
commands for PyTorch, DeepSpeed, and Horovod workloads.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import Config
from . import add_common_arguments, emit_json

console = Console()

# ---------------------------------------------------------------------------
# Framework detection patterns
# ---------------------------------------------------------------------------

_FRAMEWORK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("deepspeed", re.compile(r"\bdeepspeed\b|import\s+deepspeed")),
    ("horovod", re.compile(r"\bhorovod\b|import\s+horovod")),
    ("pytorch", re.compile(
        r"torch\.distributed|torch\.nn\.parallel|"
        r"\baccelerate\b|from\s+accelerate\b|"
        r"\btorchrun\b"
    )),
]

DEFAULT_MASTER_PORT = 29500
DEFAULT_CPUS_PER_TASK = 4


def detect_framework(script_content: str) -> str:
    """Auto-detect the distributed framework from script content.

    Returns one of: 'pytorch', 'deepspeed', 'horovod'.
    Defaults to 'pytorch' when nothing is detected.
    """
    for name, pattern in _FRAMEWORK_PATTERNS:
        if pattern.search(script_content):
            return name
    return "pytorch"


def _read_script(script_path: Path) -> str:
    """Read a script file, returning empty string on failure."""
    try:
        return script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Core analysis: build the distributed configuration dict
# ---------------------------------------------------------------------------


def build_distributed_config(
    script_path: Path,
    *,
    nodes: int = 2,
    gpus_per_node: int = 1,
    framework: str = "auto",
    master_port: int = DEFAULT_MASTER_PORT,
    cpus_per_task: int = DEFAULT_CPUS_PER_TASK,
    nccl_interface: Optional[str] = None,
    nccl_ib_disable: Optional[bool] = None,
    gpu_type: Optional[str] = None,
) -> dict[str, Any]:
    """Compute the full distributed training configuration.

    Returns a dict with keys: framework, nodes, gpus_per_node, world_size,
    env_vars, sbatch_flags, launcher_cmd.
    """
    content = _read_script(script_path)

    if framework == "auto":
        framework = detect_framework(content)

    world_size = nodes * gpus_per_node

    # Environment variables
    env_vars: dict[str, str] = {
        "MASTER_PORT": str(master_port),
        "WORLD_SIZE": str(world_size),
    }
    # MASTER_ADDR is resolved at job runtime via scontrol
    env_vars["MASTER_ADDR"] = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"

    if nccl_interface:
        env_vars["NCCL_SOCKET_IFNAME"] = nccl_interface
    if nccl_ib_disable is not None:
        env_vars["NCCL_IB_DISABLE"] = "1" if nccl_ib_disable else "0"

    # Sbatch flags
    gres_value = f"gpu:{gpus_per_node}"
    if gpu_type:
        gres_value = f"gpu:{gpu_type}:{gpus_per_node}"

    sbatch_flags: dict[str, str] = {
        "--nodes": str(nodes),
        "--ntasks-per-node": str(gpus_per_node),
        "--gres": gres_value,
        "--cpus-per-task": str(cpus_per_task),
    }

    # Launcher command suggestion
    script_name = script_path.name
    if framework == "deepspeed":
        launcher_cmd = (
            f"srun deepspeed --num_gpus={gpus_per_node} "
            f"--num_nodes={nodes} {script_name}"
        )
    elif framework == "horovod":
        launcher_cmd = f"srun --mpi=pmix python {script_name}"
    else:
        # pytorch (including accelerate)
        launcher_cmd = (
            f"srun torchrun "
            f"--nnodes={nodes} "
            f"--nproc_per_node={gpus_per_node} "
            f"--rdzv_id=$SLURM_JOB_ID "
            f"--rdzv_backend=c10d "
            f"--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT "
            f"{script_name}"
        )

    return {
        "framework": framework,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "env_vars": env_vars,
        "sbatch_flags": sbatch_flags,
        "launcher_cmd": launcher_cmd,
    }


# ---------------------------------------------------------------------------
# Rich display
# ---------------------------------------------------------------------------


def display_distributed_config(cfg: dict[str, Any]) -> None:
    """Print a rich panel summarizing the distributed configuration."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")

    table.add_row("Framework", cfg["framework"])
    table.add_row("Nodes", str(cfg["nodes"]))
    table.add_row("GPUs per node", str(cfg["gpus_per_node"]))
    table.add_row("World size", str(cfg["world_size"]))

    table.add_row("", "")
    table.add_row("[bold]Env vars[/bold]", "")
    for k, v in cfg["env_vars"].items():
        table.add_row(f"  {k}", v)

    table.add_row("", "")
    table.add_row("[bold]Sbatch flags[/bold]", "")
    for k, v in cfg["sbatch_flags"].items():
        table.add_row(f"  {k}", v)

    table.add_row("", "")
    table.add_row("[bold]Launcher[/bold]", cfg["launcher_cmd"])

    console.print(Panel(table, title="Distributed Training Config", border_style="green"))


# ---------------------------------------------------------------------------
# Submit integration (called from __main__._submit)
# ---------------------------------------------------------------------------


def register_distributed_args(submit_parser: argparse.ArgumentParser) -> None:
    """Add --distributed, --nodes, --gpus-per-node, --framework to the submit parser."""
    submit_parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable multi-node distributed training mode.",
    )
    submit_parser.add_argument(
        "--nodes",
        type=int,
        default=2,
        help="Number of nodes for distributed training (default: 2).",
    )
    submit_parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="GPUs per node for distributed training (default: 1).",
    )
    submit_parser.add_argument(
        "--framework",
        choices=["auto", "pytorch", "deepspeed", "horovod"],
        default="auto",
        help="Distributed framework (default: auto-detect from script).",
    )
    submit_parser.add_argument(
        "--master-port",
        type=int,
        default=DEFAULT_MASTER_PORT,
        help=f"Master port for rendezvous (default: {DEFAULT_MASTER_PORT}).",
    )


def handle_distributed_submit(
    args: argparse.Namespace,
    config: Config,
    base_sbatch_args: list[str],
    job_script_path: str,
) -> list[str]:
    """Inject distributed training flags into base_sbatch_args. Returns modified args list.

    Also prints the distributed configuration summary.
    """
    script_path = Path(args.job_script)

    # Pull NCCL settings from config backend if available
    nccl_interface = getattr(config, "nccl_interface", None)
    nccl_ib_disable = getattr(config, "nccl_ib_disable", None)

    gpu_type = None
    # Extract gpu type from existing --gres if present
    for arg in base_sbatch_args:
        if arg.startswith("--gres=gpu:"):
            parts = arg.split(":")
            if len(parts) >= 3:
                gpu_type = parts[1]

    cpus_per_task = DEFAULT_CPUS_PER_TASK
    for i, arg in enumerate(base_sbatch_args):
        if arg == "--cpus-per-task" and i + 1 < len(base_sbatch_args):
            try:
                cpus_per_task = int(base_sbatch_args[i + 1])
            except ValueError:
                pass

    dist_cfg = build_distributed_config(
        script_path,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        framework=args.framework,
        master_port=getattr(args, "master_port", DEFAULT_MASTER_PORT),
        cpus_per_task=cpus_per_task,
        nccl_interface=nccl_interface,
        nccl_ib_disable=nccl_ib_disable,
        gpu_type=gpu_type,
    )

    display_distributed_config(dist_cfg)

    # Inject sbatch flags
    modified_args = list(base_sbatch_args)

    for flag, value in dist_cfg["sbatch_flags"].items():
        # Remove existing conflicting flags
        modified_args = _remove_flag(modified_args, flag)
        if flag == "--gres":
            modified_args.append(f"--gres={value}")
        else:
            modified_args.extend([flag, value])

    # Inject environment variables via --export
    env_export_parts: list[str] = []
    for k, v in dist_cfg["env_vars"].items():
        env_export_parts.append(f"{k}={v}")

    # Merge with existing --export if present
    existing_export = None
    for i, arg in enumerate(modified_args):
        if arg.startswith("--export="):
            existing_export = i
            break

    if existing_export is not None:
        existing_val = modified_args[existing_export].split("=", 1)[1]
        modified_args[existing_export] = (
            f"--export={existing_val},{','.join(env_export_parts)}"
        )
    else:
        modified_args.append(f"--export=ALL,{','.join(env_export_parts)}")

    return modified_args


def _remove_flag(args: list[str], flag: str) -> list[str]:
    """Remove a flag and its value from an args list."""
    result: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        if arg.startswith(f"{flag}="):
            continue
        result.append(arg)
    return result


# ---------------------------------------------------------------------------
# Standalone command: koa distributed show <script>
# ---------------------------------------------------------------------------


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "distributed",
        help="Show distributed training configuration for a script.",
    )
    add_common_arguments(parser)
    dist_sub = parser.add_subparsers(dest="distributed_command")
    show_parser = dist_sub.add_parser(
        "show",
        help="Dry-run: show what env vars and flags would be injected.",
    )
    show_parser.add_argument("script", type=Path, help="Path to the training script.")
    show_parser.add_argument(
        "--nodes", type=int, default=2, help="Number of nodes (default: 2)."
    )
    show_parser.add_argument(
        "--gpus-per-node", type=int, default=1, help="GPUs per node (default: 1)."
    )
    show_parser.add_argument(
        "--framework",
        choices=["auto", "pytorch", "deepspeed", "horovod"],
        default="auto",
        help="Distributed framework (default: auto-detect).",
    )
    show_parser.add_argument(
        "--master-port",
        type=int,
        default=DEFAULT_MASTER_PORT,
        help=f"Master port (default: {DEFAULT_MASTER_PORT}).",
    )
    return parser


def handle(args: argparse.Namespace, config: Config) -> int:
    sub = getattr(args, "distributed_command", None)
    if sub != "show":
        console.print(
            "[bold]Distributed training is accessed via the submit command:[/bold]\n\n"
            "  koa submit script.slurm --distributed\n"
            "  koa submit script.slurm --distributed --nodes 4 --gpus-per-node 2\n"
            "  koa submit script.slurm --distributed --framework deepspeed\n\n"
            "Or use [bold]koa distributed show <script>[/bold] to preview the config.\n"
        )
        return 0

    script_path = args.script
    if not script_path.exists():
        console.print(f"[red]Script not found:[/red] {script_path}", file=sys.stderr)
        return 1

    dist_cfg = build_distributed_config(
        script_path,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        framework=args.framework,
        master_port=args.master_port,
    )

    if args.output_format == "json":
        emit_json(dist_cfg)
    else:
        display_distributed_config(dist_cfg)

    return 0
