"""koa sweep -- Parameter sweep via Slurm job arrays.

Takes a JSON/YAML parameter grid, generates a Slurm job array, and maps
``SLURM_ARRAY_TASK_ID`` to specific parameter combinations.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import tempfile
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, copy_to_remote, run_ssh

from . import add_common_arguments, emit_json

console = Console()

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")


def _load_params_file(path: Path) -> dict:
    """Load parameter grid from a JSON or YAML file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".json",):
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Params file must be a mapping, got {type(data).__name__}")
    return data


def _build_combinations(parameters: dict) -> list[dict]:
    """Generate cartesian product of all parameter values."""
    keys = sorted(parameters.keys())
    values_lists = [parameters[k] for k in keys]
    combos: list[dict] = []
    for combo in itertools.product(*values_lists):
        combos.append(dict(zip(keys, combo)))
    return combos


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "sweep",
        help="Submit a parameter sweep via Slurm job arrays.",
    )
    parser.add_argument(
        "job_script",
        type=Path,
        help="Path to the local job script.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        help="Path to parameter grid file (JSON or YAML).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show array config and parameter map without submitting.",
    )
    parser.add_argument(
        "--partition",
        help="Slurm partition to submit to.",
    )
    parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional raw sbatch arguments. Repeat for multiple flags.",
    )
    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    # Load parameter grid
    try:
        params_data = _load_params_file(args.params)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        console.print(f"[red]Failed to load params file:[/red] {exc}", file=sys.stderr)
        return 1

    parameters = params_data.get("parameters", params_data)
    if not isinstance(parameters, dict) or not parameters:
        console.print(
            "[red]Params file must contain a 'parameters' mapping with lists of values.[/red]",
            file=sys.stderr,
        )
        return 1

    # Validate that all values are lists
    for key, vals in parameters.items():
        if not isinstance(vals, list):
            console.print(
                f"[red]Parameter '{key}' must be a list, got {type(vals).__name__}.[/red]",
                file=sys.stderr,
            )
            return 1

    combinations = _build_combinations(parameters)
    n_tasks = len(combinations)

    if n_tasks == 0:
        console.print("[red]No parameter combinations generated.[/red]", file=sys.stderr)
        return 1

    # Build task-ID-to-params mapping
    task_map: dict[int, dict] = {}
    for idx, combo in enumerate(combinations):
        task_map[idx] = combo

    # Display the parameter grid
    param_keys = sorted(parameters.keys())
    table = Table(title=f"Parameter Sweep ({n_tasks} tasks)")
    table.add_column("Task ID", justify="right", style="bold")
    for key in param_keys:
        table.add_column(key, style="cyan")

    for idx in range(n_tasks):
        row = [str(idx)]
        for key in param_keys:
            row.append(str(task_map[idx][key]))
        table.add_row(*row)

    console.print(table)

    if args.output_format == "json" or args.dry_run:
        json_output = {
            "n_tasks": n_tasks,
            "parameters": parameters,
            "combinations": [
                {"task_id": idx, **combo} for idx, combo in task_map.items()
            ],
        }

    if args.dry_run:
        console.print(
            f"\n[bold]Dry-run:[/bold] Would submit [cyan]--array=0-{n_tasks - 1}[/cyan] "
            f"with {n_tasks} parameter combinations."
        )
        if args.output_format == "json":
            json_output["dry_run"] = True
            emit_json(json_output)
        return 0

    # Write params mapping to a temporary local file
    params_json_content = json.dumps(task_map, indent=2, default=str)

    job_script = Path(args.job_script)
    if not job_script.exists():
        console.print(f"[red]Job script not found: {job_script}[/red]", file=sys.stderr)
        return 1

    # Upload the job script
    from ..slurm import ensure_remote_workspace

    ensure_remote_workspace(config)
    remote_script = config.remote_code_dir / job_script.name
    try:
        copy_to_remote(config, job_script, remote_script)
    except SSHError as exc:
        console.print(f"[red]Failed to upload job script:[/red] {exc}", file=sys.stderr)
        return 1

    # Upload the params mapping file
    remote_params_file = config.remote_code_dir / "koa_sweep_params.json"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="koa-sweep-"
    ) as tmp:
        tmp.write(params_json_content)
        tmp_path = Path(tmp.name)

    try:
        copy_to_remote(config, tmp_path, remote_params_file)
    except SSHError as exc:
        console.print(f"[red]Failed to upload params file:[/red] {exc}", file=sys.stderr)
        return 1
    finally:
        tmp_path.unlink(missing_ok=True)

    # Build sbatch command
    sbatch_cmd: list[str] = [
        "sbatch",
        f"--array=0-{n_tasks - 1}",
        f"--export=ALL,KOA_SWEEP_PARAMS_FILE={remote_params_file}",
    ]

    partition = args.partition or config.default_partition
    if partition:
        sbatch_cmd.extend(["--partition", partition])

    if config.default_account:
        sbatch_cmd.extend(["--account", config.default_account])

    sbatch_cmd.extend(args.sbatch_arg or [])
    sbatch_cmd.append(str(remote_script))

    # Submit
    try:
        result = run_ssh(config, sbatch_cmd, capture_output=True)
    except SSHError as exc:
        console.print(f"[red]Submission failed:[/red] {exc}", file=sys.stderr)
        return 1

    output = result.stdout.strip() if result.stdout else ""
    match = SBATCH_JOB_ID_PATTERN.search(output)
    if not match:
        console.print(
            f"[red]Unable to parse job ID from sbatch output:[/red] {output}",
            file=sys.stderr,
        )
        return 1

    array_job_id = match.group(1)

    console.print(
        f"\n[bold green]Sweep submitted![/bold green] "
        f"Array job ID: [cyan]{array_job_id}[/cyan] "
        f"({n_tasks} tasks: 0-{n_tasks - 1})"
    )
    console.print(
        f"  Params file on cluster: {remote_params_file}\n"
        f"  Your job script should read KOA_SWEEP_PARAMS_FILE and index by SLURM_ARRAY_TASK_ID."
    )

    if args.output_format == "json":
        json_output["array_job_id"] = array_job_id
        json_output["remote_params_file"] = str(remote_params_file)
        emit_json(json_output)

    return 0
