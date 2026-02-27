"""koa env -- Environment snapshot, deployment, and comparison.

Subcommands:
    freeze   Capture the local Python environment to a lockfile.
    deploy   Deploy a frozen environment to the remote cluster.
    diff     Compare a local lockfile against the remote environment.
"""
from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.table import Table

from ..config import Config
from ..ssh import SSHError, copy_to_remote, run_ssh

from . import add_common_arguments, emit_json

console = Console()

DEFAULT_LOCKFILE = "koa-env.lock.yaml"

ENV_VARS_TO_CAPTURE = [
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CONDA_PREFIX",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_env_type() -> str:
    """Detect whether the active Python lives in conda, a venv, or is system-wide."""
    if os.environ.get("CONDA_PREFIX"):
        return "conda"
    if sys.prefix != sys.base_prefix:
        return "pip"
    return "pip"


def _get_local_packages(env_type: str) -> list[str]:
    """Return a sorted list of pinned package specs from the active environment."""
    if env_type == "conda":
        try:
            result = subprocess.run(
                ["conda", "list", "--export"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                lines = []
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        lines.append(line)
                return sorted(lines)
        except FileNotFoundError:
            pass

    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _get_cuda_version() -> Optional[str]:
    """Try to detect the local CUDA version."""
    for cmd in [["nvcc", "--version"], ["nvidia-smi"]]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                output = result.stdout
                # nvcc prints "release X.Y"
                for token in output.split():
                    if "." in token and token.replace(".", "").replace(",", "").isdigit():
                        return token.rstrip(",")
        except FileNotFoundError:
            continue
    return None


def _get_env_vars() -> dict[str, Optional[str]]:
    captured: dict[str, Optional[str]] = {}
    for var in ENV_VARS_TO_CAPTURE:
        val = os.environ.get(var)
        if val is not None:
            captured[var] = val
    return captured


def _load_lockfile(path: Path) -> dict:
    """Load and return a koa-env lockfile."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _remote_pip_freeze(config: Config) -> list[str]:
    """Run pip freeze on the remote cluster and return sorted package list."""
    result = run_ssh(
        config,
        ["bash", "-lc", "pip freeze 2>/dev/null || python3 -m pip freeze 2>/dev/null"],
        capture_output=True,
        check=False,
    )
    return sorted(
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip()
    )


def _parse_package_name_version(spec: str) -> tuple[str, str]:
    """Extract (name, version) from a pip-style spec like 'torch==2.2.0'."""
    for sep in ("==", ">=", "<=", "!=", "~="):
        if sep in spec:
            parts = spec.split(sep, 1)
            return parts[0].strip().lower(), parts[1].strip()
    return spec.strip().lower(), ""


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _handle_freeze(args, config: Config) -> int:
    """Capture the local Python environment to a YAML lockfile."""
    env_type = _detect_env_type()
    packages = _get_local_packages(env_type)
    python_version = platform.python_version()
    cuda_version = _get_cuda_version()
    os_info = platform.platform()
    env_vars = _get_env_vars()

    lockfile: dict = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "python_version": python_version,
        "cuda_version": cuda_version or "not detected",
        "os": os_info,
        "env_type": env_type,
        "packages": packages,
    }
    if env_vars:
        lockfile["env_vars"] = env_vars

    output_path = Path(args.output or DEFAULT_LOCKFILE)
    output_path.write_text(yaml.safe_dump(lockfile, sort_keys=False), encoding="utf-8")

    if args.output_format == "json":
        emit_json(lockfile)
        return 0

    console.print(f"[bold green]Environment frozen.[/bold green]")
    console.print(f"  Env type    : {env_type}")
    console.print(f"  Python      : {python_version}")
    console.print(f"  CUDA        : {cuda_version or 'not detected'}")
    console.print(f"  Packages    : {len(packages)}")
    console.print(f"  Output      : {output_path}")
    return 0


def _handle_deploy(args, config: Config) -> int:
    """Deploy a frozen environment to the remote cluster."""
    lockfile_path = Path(args.file or DEFAULT_LOCKFILE)
    if not lockfile_path.exists():
        console.print(
            f"[red]Lockfile not found: {lockfile_path}[/red]\n"
            "Run `koa env freeze` first.",
            file=sys.stderr,
        )
        return 1

    lock = _load_lockfile(lockfile_path)
    packages = lock.get("packages", [])
    if not packages:
        console.print("[red]No packages found in lockfile.[/red]", file=sys.stderr)
        return 1

    env_type = lock.get("env_type", "pip")

    console.print(
        f"[bold]Deploying {len(packages)} packages to remote cluster...[/bold]"
    )

    # Convert to requirements.txt format for pip
    req_lines = []
    for pkg in packages:
        # conda export format: name=version=build -> name==version
        if env_type == "conda" and "=" in pkg and "==" not in pkg:
            parts = pkg.split("=")
            if len(parts) >= 2:
                req_lines.append(f"{parts[0]}=={parts[1]}")
                continue
        req_lines.append(pkg)

    # Write a temporary requirements file and upload
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="koa-reqs-", delete=False
    ) as tmp:
        tmp.write("\n".join(req_lines) + "\n")
        tmp_path = Path(tmp.name)

    remote_reqs = Path(f"/tmp/koa-reqs-{os.getpid()}.txt")
    try:
        copy_to_remote(config, tmp_path, remote_reqs)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Install on remote
    install_cmd = (
        f"pip install -r {shlex.quote(str(remote_reqs))} "
        f"&& rm -f {shlex.quote(str(remote_reqs))}"
    )
    console.print("[dim]Running pip install on remote...[/dim]")
    try:
        result = run_ssh(
            config,
            ["bash", "-lc", install_cmd],
            capture_output=True,
            check=False,
        )
    except SSHError as exc:
        console.print(f"[red]Deploy failed: {exc}[/red]", file=sys.stderr)
        return 1

    if result.returncode != 0:
        console.print(
            f"[red]pip install failed (exit {result.returncode}).[/red]",
            file=sys.stderr,
        )
        if result.stderr:
            console.print(result.stderr, style="dim red")
        return 1

    # Verify: compare remote pip freeze with lockfile
    remote_pkgs = _remote_pip_freeze(config)
    local_map = dict(_parse_package_name_version(p) for p in packages)
    remote_map = dict(_parse_package_name_version(p) for p in remote_pkgs)

    mismatches = 0
    for name, local_ver in local_map.items():
        remote_ver = remote_map.get(name)
        if remote_ver is None or (local_ver and remote_ver != local_ver):
            mismatches += 1

    result_data = {
        "status": "deployed",
        "packages_requested": len(packages),
        "packages_installed": len(remote_map),
        "mismatches": mismatches,
    }

    if args.output_format == "json":
        emit_json(result_data)
        return 0

    console.print(f"[bold green]Deploy complete.[/bold green]")
    console.print(f"  Requested   : {len(packages)} packages")
    console.print(f"  Installed   : {len(remote_map)} packages")
    if mismatches:
        console.print(f"  [yellow]Mismatches  : {mismatches}[/yellow]")
    else:
        console.print(f"  Mismatches  : 0")
    return 0


def _handle_diff(args, config: Config) -> int:
    """Compare a local lockfile against the remote environment."""
    lockfile_path = Path(args.file or DEFAULT_LOCKFILE)
    if not lockfile_path.exists():
        console.print(
            f"[red]Lockfile not found: {lockfile_path}[/red]\n"
            "Run `koa env freeze` first.",
            file=sys.stderr,
        )
        return 1

    lock = _load_lockfile(lockfile_path)
    local_packages = lock.get("packages", [])
    local_map = dict(_parse_package_name_version(p) for p in local_packages)

    console.print("[dim]Fetching remote environment...[/dim]")
    remote_pkgs = _remote_pip_freeze(config)
    remote_map = dict(_parse_package_name_version(p) for p in remote_pkgs)

    all_names = sorted(set(local_map) | set(remote_map))

    rows: list[dict] = []
    n_match = 0
    n_mismatch = 0
    n_added = 0
    n_removed = 0

    for name in all_names:
        local_ver = local_map.get(name)
        remote_ver = remote_map.get(name)

        if local_ver is not None and remote_ver is not None:
            if local_ver == remote_ver:
                status = "match"
                n_match += 1
            else:
                status = "mismatch"
                n_mismatch += 1
        elif local_ver is not None:
            status = "added"
            n_added += 1
        else:
            status = "removed"
            n_removed += 1

        rows.append({
            "package": name,
            "local": local_ver or "",
            "remote": remote_ver or "",
            "status": status,
        })

    summary = {
        "matching": n_match,
        "mismatched": n_mismatch,
        "local_only": n_added,
        "remote_only": n_removed,
        "total": len(all_names),
    }

    if args.output_format == "json":
        emit_json({"packages": rows, "summary": summary})
        return 0

    # Show only non-matching rows by default (with a flag for all)
    display_rows = [r for r in rows if r["status"] != "match"] if not args.all else rows

    if not display_rows and not args.all:
        console.print("[bold green]Environments are in sync.[/bold green]")
        console.print(f"  {n_match} packages match.")
        return 0

    table = Table(title="Environment Diff")
    table.add_column("Package", style="cyan")
    table.add_column("Local", style="green")
    table.add_column("Remote", style="blue")
    table.add_column("Status")

    status_style = {
        "match": "[green]match[/green]",
        "mismatch": "[yellow]mismatch[/yellow]",
        "added": "[cyan]local only[/cyan]",
        "removed": "[red]remote only[/red]",
    }

    for row in display_rows:
        table.add_row(
            row["package"],
            row["local"],
            row["remote"],
            status_style.get(row["status"], row["status"]),
        )

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {n_match} matching, "
        f"{n_mismatch} mismatched, "
        f"{n_added} local only, "
        f"{n_removed} remote only"
    )
    return 0


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "env",
        help="Environment snapshot, deployment, and comparison.",
    )
    sub = parser.add_subparsers(dest="env_command", required=True)

    # --- freeze ---
    freeze_parser = sub.add_parser("freeze", help="Capture local environment to a lockfile.")
    freeze_parser.add_argument(
        "--output",
        default=None,
        help=f"Output lockfile path (default: {DEFAULT_LOCKFILE}).",
    )
    add_common_arguments(freeze_parser)

    # --- deploy ---
    deploy_parser = sub.add_parser("deploy", help="Deploy frozen environment to the remote cluster.")
    deploy_parser.add_argument(
        "--file",
        default=None,
        help=f"Path to the lockfile (default: {DEFAULT_LOCKFILE}).",
    )
    add_common_arguments(deploy_parser)

    # --- diff ---
    diff_parser = sub.add_parser("diff", help="Compare local lockfile against remote environment.")
    diff_parser.add_argument(
        "--file",
        default=None,
        help=f"Path to the lockfile (default: {DEFAULT_LOCKFILE}).",
    )
    diff_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all packages, including matching ones.",
    )
    add_common_arguments(diff_parser)

    add_common_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def handle(args, config: Config) -> int:
    cmd = args.env_command
    if cmd == "freeze":
        return _handle_freeze(args, config)
    if cmd == "deploy":
        return _handle_deploy(args, config)
    if cmd == "diff":
        return _handle_diff(args, config)

    console.print(
        "[red]Unknown env subcommand.[/red] Use: freeze, deploy, diff.",
        file=sys.stderr,
    )
    return 1
