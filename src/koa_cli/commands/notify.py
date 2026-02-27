"""koa notify -- Job state change alerts via webhooks.

Runs a background poller that watches ``squeue`` for state transitions and
fires Slack / Discord webhook notifications.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

from ..config import Config
from ..ssh import SSHError, run_ssh

from . import add_common_arguments, emit_json

console = Console()

NOTIFY_CONFIG_DIR = Path("~/.config/koa").expanduser()
NOTIFY_CONFIG_PATH = NOTIFY_CONFIG_DIR / "notify.yaml"
NOTIFY_PID_PATH = NOTIFY_CONFIG_DIR / "notify.pid"

DEFAULT_WATCH_STATES = ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"]
DEFAULT_POLL_INTERVAL = 60


def _load_notify_config() -> dict:
    if not NOTIFY_CONFIG_PATH.exists():
        return {}
    try:
        return yaml.safe_load(NOTIFY_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_notify_config(data: dict) -> None:
    NOTIFY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTIFY_CONFIG_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    # Restrict permissions -- webhook URLs are sensitive
    NOTIFY_CONFIG_PATH.chmod(0o600)


def _send_webhook(url: str, webhook_type: str, message: str) -> bool:
    """Send a notification to a Slack or Discord webhook. Returns True on success."""
    if webhook_type == "slack":
        payload = json.dumps({"text": message}).encode("utf-8")
    elif webhook_type == "discord":
        payload = json.dumps({"content": message}).encode("utf-8")
    else:
        payload = json.dumps({"text": message}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status < 400
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def _get_user_jobs(config: Config) -> dict[str, dict]:
    """Query squeue for all jobs belonging to the configured user.
    Returns {job_id: {"name": ..., "state": ...}}.
    """
    result = run_ssh(
        config,
        [
            "squeue",
            "-u", config.user,
            "-h",
            "-o", r"%i|%j|%T",
        ],
        capture_output=True,
        check=False,
    )
    jobs: dict[str, dict] = {}
    for line in (result.stdout or "").splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) == 3:
            jobs[parts[0]] = {"name": parts[1], "state": parts[2]}
    return jobs


def _get_specific_jobs(config: Config, job_ids: list[str]) -> dict[str, dict]:
    """Query squeue for specific job IDs."""
    id_csv = ",".join(job_ids)
    result = run_ssh(
        config,
        [
            "squeue",
            "-h",
            "-j", id_csv,
            "-o", r"%i|%j|%T",
        ],
        capture_output=True,
        check=False,
    )
    jobs: dict[str, dict] = {}
    for line in (result.stdout or "").splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) == 3:
            jobs[parts[0]] = {"name": parts[1], "state": parts[2]}
    return jobs


def _poll_loop(
    config: Config,
    job_ids: Optional[list[str]],
    interval: int,
    webhooks: list[dict],
    watch_states: list[str],
) -> None:
    """Main polling loop. Detects state changes and fires webhooks."""
    previous_states: dict[str, str] = {}

    # Seed initial state
    if job_ids:
        current = _get_specific_jobs(config, job_ids)
    else:
        current = _get_user_jobs(config)

    for jid, info in current.items():
        previous_states[jid] = info["state"]

    console.print(
        f"[bold]Monitoring {len(previous_states)} job(s), "
        f"polling every {interval}s.[/bold] Press Ctrl+C to stop."
    )

    tracked_ids = set(job_ids) if job_ids else None
    vanished_checks: dict[str, int] = {}

    while True:
        time.sleep(interval)

        try:
            if tracked_ids:
                current = _get_specific_jobs(config, list(tracked_ids))
            else:
                current = _get_user_jobs(config)
        except SSHError:
            console.print("[yellow]SSH poll failed, retrying next cycle...[/yellow]")
            continue

        # Detect state changes
        all_ids = set(previous_states) | set(current)
        for jid in all_ids:
            old_state = previous_states.get(jid)
            new_info = current.get(jid)
            new_state = new_info["state"] if new_info else None
            job_name = new_info["name"] if new_info else "(unknown)"

            if new_state is None and old_state is not None:
                # Job vanished from squeue -- likely completed/failed/cancelled.
                # Give it one extra cycle to avoid false positives.
                vanished_checks[jid] = vanished_checks.get(jid, 0) + 1
                if vanished_checks[jid] < 2:
                    continue
                vanished_checks.pop(jid, None)
                inferred_state = "COMPLETED"
                if inferred_state.upper() in [s.upper() for s in watch_states]:
                    msg = f"KOA Job {jid} ({job_name}): {old_state} -> {inferred_state} (left queue)"
                    console.print(f"[bold yellow]{msg}[/bold yellow]")
                    for wh in webhooks:
                        _send_webhook(wh["url"], wh.get("type", "slack"), msg)
                previous_states.pop(jid, None)
                if tracked_ids:
                    tracked_ids.discard(jid)
                continue
            else:
                vanished_checks.pop(jid, None)

            if new_state and new_state != old_state:
                previous_states[jid] = new_state
                if new_state.upper() in [s.upper() for s in watch_states]:
                    msg = f"KOA Job {jid} ({job_name}): {old_state or 'NEW'} -> {new_state}"
                    console.print(f"[bold cyan]{msg}[/bold cyan]")
                    for wh in webhooks:
                        _send_webhook(wh["url"], wh.get("type", "slack"), msg)
            elif new_state and old_state is None:
                # Newly appeared job
                previous_states[jid] = new_state

        # Stop if tracking specific jobs and all are gone
        if tracked_ids and not tracked_ids:
            console.print("[bold green]All tracked jobs have finished.[/bold green]")
            break


def _handle_setup(args, config: Config) -> int:
    """Configure webhook URL interactively or via flags."""
    notify_config = _load_notify_config()

    if args.webhook_url:
        wh_type = args.webhook_type or "slack"
        webhooks = notify_config.get("webhooks", [])
        # Avoid duplicates
        existing_urls = {w["url"] for w in webhooks}
        if args.webhook_url not in existing_urls:
            webhooks.append({"url": args.webhook_url, "type": wh_type})
        notify_config["webhooks"] = webhooks
        notify_config.setdefault("watch_states", DEFAULT_WATCH_STATES)
        notify_config.setdefault("poll_interval", DEFAULT_POLL_INTERVAL)
        _save_notify_config(notify_config)
        console.print(
            f"[bold green]Added {wh_type} webhook.[/bold green]\n"
            f"  Config: {NOTIFY_CONFIG_PATH}"
        )
    else:
        # Show current config
        if notify_config:
            console.print("[bold]Current notify configuration:[/bold]")
            console.print(yaml.safe_dump(notify_config, sort_keys=False))
        else:
            console.print(
                "No notify configuration found.\n"
                "Use [cyan]koa notify setup --webhook-url URL[/cyan] to add a webhook."
            )

    if args.output_format == "json":
        emit_json(notify_config)

    return 0


def _handle_start(args, config: Config) -> int:
    """Start background polling for job state changes."""
    notify_config = _load_notify_config()
    webhooks = notify_config.get("webhooks", [])
    if not webhooks:
        console.print(
            "[red]No webhooks configured.[/red] Run `koa notify setup --webhook-url URL` first.",
            file=sys.stderr,
        )
        return 1

    watch_states = notify_config.get("watch_states", DEFAULT_WATCH_STATES)
    interval = args.interval or notify_config.get("poll_interval", DEFAULT_POLL_INTERVAL)

    job_ids: Optional[list[str]] = None
    if args.job_ids:
        job_ids = [jid.strip() for jid in args.job_ids.split(",") if jid.strip()]
    elif not args.all:
        console.print(
            "[red]Specify --job-ids or --all to start monitoring.[/red]",
            file=sys.stderr,
        )
        return 1

    if args.daemon:
        try:
            pid = os.fork()
        except OSError as exc:
            console.print(f"[red]Failed to fork daemon: {exc}[/red]", file=sys.stderr)
            return 1

        if pid > 0:
            # Parent: write PID and exit
            NOTIFY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            NOTIFY_PID_PATH.write_text(str(pid), encoding="utf-8")
            console.print(
                f"[bold green]Notify daemon started (PID {pid}).[/bold green]\n"
                f"  PID file: {NOTIFY_PID_PATH}\n"
                f"  Stop with: kill {pid}"
            )
            return 0
        else:
            # Child: detach and run poll loop
            os.setsid()
            # Redirect stdio to /dev/null
            devnull = os.open(os.devnull, os.O_RDWR)
            os.dup2(devnull, 0)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)

    # Write PID file for foreground mode too
    NOTIFY_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTIFY_PID_PATH.write_text(str(os.getpid()), encoding="utf-8")

    try:
        _poll_loop(config, job_ids, interval, webhooks, watch_states)
    except KeyboardInterrupt:
        console.print("\n[bold]Stopped monitoring.[/bold]")
    finally:
        if NOTIFY_PID_PATH.exists():
            try:
                NOTIFY_PID_PATH.unlink()
            except OSError:
                pass

    return 0


def _handle_status(args, config: Config) -> int:
    """Show the current notify daemon status."""
    notify_config = _load_notify_config()
    webhooks = notify_config.get("webhooks", [])

    info: dict = {
        "daemon_running": False,
        "pid": None,
        "webhook_count": len(webhooks),
        "config_path": str(NOTIFY_CONFIG_PATH),
    }

    if NOTIFY_PID_PATH.exists():
        try:
            pid = int(NOTIFY_PID_PATH.read_text(encoding="utf-8").strip())
            # Check if process is alive
            os.kill(pid, 0)
            info["daemon_running"] = True
            info["pid"] = pid
        except (ValueError, ProcessLookupError, PermissionError):
            # Stale PID file
            info["daemon_running"] = False

    if args.output_format == "json":
        emit_json(info)
        return 0

    if info["daemon_running"]:
        console.print(f"[bold green]Notify daemon is running (PID {info['pid']}).[/bold green]")
    else:
        console.print("[bold yellow]Notify daemon is not running.[/bold yellow]")

    console.print(f"  Webhooks configured: {info['webhook_count']}")
    console.print(f"  Config: {info['config_path']}")

    if webhooks:
        for wh in webhooks:
            url_preview = wh["url"][:50] + "..." if len(wh["url"]) > 50 else wh["url"]
            console.print(f"    - [{wh.get('type', 'unknown')}] {url_preview}")

    return 0


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "notify",
        help="Job state change alerts via webhooks (Slack, Discord).",
    )
    sub = parser.add_subparsers(dest="notify_command", required=True)

    # --- setup ---
    setup_parser = sub.add_parser("setup", help="Configure webhook URLs.")
    setup_parser.add_argument(
        "--webhook-url",
        help="Webhook URL to add (Slack or Discord).",
    )
    setup_parser.add_argument(
        "--webhook-type",
        choices=["slack", "discord"],
        default="slack",
        help="Type of webhook (default: slack).",
    )
    add_common_arguments(setup_parser)

    # --- start ---
    start_parser = sub.add_parser("start", help="Start polling for job state changes.")
    start_parser.add_argument(
        "--job-ids",
        help="Comma-separated job IDs to monitor.",
    )
    start_parser.add_argument(
        "--all",
        action="store_true",
        help="Monitor all jobs for the configured user.",
    )
    start_parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help=f"Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL}).",
    )
    start_parser.add_argument(
        "--daemon",
        action="store_true",
        help="Fork to background as a daemon process.",
    )
    add_common_arguments(start_parser)

    # --- status ---
    status_parser = sub.add_parser("status", help="Show current notify daemon status.")
    add_common_arguments(status_parser)

    add_common_arguments(parser)
    return parser


def handle(args, config: Config) -> int:
    cmd = args.notify_command
    if cmd == "setup":
        return _handle_setup(args, config)
    if cmd == "start":
        return _handle_start(args, config)
    if cmd == "status":
        return _handle_status(args, config)

    console.print(
        "[red]Unknown notify subcommand.[/red] Use: setup, start, status.",
        file=sys.stderr,
    )
    return 1
