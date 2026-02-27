from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from .config import Config
from .ssh import copy_from_remote, run_ssh

RUN_INDEX_FILENAME = "runs.json"
RUN_LOOKBACK_HOURS = 48


def _index_path(local_results_dir: Path) -> Path:
    return local_results_dir / RUN_INDEX_FILENAME


def _load_index(local_results_dir: Path) -> Dict[str, dict]:
    path = _index_path(local_results_dir)
    if not path.exists():
        return {"version": 1, "runs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "runs": {}}
    if "runs" not in data or not isinstance(data["runs"], dict):
        data["runs"] = {}
    return data


def _save_index(local_results_dir: Path, data: Dict[str, dict]) -> None:
    path = _index_path(local_results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def record_submission(
    config: Config,
    *,
    job_id: str,
    sbatch_args: list[str],
    manifest: dict,
    local_job_dir: Optional[Path],
    remote_job_dir: Optional[Path],
    description: Optional[str] = None,
) -> None:
    if not config.local_results_dir:
        return
    local_results = config.local_results_dir.expanduser()
    index = _load_index(local_results)
    runs = index.setdefault("runs", {})

    env_hashes = manifest.get("env_hashes") or {}
    entry = {
        "job_id": job_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "user": config.user,
        "host": config.host,
        "status": "UNKNOWN",
        "sbatch_args": sbatch_args,
        "env_hashes": env_hashes,
        "remote_job_dir": str(remote_job_dir) if remote_job_dir else None,
        "local_job_dir": str(local_job_dir) if local_job_dir else None,
        "git": manifest.get("git"),
        "description": description,
    }
    runs[job_id] = entry
    _save_index(local_results, index)


def list_runs(config: Config) -> list[dict]:
    if not config.local_results_dir:
        return []
    index = _load_index(config.local_results_dir.expanduser())
    runs = index.get("runs", {})
    entries = list(runs.values())
    entries.sort(key=lambda item: item.get("submitted_at") or "", reverse=True)
    return entries


def list_all_runs(config: Config) -> list[dict]:
    """
    Aggregate run entries across every project under the configured local_root.
    """
    if not config.local_root:
        return []

    root = config.local_root.expanduser()
    projects_dir = root / "projects"
    if not projects_dir.exists():
        return []

    aggregated: list[dict] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RUN_LOOKBACK_HOURS)
    for project_dir in sorted(projects_dir.iterdir()):
        jobs_dir = project_dir / "jobs"
        if not jobs_dir.is_dir():
            continue
        index = _load_index(jobs_dir)
        runs = index.get("runs", {})
        for entry in runs.values():
            enriched = dict(entry)
            enriched.setdefault("project_name", project_dir.name)
            submitted_raw = enriched.get("submitted_at")
            if submitted_raw:
                try:
                    submitted_ts = datetime.fromisoformat(submitted_raw.replace("Z", "+00:00"))
                except ValueError:
                    submitted_ts = None
            else:
                submitted_ts = None
            if submitted_ts and submitted_ts < cutoff:
                continue
            aggregated.append(enriched)

    aggregated.sort(key=lambda item: item.get("submitted_at") or "", reverse=True)
    return aggregated


def show_run(config: Config, job_id: str) -> Optional[dict]:
    runs = list_runs(config)
    for entry in runs:
        if entry.get("job_id") == job_id:
            return entry
    return None


def _project_jobs_dirs(config: Config) -> list[Path]:
    dirs: list[Path] = []
    if config.local_root:
        projects_dir = config.local_root.expanduser() / "projects"
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                jobs_dir = project_dir / "jobs"
                if jobs_dir.is_dir():
                    dirs.append(jobs_dir)
    elif config.local_results_dir:
        dirs.append(config.local_results_dir.expanduser())
    return dirs


def _mutate_run_entry(config: Config, job_id: str, mutator) -> bool:
    for jobs_dir in _project_jobs_dirs(config):
        index = _load_index(jobs_dir)
        runs = index.get("runs", {})
        if job_id in runs:
            mutator(runs[job_id])
            _save_index(jobs_dir, index)
            return True
    return False


def set_run_description(config: Config, job_id: str, description: Optional[str]) -> bool:
    def _mutate(entry: dict) -> None:
        entry["description"] = description

    return _mutate_run_entry(config, job_id, _mutate)


def delete_run_entry(config: Config, job_id: str) -> bool:
    for jobs_dir in _project_jobs_dirs(config):
        index = _load_index(jobs_dir)
        runs = index.get("runs", {})
        if job_id in runs:
            runs.pop(job_id, None)
            _save_index(jobs_dir, index)
            return True
    return False


def _batched(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _squeue_states(config: Config, job_ids: Iterable[str]) -> Dict[str, str]:
    ids = list(job_ids)
    states: Dict[str, str] = {}
    for chunk in _batched(ids, 50):
        id_csv = ",".join(chunk)
        result = run_ssh(
            config,
            [
                "squeue",
                "-h",
                "-j",
                id_csv,
                "-o",
                r"%i|%T",
            ],
            capture_output=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                states[parts[0]] = parts[1]
    return states


def _sacct_states(config: Config, job_ids: Iterable[str]) -> Dict[str, str]:
    ids = list(job_ids)
    states: Dict[str, str] = {}
    for chunk in _batched(ids, 50):
        id_csv = ",".join(chunk)
        result = run_ssh(
            config,
            [
                "sacct",
                "-P",
                "-n",
                "-j",
                id_csv,
                "-o",
                "JobIDRaw,State",
            ],
            capture_output=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) == 2 and parts[0].isdigit():
                states[parts[0]] = parts[1]
    return states


def sync_statuses(config: Config) -> int:
    if not config.local_results_dir:
        return 0
    local_results = config.local_results_dir.expanduser()
    index = _load_index(local_results)
    runs = index.get("runs", {})
    if not runs:
        return 0

    job_ids = list(runs.keys())
    active_states = _squeue_states(config, job_ids)
    historical_states = _sacct_states(config, job_ids)
    updates = 0

    for job_id, entry in runs.items():
        state = active_states.get(job_id) or historical_states.get(job_id)
        if state and entry.get("status") != state:
            entry["status"] = state
            updates += 1

        if state and state.upper().startswith("COMPLET"):
            entry.setdefault("synced_at", None)
            if entry.get("synced_at") is None:
                remote_dir = entry.get("remote_job_dir")
                local_dir = entry.get("local_job_dir")
                if remote_dir and local_dir:
                    remote_path = Path(remote_dir)
                    local_path = Path(local_dir).expanduser()
                    local_path.mkdir(parents=True, exist_ok=True)
                    try:
                        copy_from_remote(
                            config,
                            remote_path,
                            local_path,
                            recursive=True,
                        )
                        entry["synced_at"] = datetime.now(timezone.utc).isoformat()
                    except Exception as exc:
                        entry.setdefault("sync_errors", []).append(str(exc))

    if updates:
        _save_index(local_results, index)
    return updates
