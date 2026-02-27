from __future__ import annotations

import shlex
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence

from .config import Config
from .runs import list_all_runs
from .slurm import get_job_io_paths, list_jobs
from .ssh import SSHError, run_ssh

SACCT_COLUMNS = [
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "Submit",
    "Start",
    "End",
    "Elapsed",
    # "AllocGRES", # These don't work on Koa
    # "ReqGRES",
    "AllocTRES",
    "ReqTRES",
    "TRESUsageInTot",
    "TRESUsageInAve",
    "NodeList",
    "Partition",
    "Reason",
    "MaxRSS",
]

RECENT_HISTORY_LOOKBACK_DAYS = 1


@dataclass
class JobRecord:
    job_id: str
    job_name: Optional[str]
    status: Optional[str]
    reason: Optional[str]
    local_job_dir: Optional[str]
    remote_job_dir: Optional[str]
    submitted_at: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    elapsed: Optional[str]
    exit_code: Optional[str]
    partition: Optional[str]
    nodes: Optional[str]
    node_list: Optional[str]
    requested_gpus: Optional[str]
    allocated_gpus: Optional[str]
    alloc_gres: Optional[str]
    req_gres: Optional[str]
    alloc_tres: Dict[str, str]
    req_tres: Dict[str, str]
    tres_usage_tot: Dict[str, str]
    tres_usage_ave: Dict[str, str]
    sstat_usage: Dict[str, str]
    max_rss: Optional[str]
    gpu_summary: Optional[str]
    dashboard_url: Optional[str]
    sbatch_args: List[str]
    env_hashes: Dict[str, str]
    git: Optional[dict]
    expected_end: Optional[str] = None
    description: Optional[str] = None
    has_local_record: bool = False


def _batched(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in iterable:
        if not item:
            continue
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _parse_tres(value: Optional[str]) -> Dict[str, str]:
    if not value:
        return {}
    parts = [part.strip() for part in value.split(",") if part.strip()]
    data: Dict[str, str] = {}
    for part in parts:
        if "=" in part:
            key, raw = part.split("=", 1)
            data[key.strip()] = raw.strip()
    return data


def _parse_gres(value: Optional[str]) -> List[dict]:
    if not value:
        return []
    entries: List[dict] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        pieces = token.split(":")
        if not pieces:
            continue
        kind = pieces[0]
        model = pieces[1] if len(pieces) > 1 and pieces[1] else None
        count_value: Optional[int] = None
        if len(pieces) >= 2 and pieces[-1].isdigit():
            count_value = int(pieces[-1])
        entries.append(
            {
                "raw": token,
                "type": kind,
                "model": model if (model and (count_value is None or model != str(count_value))) else None,
                "count": count_value,
            }
        )
    return entries


def _summarize_gpus(gres: Optional[str]) -> Optional[str]:
    entries = _parse_gres(gres)
    gpu_entries = [entry for entry in entries if entry["type"].startswith("gpu")]
    if not gpu_entries:
        return None
    pieces = []
    for entry in gpu_entries:
        label = entry["model"] or entry["type"]
        if entry["count"]:
            label = f"{label} x{entry['count']}"
        pieces.append(label)
    return ", ".join(pieces) if pieces else None


def _summarize_gpus_from_tres(tres: Dict[str, str]) -> Optional[str]:
    if not tres:
        return None
    specific: list[str] = []
    generic: Optional[str] = None
    for key, raw in tres.items():
        if not key.startswith("gres/gpu"):
            continue
        _, _, model = key.partition(":")
        label = model or "GPU"
        count_text = raw or ""
        try:
            count_value = int(float(count_text)) if count_text else None
        except ValueError:
            count_value = None
        if count_value is None:
            label_str = label
        elif count_value <= 1:
            label_str = label
        else:
            label_str = f"{label} x{count_value}"
        if model:
            specific.append(label_str)
        else:
            generic = label_str
    if specific:
        return ", ".join(specific)
    return generic


def _canonical_host(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.split(".")[0].lower()


def _parse_pipe_table(raw: str, min_fields: int) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("JOBID") or line.startswith("HOSTNAMES"):
            continue
        parts = line.split("|")
        if len(parts) >= min_fields:
            rows.append(parts)
    return rows




def _parse_squeue(config: Config) -> Dict[str, dict]:
    output = list_jobs(config)
    entries: Dict[str, dict] = {}
    for parts in _parse_pipe_table(output, 7):
        job_id = parts[0].strip()
        if not job_id:
            continue
        entries[job_id] = {
            "job_id": job_id,
            "name": parts[1].strip(),
            "state": parts[2].strip(),
            "run_time": parts[3].strip(),
            "time_limit": parts[4].strip(),
            "nodes": parts[5].strip(),
            "reason": parts[6].strip(),
        }
    return entries


def _fetch_recent_sacct(config: Config) -> Dict[str, dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=RECENT_HISTORY_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    columns = SACCT_COLUMNS
    cmd = [
        "sacct",
        "-P",
        "-n",
        "-u",
        config.user,
        "-S",
        since,
        "-o",
        ",".join(columns),
    ]
    result = run_ssh(config, cmd, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return {}

    data: Dict[str, dict] = {}
    for parts in _parse_pipe_table(result.stdout, len(columns)):
        row = [part.strip() or None for part in parts[: len(columns)]]
        row_map = {columns[idx]: row[idx] for idx in range(len(row))}
        job_id = row_map.get("JobIDRaw") or ""
        if not job_id or "." in job_id:
            continue  # Skip per-step rows; keep top-level job id only.
        data[job_id] = {
            "job_id": job_id,
            "job_name": row_map.get("JobName"),
            "state": row_map.get("State"),
            "exit_code": row_map.get("ExitCode"),
            "submit": row_map.get("Submit"),
            "start": row_map.get("Start"),
            "end": row_map.get("End"),
            "elapsed": row_map.get("Elapsed"),
            "alloc_gres": row_map.get("AllocGRES"),
            "req_gres": row_map.get("ReqGRES"),
            "alloc_tres": _parse_tres(row_map.get("AllocTRES")),
            "req_tres": _parse_tres(row_map.get("ReqTRES")),
            "tres_usage_tot": _parse_tres(row_map.get("TRESUsageInTot")),
            "tres_usage_ave": _parse_tres(row_map.get("TRESUsageInAve")),
            "node_list": row_map.get("NodeList"),
            "partition": row_map.get("Partition"),
            "reason": row_map.get("Reason"),
            "max_rss": row_map.get("MaxRSS"),
        }
    return data


def _fetch_sstat(config: Config, job_ids: Iterable[str]) -> Dict[str, dict]:
    running = [job_id for job_id in job_ids if job_id]
    if not running:
        return {}
    data: Dict[str, dict] = {}
    for chunk in _batched(running, 20):
        job_spec = ",".join(f"{job_id}.batch" for job_id in chunk)
        cmd = [
            "sstat",
            "-P",
            "-n",
            "-j",
            job_spec,
            "-o",
            "JobID,MaxRSS,AveRSS,AveCPU,TRESUsageInAve,TRESUsageInMax",
        ]
        result = run_ssh(config, cmd, capture_output=True, check=False)
        if result.returncode != 0 or not result.stdout:
            continue
        for parts in _parse_pipe_table(result.stdout, 6):
            job_raw = parts[0].strip()
            if not job_raw:
                continue
            job_id = job_raw.split(".")[0]
            data[job_id] = {
                "job_id": job_id,
                "max_rss": parts[1].strip(),
                "ave_rss": parts[2].strip(),
                "ave_cpu": parts[3].strip(),
                "tres_usage_ave": _parse_tres(parts[4]),
                "tres_usage_max": _parse_tres(parts[5]),
            }
    return data


def _extract_partition_from_sbatch(args: Optional[Sequence[str]]) -> Optional[str]:
    if not args:
        return None
    for idx, arg in enumerate(args):
        if arg.startswith("--partition="):
            return arg.split("=", 1)[1]
        if arg == "--partition" and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith("-p") and len(arg) > 2:
            return arg[2:]
        if arg == "-p" and idx + 1 < len(args):
            return args[idx + 1]
    return None


def _infer_remote_dir(
    run_entry: Optional[dict],
) -> Optional[str]:
    if run_entry and run_entry.get("remote_job_dir"):
        return run_entry["remote_job_dir"]
    return None


def _read_local_log(path: Path, lines: int) -> Optional[str]:
    if not path.exists():
        return None
    buffer = deque(maxlen=lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            buffer.append(line.rstrip("\n"))
    return "\n".join(buffer)


def get_job_log_tail(
    config: Config,
    job_id: str,
    *,
    local_job_dir: Optional[str],
    remote_job_dir: Optional[str] = None,
    stream: str = "stdout",
    lines: int = 200,
) -> dict:
    """
    Return up to `lines` lines from stdout/stderr for a given job. Prefers the local mirror when present.
    """
    filename = "job.log" if stream == "stdout" else "job.err"
    if local_job_dir:
        local_path = Path(local_job_dir).expanduser() / filename
        content = _read_local_log(local_path, lines)
        if content is not None:
            return {
                "source": f"local:{local_path}",
                "content": content,
            }

    remote_path: Optional[str] = None
    if remote_job_dir:
        remote_path = str(PurePosixPath(remote_job_dir) / filename)

    if not remote_path:
        try:
            io_paths = get_job_io_paths(config, job_id)
        except SSHError as exc:
            raise RuntimeError(f"Unable to resolve remote log paths for job {job_id}: {exc}") from exc

        remote_path = io_paths.stdout if stream == "stdout" else io_paths.stderr
    if not remote_path:
        raise FileNotFoundError(f"No remote log path found for job {job_id} ({stream}).")

    quoted_path = shlex.quote(remote_path)
    result = run_ssh(
        config,
        ["bash", "-lc", f"tail -n {lines} {quoted_path}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read remote log ({remote_path}): {result.stderr.strip() if result.stderr else 'unknown error'}"
        )
    return {
        "source": f"remote:{remote_path}",
        "content": result.stdout.rstrip("\n"),
    }


def build_dashboard_url(config: Config, remote_path: Optional[str]) -> Optional[str]:
    if not config.dashboard_base_url or not remote_path:
        return None
    base = config.dashboard_base_url.rstrip("/")
    remote_clean = remote_path.lstrip("/")
    return f"{base}//{remote_clean}"


def collect_job_records(config: Config) -> List[JobRecord]:
    target_host = _canonical_host(config.host)
    runs = []
    for entry in list_all_runs(config):
        job_id = entry.get("job_id")
        if not job_id:
            continue
        entry_user = entry.get("user")
        entry_host = _canonical_host(entry.get("host"))
        if entry_user != config.user:
            continue
        if target_host and entry_host and entry_host != target_host:
            continue
        runs.append(entry)
    run_index = {entry["job_id"]: entry for entry in runs}

    squeue_details = _parse_squeue(config)
    sacct_details = _fetch_recent_sacct(config)

    job_ids = set(run_index.keys()) | set(squeue_details.keys())
    sacct_details = {job_id: sacct_details[job_id] for job_id in job_ids if job_id in sacct_details}

    running_ids: list[str] = []
    for job_id, item in squeue_details.items():
        state = (item.get("state") or "").upper()
        if state.startswith("RUN") or state == "COMPLETING" or state == "SUSPENDED":
            running_ids.append(job_id)
    sstat_details = _fetch_sstat(config, running_ids)

    def _build_record(job_id: str, run_entry: Optional[dict]) -> JobRecord:
        sacct_row = sacct_details.get(job_id, {}) or {}
        squeue_row = squeue_details.get(job_id, {}) or {}
        sstat_row = sstat_details.get(job_id, {}) or {}

        status = sacct_row.get("state") or squeue_row.get("state") or (run_entry.get("status") if run_entry else None)
        reason = sacct_row.get("reason") or squeue_row.get("reason")
        job_name = sacct_row.get("job_name") or squeue_row.get("name")
        local_dir = run_entry.get("local_job_dir") if run_entry else None
        remote_dir = _infer_remote_dir(run_entry)
        dashboard_url = build_dashboard_url(config, remote_dir)

        allocated_gres = sacct_row.get("alloc_gres")
        requested_gres = sacct_row.get("req_gres")
        gpu_summary = (
            _summarize_gpus(allocated_gres)
            or _summarize_gpus(requested_gres)
            or _summarize_gpus_from_tres(sacct_row.get("alloc_tres") or {})
            or _summarize_gpus_from_tres(sacct_row.get("req_tres") or {})
        )

        partition = sacct_row.get("partition")
        if not partition and run_entry:
            partition = _extract_partition_from_sbatch(run_entry.get("sbatch_args"))
        if not partition:
            partition = squeue_row.get("partition")

        submitted_at = (run_entry.get("submitted_at") if run_entry else None) or sacct_row.get("submit")
        started_at = sacct_row.get("start")
        ended_at = sacct_row.get("end")
        elapsed = sacct_row.get("elapsed") or squeue_row.get("run_time")

        alloc_tres = sacct_row.get("alloc_tres") or {}
        req_tres = sacct_row.get("req_tres") or {}

        return JobRecord(
            job_id=job_id,
            job_name=job_name,
            status=status,
            reason=reason,
            local_job_dir=local_dir,
            remote_job_dir=remote_dir,
            submitted_at=submitted_at,
            started_at=started_at,
            ended_at=ended_at,
            elapsed=elapsed,
            exit_code=sacct_row.get("exit_code"),
            partition=partition,
            nodes=squeue_row.get("nodes"),
            node_list=sacct_row.get("node_list"),
            requested_gpus=requested_gres,
            allocated_gpus=allocated_gres,
            alloc_gres=allocated_gres,
            req_gres=requested_gres,
            alloc_tres=alloc_tres,
            req_tres=req_tres,
            tres_usage_tot=sacct_row.get("tres_usage_tot") or {},
            tres_usage_ave=sacct_row.get("tres_usage_ave") or {},
            sstat_usage=sstat_row,
            max_rss=sacct_row.get("max_rss") or sstat_row.get("max_rss"),
            gpu_summary=gpu_summary,
            dashboard_url=dashboard_url,
            sbatch_args=run_entry.get("sbatch_args", []) if run_entry else [],
            env_hashes=run_entry.get("env_hashes", {}) if run_entry else {},
            git=run_entry.get("git") if run_entry else None,
            expected_end=None,
            description=run_entry.get("description") if run_entry else None,
            has_local_record=bool(run_entry),
        )

    job_records = [_build_record(job_id, run_index.get(job_id)) for job_id in job_ids]

    job_records.sort(key=lambda record: record.submitted_at or "", reverse=True)
    return job_records


def fetch_gpu_nodes(config: Config) -> List[dict]:
    result = run_ssh(
        config,
        ["sinfo", "-N", "-o", r"%N|%G|%T|%C|%P"],
        capture_output=True,
        check=False,
    )
    entries: List[dict] = []
    if result.returncode != 0 or not result.stdout:
        return entries
    for parts in _parse_pipe_table(result.stdout, 5):
        node = parts[0].strip()
        gres = parts[1].strip()
        state = parts[2].strip()
        cpu_block = parts[3].strip()
        partition = parts[4].strip()
        if "gpu" not in gres.lower():
            continue
        cpu_fields = [field.strip() for field in cpu_block.split("/") if field.strip()]
        alloc_cpu = idle_cpu = other_cpu = total_cpu = None
        if len(cpu_fields) == 4:
            alloc_cpu, idle_cpu, other_cpu, total_cpu = cpu_fields
        entries.append(
            {
                "node": node,
                "gres": gres,
                "state": state,
                "cpus": {
                    "allocated": alloc_cpu,
                    "idle": idle_cpu,
                    "other": other_cpu,
                    "total": total_cpu,
                },
                "partition": partition,
                "gpu_summary": _summarize_gpus(gres),
            }
        )
    return entries


def last_updated_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_record_to_dict(record: JobRecord) -> dict:
    return asdict(record)
