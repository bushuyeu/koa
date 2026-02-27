"""KOA MCP Server -- exposes HPC cluster operations as AI-agent-callable tools."""
from __future__ import annotations

import json
import shlex
import sys
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .config import load_config, Config
from .ssh import run_ssh, SSHError
from .slurm import (
    list_jobs,
    queue_status,
    get_cluster_availability,
    cancel_job,
    get_available_gpus,
    select_best_gpu,
    get_job_io_paths,
)
from .runs import list_runs, show_run, sync_statuses

mcp = FastMCP("koa-mcp")


def _load_cfg() -> Config:
    """Load KOA configuration, raising a descriptive error on failure."""
    return load_config()


def _ssh(config: Config, cmd: str) -> str:
    """Run a remote command via SSH and return stdout."""
    result = run_ssh(config, ["bash", "-lc", cmd], capture_output=True)
    return (result.stdout or "").strip()


def _parse_pipe_table(raw: str, headers: list[str]) -> list[dict[str, str]]:
    """Parse pipe-delimited SLURM output into a list of dicts."""
    rows: list[dict[str, str]] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # Skip header lines
        if parts and parts[0].upper() in ("JOBID", "NODELIST", "NODE", "PARTITION"):
            continue
        row = {}
        for i, header in enumerate(headers):
            row[header] = parts[i] if i < len(parts) else ""
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Tool: koa_cluster_status
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_cluster_status() -> str:
    """Get cluster overview: GPU/node availability summary plus queue summary.

    Returns a JSON object with 'availability' (GPU counts by type and state)
    and 'queue_summary' (running/pending job counts).
    """
    config = _load_cfg()

    # Availability
    raw_avail = get_cluster_availability(config)
    from collections import defaultdict
    from .slurm import GPU_NAME_MAP

    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for line in raw_avail.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        gres_raw = parts[2].strip()
        state = parts[3].strip().rstrip("*").lower()
        if state in ("idle",):
            bucket = "idle"
        elif state in ("mixed", "mix", "allocated", "alloc"):
            bucket = "mixed"
        else:
            bucket = "down"
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
                count = int(count_str)
            except ValueError:
                continue
            summary[gpu_name][bucket] += count

    gpu_summary = {}
    for gpu_type in sorted(summary):
        buckets = summary[gpu_type]
        gpu_summary[gpu_type] = {
            "idle": buckets.get("idle", 0),
            "mixed": buckets.get("mixed", 0),
            "down": buckets.get("down", 0),
            "total": sum(buckets.values()),
        }

    # Queue summary
    raw_queue = queue_status(config)
    running = 0
    pending = 0
    for line in raw_queue.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        state = parts[3].strip() if len(parts) > 3 else ""
        if state == "RUNNING":
            running += 1
        elif state == "PENDING":
            pending += 1

    return json.dumps({
        "gpu_availability": gpu_summary,
        "queue_summary": {"running": running, "pending": pending},
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_jobs
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_jobs(format: str = "json") -> str:
    """List the current user's active SLURM jobs.

    Args:
        format: Output format, 'json' (default) or 'table'.
    """
    config = _load_cfg()
    raw = list_jobs(config)
    if not raw or not raw.strip():
        return json.dumps({"jobs": [], "message": "No active jobs."})

    headers = ["job_id", "name", "state", "time", "time_limit", "nodes", "nodelist_reason"]
    jobs = _parse_pipe_table(raw, headers)
    return json.dumps({"jobs": jobs}, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_queue
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_queue(partition: Optional[str] = None) -> str:
    """Show the full cluster queue.

    Args:
        partition: Filter to a specific partition (optional).
    """
    config = _load_cfg()
    raw = queue_status(config, partition=partition)
    if not raw or not raw.strip():
        return json.dumps({"queue": [], "message": "Queue is empty."})

    headers = ["job_id", "user", "name", "state", "time", "time_limit", "nodes", "cpus", "min_memory", "nodelist_reason"]
    queue = _parse_pipe_table(raw, headers)
    return json.dumps({"queue": queue, "count": len(queue)}, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_availability
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_availability(partition: Optional[str] = None) -> str:
    """Show GPU/node inventory across the cluster.

    Args:
        partition: Filter to a specific partition (optional).
    """
    config = _load_cfg()
    raw = get_cluster_availability(config, partition=partition)
    if not raw or not raw.strip():
        return json.dumps({"nodes": [], "message": "No node data available."})

    from .slurm import GPU_NAME_MAP

    nodes: list[dict] = []
    for line in raw.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) < 6:
            continue
        gres_raw = parts[2].strip()
        gpu_entries = []
        for entry in gres_raw.split(","):
            entry = entry.strip()
            if entry.startswith("gpu:"):
                segments = entry.split(":")
                if len(segments) >= 3:
                    gpu_name = GPU_NAME_MAP.get(segments[1].lower(), segments[1].lower())
                    count_str = segments[2].split("(")[0]
                    gpu_entries.append({"type": gpu_name, "count": int(count_str) if count_str.isdigit() else 0})

        nodes.append({
            "node": parts[0].strip(),
            "partition": parts[1].strip().rstrip("*"),
            "gpus": gpu_entries,
            "state": parts[3].strip().rstrip("*"),
            "cpus": parts[4].strip(),
            "memory": parts[5].strip(),
        })

    return json.dumps({"nodes": nodes, "count": len(nodes)}, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_cancel
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_cancel(job_id: str) -> str:
    """Cancel a SLURM job by ID.

    Args:
        job_id: The SLURM job ID to cancel.
    """
    config = _load_cfg()
    cancel_job(config, job_id)
    return json.dumps({"status": "cancelled", "job_id": job_id})


# ---------------------------------------------------------------------------
# Tool: koa_logs
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_logs(job_id: str, stream: str = "stdout", lines: int = 100) -> str:
    """Get the last N lines of a job's stdout or stderr log.

    Args:
        job_id: The SLURM job ID.
        stream: 'stdout' or 'stderr' (default: 'stdout').
        lines: Number of lines to retrieve (default: 100).
    """
    config = _load_cfg()
    io_paths = get_job_io_paths(config, job_id)
    target_path = io_paths.stdout if stream == "stdout" else io_paths.stderr

    if not target_path or target_path in ("UNKNOWN", "UNDEFINED"):
        return json.dumps({"error": f"No {stream} log path for job {job_id}."})

    quoted = shlex.quote(target_path)
    result = run_ssh(
        config,
        ["bash", "-lc", f"tail -n {max(1, lines)} {quoted}"],
        capture_output=True,
        check=False,
    )
    output = (result.stdout or "").strip()
    return json.dumps({
        "job_id": job_id,
        "stream": stream,
        "path": target_path,
        "lines": output.splitlines(),
        "line_count": len(output.splitlines()),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_runs_list
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_runs_list(limit: int = 20) -> str:
    """List recorded job runs from the local manifest store.

    Args:
        limit: Maximum number of runs to return (default: 20).
    """
    config = _load_cfg()
    runs = list_runs(config)
    limited = runs[:max(1, limit)]
    return json.dumps({"runs": limited, "total": len(runs)}, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool: koa_runs_show
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_runs_show(job_id: str) -> str:
    """Show details for a specific recorded run.

    Args:
        job_id: The SLURM job ID to look up.
    """
    config = _load_cfg()
    entry = show_run(config, job_id)
    if not entry:
        return json.dumps({"error": f"No run recorded with job ID {job_id}."})
    return json.dumps(entry, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool: koa_optimize
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_optimize(
    job_script: str,
    partitions: Optional[str] = None,
    gpu_types: Optional[str] = None,
) -> str:
    """Find the optimal GPU configuration for a job via dry-run scheduling.

    Uses 'sbatch --test-only' to simulate scheduling across GPU type and
    partition combinations. Returns results ranked by estimated start time.

    Args:
        job_script: Path to the job script (remote path on the cluster).
        partitions: Comma-separated partitions to test (optional, auto-detected).
        gpu_types: Comma-separated GPU types to test (optional, auto-detected).
    """
    import re
    from datetime import datetime

    config = _load_cfg()

    # Detect partitions
    if partitions:
        part_list = [p.strip() for p in partitions.split(",") if p.strip()]
    elif config.default_partition:
        part_list = [config.default_partition]
    else:
        try:
            result = run_ssh(
                config,
                ["sinfo", "--Format=partitionname", "--noheader"],
                capture_output=True,
            )
            part_list = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        except SSHError:
            part_list = ["kill-shared"]
    if not part_list:
        part_list = ["kill-shared"]

    # Detect GPU types
    if gpu_types:
        gpu_list = [g.strip() for g in gpu_types.split(",") if g.strip()]
    else:
        gpu_set: set[str] = set()
        for part in part_list:
            try:
                result = run_ssh(
                    config,
                    ["sinfo", "-p", part, "--Format=gres:40", "--noheader"],
                    capture_output=True,
                )
                for line in result.stdout.splitlines():
                    for entry in line.strip().split(","):
                        entry = entry.strip()
                        if entry.startswith("gpu:"):
                            segments = entry.split(":")
                            if len(segments) >= 2:
                                gpu_set.add(segments[1].lower())
            except SSHError:
                continue
        gpu_list = sorted(gpu_set) if gpu_set else ["rtx2080ti"]

    # Parse GPU count from script
    try:
        result = run_ssh(
            config,
            ["grep", "-oP", r"(?<=#SBATCH\s--gres=gpu)(?::[\w]+)?:(\d+)", job_script],
            capture_output=True,
            check=False,
        )
        gpu_count = 1
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().splitlines()[0]
            for p in reversed(line.strip(":").split(":")):
                if p.isdigit():
                    gpu_count = int(p)
                    break
    except SSHError:
        gpu_count = 1

    # Parse walltime from script
    try:
        result = run_ssh(
            config,
            ["grep", "-oP", r"(?<=#SBATCH\s--time=)\S+", job_script],
            capture_output=True,
            check=False,
        )
        walltime = "02:00:00"
        if result.returncode == 0 and result.stdout.strip():
            walltime = result.stdout.strip().splitlines()[0]
    except SSHError:
        walltime = "02:00:00"

    start_re = re.compile(r"to start at\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    now = datetime.now()
    results: list[dict] = []

    for gpu_type in gpu_list:
        for partition in part_list:
            gres = f"gpu:{gpu_type}:{gpu_count}"
            cmd = [
                "sbatch", "--test-only",
                f"--gres={gres}",
                f"--partition={partition}",
                f"--time={walltime}",
                job_script,
            ]
            try:
                result = run_ssh(config, cmd, capture_output=True, check=False)
                combined = (result.stdout or "") + "\n" + (result.stderr or "")
            except SSHError as exc:
                combined = str(exc)

            match = start_re.search(combined)
            entry: dict[str, Any] = {
                "gpu_type": gpu_type,
                "gpu_count": gpu_count,
                "partition": partition,
                "walltime": walltime,
                "gres": gres,
            }

            if match:
                try:
                    start_time = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
                    delta = start_time - now
                    wait_seconds = max(0, int(delta.total_seconds()))
                    entry["est_start"] = start_time.strftime("%Y-%m-%d %H:%M:%S")
                    entry["wait_seconds"] = wait_seconds
                    entry["error"] = None
                except ValueError:
                    entry["est_start"] = None
                    entry["wait_seconds"] = None
                    entry["error"] = "Failed to parse start time"
            else:
                error_msg = combined.strip()[:200] if combined.strip() else "No start time returned"
                entry["est_start"] = None
                entry["wait_seconds"] = None
                entry["error"] = error_msg

            results.append(entry)

    # Sort: viable first by wait_seconds, then errors
    results.sort(key=lambda r: (r["wait_seconds"] is None, r["wait_seconds"] or 0))

    for i, entry in enumerate(results, 1):
        entry["rank"] = i if entry["est_start"] else None

    recommendation = None
    viable = [r for r in results if r["est_start"]]
    if viable:
        best = viable[0]
        recommendation = (
            f"Use {best['gpu_type']} x{best['gpu_count']} on {best['partition']} "
            f"(estimated start: {best['est_start']})"
        )

    return json.dumps({
        "results": results,
        "recommendation": recommendation,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_why
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_why(job_id: str) -> str:
    """Explain why a job is pending. Decodes SLURM reason codes into plain English with actionable advice.

    Args:
        job_id: The SLURM job ID to investigate.
    """
    config = _load_cfg()

    reason_advice = {
        "Priority": "Job is waiting for higher-priority jobs to complete. Consider reducing resource requests or using a less-contended partition.",
        "Resources": "Requested resources are not currently available. Check 'koa availability' for idle GPUs or reduce GPU/memory requests.",
        "QOSMaxJobsPerUserLimit": "You have reached your QOS job limit. Wait for running jobs to finish or cancel unneeded ones.",
        "QOSMaxGRESPerUser": "You have reached your GPU quota. Cancel or reduce GPU requests on other jobs.",
        "AssocMaxJobsLimit": "Account job limit reached. Coordinate with your group to free slots.",
        "ReqNodeNotAvail": "Requested node(s) are down or reserved. Try a different partition or remove node constraints.",
        "PartitionNodeLimit": "Partition node limit exceeded. Request fewer nodes.",
        "PartitionTimeLimit": "Walltime exceeds partition maximum. Reduce --time or use a partition with higher limits.",
        "Dependency": "Job depends on another job that hasn't completed yet.",
        "JobHeldUser": "Job is held by user. Release with 'scontrol release <job_id>'.",
        "JobHeldAdmin": "Job is held by an administrator. Contact cluster support.",
        "BeginTime": "Job is scheduled to start at a future time.",
        "NonZeroExitCode": "A dependency job failed. Check dependency chain.",
        "BadConstraints": "Requested constraints cannot be satisfied. Check available features with 'sinfo -o \"%f\"'.",
    }

    try:
        result = run_ssh(
            config,
            ["scontrol", "show", "job", str(job_id)],
            capture_output=True,
        )
    except SSHError as exc:
        return json.dumps({"error": f"Failed to query job {job_id}: {exc}"})

    output = result.stdout or ""

    # Parse key fields
    fields: dict[str, str] = {}
    for line in output.splitlines():
        for token in line.strip().split():
            if "=" in token:
                key, _, value = token.partition("=")
                fields[key] = value

    state = fields.get("JobState", "UNKNOWN")
    reason = fields.get("Reason", "None")
    partition = fields.get("Partition", "")
    num_nodes = fields.get("NumNodes", "")
    num_cpus = fields.get("NumCPUs", "")
    tres_req = fields.get("ReqTRES", "")
    time_limit = fields.get("TimeLimit", "")
    submit_time = fields.get("SubmitTime", "")
    start_time = fields.get("StartTime", "")
    eligible_time = fields.get("EligibleTime", "")

    advice = reason_advice.get(reason, f"Reason code '{reason}' -- check SLURM documentation or ask cluster support.")

    return json.dumps({
        "job_id": job_id,
        "state": state,
        "reason": reason,
        "advice": advice,
        "details": {
            "partition": partition,
            "num_nodes": num_nodes,
            "num_cpus": num_cpus,
            "tres_requested": tres_req,
            "time_limit": time_limit,
            "submit_time": submit_time,
            "start_time": start_time,
            "eligible_time": eligible_time,
        },
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_audit
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_audit(days: int = 7, max_jobs: int = 20) -> str:
    """Audit recent job efficiency. Analyzes sacct history for memory, time, and CPU waste, and suggests right-sized resource requests.

    Args:
        days: Lookback period in days (default: 7).
        max_jobs: Maximum jobs to analyze (default: 20).
    """
    config = _load_cfg()
    days = int(max(1, days))
    max_jobs = int(max(1, max_jobs))

    sacct_cmd = (
        f"sacct -u {shlex.quote(config.user)} "
        f"--format=JobID,JobName%30,MaxRSS,ReqMem,Elapsed,Timelimit,AllocCPUS,TotalCPU,AllocTRES%60,State%20 "
        f"-P -n "
        f"--starttime=$(date -d '{days} days ago' +%Y-%m-%d 2>/dev/null || date -v-{days}d +%Y-%m-%d)"
    )

    try:
        result = run_ssh(config, ["bash", "-lc", sacct_cmd], capture_output=True)
    except SSHError as exc:
        return json.dumps({"error": f"Failed to query sacct: {exc}"})

    def _parse_mem_mb(value: str) -> Optional[float]:
        if not value or value in ("0", "", "0n"):
            return None
        value = value.strip().rstrip("nc")
        if not value:
            return None
        multipliers = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
        suffix = value[-1].upper()
        if suffix in multipliers:
            try:
                return float(value[:-1]) * multipliers[suffix]
            except ValueError:
                return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_time_s(t: str) -> Optional[float]:
        if not t or t in ("", "UNLIMITED", "Partition_Limit"):
            return None
        t = t.strip()
        d = 0
        if "-" in t:
            dp, t = t.split("-", 1)
            try:
                d = int(dp)
            except ValueError:
                return None
        parts = t.split(":")
        try:
            if len(parts) == 3:
                return d * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return d * 86400 + int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 1:
                return d * 86400 + float(parts[0])
        except ValueError:
            return None
        return None

    lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    jobs: list[dict] = []

    for line in lines:
        fields = line.split("|")
        if len(fields) < 10:
            continue
        job_id = fields[0].strip()
        if "." in job_id:
            continue
        state = fields[9].strip()
        if state in ("PENDING", "CANCELLED+", "CANCELLED"):
            continue

        max_rss_mb = _parse_mem_mb(fields[2].strip())
        req_mem_mb = _parse_mem_mb(fields[3].strip())
        elapsed_s = _parse_time_s(fields[4].strip())
        timelimit_s = _parse_time_s(fields[5].strip())
        total_cpu_s = _parse_time_s(fields[7].strip())
        try:
            alloc_cpus = int(fields[6].strip())
        except ValueError:
            alloc_cpus = 0

        mem_eff = min(max_rss_mb / req_mem_mb, 1.0) if max_rss_mb and req_mem_mb and req_mem_mb > 0 else None
        time_eff = min(elapsed_s / timelimit_s, 1.0) if elapsed_s is not None and timelimit_s and timelimit_s > 0 else None
        cpu_eff = min(total_cpu_s / (elapsed_s * alloc_cpus), 1.0) if total_cpu_s is not None and elapsed_s and alloc_cpus and elapsed_s > 0 else None

        suggested_mem = None
        if max_rss_mb and max_rss_mb > 0:
            val = max_rss_mb * 1.3
            suggested_mem = f"{val / 1024:.1f}G" if val >= 1024 else f"{val:.0f}M"

        suggested_time = None
        if elapsed_s is not None and elapsed_s > 0:
            val = elapsed_s * 1.3
            s = int(val)
            d, s = divmod(s, 86400)
            h, s = divmod(s, 3600)
            m, s = divmod(s, 60)
            suggested_time = f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"

        jobs.append({
            "job_id": job_id,
            "job_name": fields[1].strip(),
            "state": state,
            "mem_efficiency": round(mem_eff, 3) if mem_eff is not None else None,
            "time_efficiency": round(time_eff, 3) if time_eff is not None else None,
            "cpu_efficiency": round(cpu_eff, 3) if cpu_eff is not None else None,
            "suggested_mem": suggested_mem,
            "suggested_time": suggested_time,
        })

        if len(jobs) >= max_jobs:
            break

    # Compute averages
    mem_effs = [j["mem_efficiency"] for j in jobs if j["mem_efficiency"] is not None]
    time_effs = [j["time_efficiency"] for j in jobs if j["time_efficiency"] is not None]
    cpu_effs = [j["cpu_efficiency"] for j in jobs if j["cpu_efficiency"] is not None]

    return json.dumps({
        "jobs": jobs,
        "summary": {
            "job_count": len(jobs),
            "avg_mem_efficiency": round(sum(mem_effs) / len(mem_effs), 3) if mem_effs else None,
            "avg_time_efficiency": round(sum(time_effs) / len(time_effs), 3) if time_effs else None,
            "avg_cpu_efficiency": round(sum(cpu_effs) / len(cpu_effs), 3) if cpu_effs else None,
        },
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_limits
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_limits() -> str:
    """Show QOS limits, account associations, fair-share scores, and current usage."""
    config = _load_cfg()
    data: dict[str, Any] = {}

    # QOS limits
    try:
        result = run_ssh(
            config,
            ["sacctmgr", "show", "qos",
             "format=Name,MaxTRESPerUser,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxWall",
             "-P", "-n"],
            capture_output=True,
        )
        qos_rows = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) >= 5:
                qos_rows.append({
                    "name": parts[0],
                    "max_tres_per_user": parts[1],
                    "max_jobs_per_user": parts[2],
                    "max_submit_per_user": parts[3],
                    "max_wall": parts[4],
                })
        data["qos"] = qos_rows
    except SSHError as exc:
        data["qos_error"] = str(exc)

    # Account associations
    try:
        result = run_ssh(
            config,
            ["sacctmgr", "show", "assoc",
             f"where user={shlex.quote(config.user)}",
             "format=Account,User,QOS,MaxTRESPerUser,MaxJobs,MaxSubmit,GrpTRES",
             "-P", "-n"],
            capture_output=True,
        )
        assoc_rows = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) >= 7:
                assoc_rows.append({
                    "account": parts[0],
                    "user": parts[1],
                    "qos": parts[2],
                    "max_tres_per_user": parts[3],
                    "max_jobs": parts[4],
                    "max_submit": parts[5],
                    "grp_tres": parts[6],
                })
        data["associations"] = assoc_rows
    except SSHError as exc:
        data["associations_error"] = str(exc)

    # Fair-share
    try:
        result = run_ssh(
            config,
            ["sshare", "-u", config.user, "-l", "-P"],
            capture_output=True,
        )
        share_rows = []
        lines = result.stdout.strip().splitlines()
        for line in lines:
            parts = [p.strip() for p in line.strip().split("|")]
            # Skip header
            if parts and parts[0].lower() in ("account", "cluster"):
                continue
            if len(parts) >= 8:
                share_rows.append({
                    "account": parts[0],
                    "user": parts[1],
                    "raw_shares": parts[2],
                    "norm_shares": parts[3],
                    "raw_usage": parts[4],
                    "norm_usage": parts[5],
                    "effectiveness": parts[6],
                    "fairshare": parts[7],
                })
        data["fair_share"] = share_rows
    except SSHError as exc:
        data["fair_share_error"] = str(exc)

    # Current usage
    try:
        result = run_ssh(
            config,
            ["squeue", "-u", config.user, "-h", "-o", "%i|%T|%b"],
            capture_output=True,
        )
        running = 0
        pending = 0
        gpus = 0
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 3:
                continue
            if parts[1] == "RUNNING":
                running += 1
                for entry in parts[2].split(","):
                    entry = entry.strip()
                    if "gpu" in entry.lower():
                        segs = entry.split(":")
                        try:
                            gpus += int(segs[-1])
                        except (ValueError, IndexError):
                            gpus += 1
            elif parts[1] == "PENDING":
                pending += 1
        data["current_usage"] = {"running": running, "pending": pending, "gpus_in_use": gpus}
    except SSHError:
        data["current_usage"] = {"running": 0, "pending": 0, "gpus_in_use": 0}

    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_spy
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_spy(partition: Optional[str] = None) -> str:
    """Queue intelligence: queue depth, next GPUs to free, historical wait times, and partition overview.

    Args:
        partition: Filter to a specific partition (optional).
    """
    from datetime import datetime, timedelta

    config = _load_cfg()

    def _parse_pipe_rows(raw: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append([c.strip() for c in line.split("|")])
        return rows

    # 1. Queue depth per partition
    queue_depth: dict[str, dict[str, int]] = {}
    try:
        result = run_ssh(config, ["squeue", "-h", "-o", "%P|%T"], capture_output=True)
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 2:
                continue
            part = row[0].rstrip("*")
            state = row[1]
            if partition and part != partition:
                continue
            queue_depth.setdefault(part, {"RUNNING": 0, "PENDING": 0})
            if state == "RUNNING":
                queue_depth[part]["RUNNING"] += 1
            elif state == "PENDING":
                queue_depth[part]["PENDING"] += 1
    except SSHError:
        pass

    # 2. Next GPUs to free
    next_free: list[dict] = []
    try:
        result = run_ssh(
            config,
            ["squeue", "-t", "RUNNING", "--sort=e", "-o", "%N|%b|%e|%u|%j|%P", "-h"],
            capture_output=True,
        )
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 6:
                continue
            part = row[5].rstrip("*")
            if partition and part != partition:
                continue
            if not row[1] or "gpu" not in row[1].lower():
                continue
            # Compute time left
            time_left = "-"
            try:
                end_dt = datetime.strptime(row[2], "%Y-%m-%dT%H:%M:%S")
                delta = end_dt - datetime.now()
                if delta.total_seconds() < 0:
                    time_left = "ending"
                else:
                    total_secs = int(delta.total_seconds())
                    h, r = divmod(total_secs, 3600)
                    m, s = divmod(r, 60)
                    time_left = f"{h}:{m:02d}:{s:02d}"
            except (ValueError, TypeError):
                pass
            next_free.append({
                "node": row[0],
                "gpus": row[1],
                "end_time": row[2],
                "time_left": time_left,
                "user": row[3],
                "job_name": row[4],
                "partition": part,
            })
    except SSHError:
        pass

    # 3. Historical wait times
    wait_stats: dict[str, list[float]] = {}
    try:
        result = run_ssh(
            config,
            ["sacct", "-u", config.user,
             "--format=JobID,Submit,Start,Partition",
             "-P", "-n", "--starttime=now-7days"],
            capture_output=True,
        )
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 4:
                continue
            if "." in row[0]:
                continue
            part = row[3].rstrip("*")
            if partition and part != partition:
                continue
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    submit_dt = datetime.strptime(row[1].strip(), fmt)
                    start_dt = datetime.strptime(row[2].strip(), fmt)
                    if start_dt >= submit_dt:
                        wait_stats.setdefault(part, []).append((start_dt - submit_dt).total_seconds())
                    break
                except ValueError:
                    continue
    except SSHError:
        pass

    computed_waits: dict[str, dict] = {}
    for part_name, vals in wait_stats.items():
        s = sorted(vals)
        n = len(s)
        median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0 if n else 0
        computed_waits[part_name] = {
            "avg_seconds": sum(vals) / len(vals) if vals else 0,
            "median_seconds": median,
            "min_seconds": min(vals) if vals else 0,
            "max_seconds": max(vals) if vals else 0,
            "sample_size": len(vals),
        }

    # 4. Partition overview
    partition_info: list[dict] = []
    try:
        result = run_ssh(
            config,
            ["sinfo", "-o", "%P|%a|%l|%D|%G|%C", "--noheader"],
            capture_output=True,
        )
        for row in _parse_pipe_rows(result.stdout):
            if len(row) < 6:
                continue
            part_name = row[0].rstrip("*")
            if partition and part_name != partition:
                continue
            partition_info.append({
                "partition": part_name,
                "state": row[1],
                "max_wall": row[2],
                "nodes": row[3],
                "gpus": row[4],
                "cpus": row[5],
            })
    except SSHError:
        pass

    return json.dumps({
        "queue_depth": queue_depth,
        "next_gpus_to_free": next_free[:10],
        "historical_wait_times": computed_waits,
        "partition_overview": partition_info,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_priority
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_priority() -> str:
    """Show the user's fair-share priority breakdown, ranking among pending jobs, and estimated recovery."""
    config = _load_cfg()
    data: dict[str, Any] = {}

    # Priority factors for user's pending jobs
    try:
        result = run_ssh(
            config,
            ["sprio", "-u", config.user, "-l", "-h"],
            capture_output=True,
        )
        priorities = []
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 7:
                priorities.append({
                    "job_id": parts[0],
                    "user": parts[1],
                    "priority": parts[2],
                    "age": parts[3],
                    "fairshare": parts[4],
                    "job_size": parts[5],
                    "partition": parts[6],
                })
        data["priority_factors"] = priorities
    except SSHError as exc:
        data["priority_error"] = str(exc)

    # Fair-share score
    try:
        result = run_ssh(
            config,
            ["sshare", "-u", config.user, "-l", "-P"],
            capture_output=True,
        )
        shares = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.strip().split("|")]
            if parts and parts[0].lower() in ("account", "cluster"):
                continue
            if len(parts) >= 8:
                shares.append({
                    "account": parts[0],
                    "user": parts[1],
                    "raw_shares": parts[2],
                    "norm_shares": parts[3],
                    "raw_usage": parts[4],
                    "norm_usage": parts[5],
                    "effectiveness": parts[6],
                    "fairshare": parts[7],
                })
        data["fair_share"] = shares
    except SSHError as exc:
        data["fair_share_error"] = str(exc)

    # Ranking among all pending jobs
    try:
        result = run_ssh(
            config,
            ["sprio", "-l", "-h", "--sort=-Y"],
            capture_output=True,
        )
        all_pending = result.stdout.strip().splitlines()
        total_pending = len(all_pending)
        user_rank = None
        for i, line in enumerate(all_pending, 1):
            if config.user in line:
                user_rank = i
                break
        data["pending_ranking"] = {
            "total_pending_jobs": total_pending,
            "your_best_rank": user_rank,
        }
    except SSHError:
        data["pending_ranking"] = None

    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_efficiency
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_efficiency(job_id: str) -> str:
    """Check live GPU utilization for a running job via nvidia-smi over SSH.

    Args:
        job_id: The SLURM job ID of a running job.
    """
    config = _load_cfg()

    # Get the node the job is running on
    try:
        result = run_ssh(
            config,
            ["squeue", "-j", str(job_id), "-h", "-o", "%N|%T|%b|%M|%l"],
            capture_output=True,
        )
    except SSHError as exc:
        return json.dumps({"error": f"Failed to query job {job_id}: {exc}"})

    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return json.dumps({"error": f"Job {job_id} not found or not running."})

    parts = lines[0].split("|")
    node = parts[0].strip() if parts else ""
    state = parts[1].strip() if len(parts) > 1 else ""
    gres = parts[2].strip() if len(parts) > 2 else ""
    elapsed = parts[3].strip() if len(parts) > 3 else ""
    time_limit = parts[4].strip() if len(parts) > 4 else ""

    if state != "RUNNING":
        return json.dumps({
            "error": f"Job {job_id} is in state {state}, not RUNNING.",
            "state": state,
        })

    # Query nvidia-smi on the node
    gpu_data: list[dict] = []
    try:
        nvidia_cmd = (
            f"ssh {node} nvidia-smi "
            f"--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu "
            f"--format=csv,noheader,nounits 2>/dev/null"
        )
        result = run_ssh(config, ["bash", "-lc", nvidia_cmd], capture_output=True, check=False)
        for line in result.stdout.strip().splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) >= 7:
                gpu_data.append({
                    "index": fields[0],
                    "name": fields[1],
                    "gpu_util_pct": int(fields[2]) if fields[2].isdigit() else fields[2],
                    "mem_util_pct": int(fields[3]) if fields[3].isdigit() else fields[3],
                    "mem_used_mb": int(fields[4]) if fields[4].isdigit() else fields[4],
                    "mem_total_mb": int(fields[5]) if fields[5].isdigit() else fields[5],
                    "temperature_c": int(fields[6]) if fields[6].isdigit() else fields[6],
                })
    except SSHError:
        pass

    # Query sstat for CPU/memory
    sstat_data: dict = {}
    try:
        result = run_ssh(
            config,
            ["sstat", "-j", str(job_id), "--format=AveCPU,AveRSS,MaxRSS,AveVMSize", "-P", "-n"],
            capture_output=True,
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            fields = [f.strip() for f in line.split("|")]
            if len(fields) >= 4:
                sstat_data = {
                    "avg_cpu": fields[0],
                    "avg_rss": fields[1],
                    "max_rss": fields[2],
                    "avg_vmsize": fields[3],
                }
                break
    except SSHError:
        pass

    # Compute overall GPU utilization
    avg_gpu_util = None
    if gpu_data:
        utils = [g["gpu_util_pct"] for g in gpu_data if isinstance(g["gpu_util_pct"], int)]
        if utils:
            avg_gpu_util = sum(utils) / len(utils)

    return json.dumps({
        "job_id": job_id,
        "node": node,
        "state": state,
        "elapsed": elapsed,
        "time_limit": time_limit,
        "gres": gres,
        "gpus": gpu_data,
        "avg_gpu_utilization_pct": round(avg_gpu_util, 1) if avg_gpu_util is not None else None,
        "sstat": sstat_data,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_watch_once
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_watch_once(gpu_type: Optional[str] = None, partition: Optional[str] = None) -> str:
    """Check GPU availability once (non-blocking snapshot). Useful for monitoring without a persistent watch loop.

    Args:
        gpu_type: Filter for a specific GPU type (e.g. 'a100', 'h100').
        partition: Filter to a specific partition.
    """
    config = _load_cfg()
    available = get_available_gpus(config, partition=partition)

    if gpu_type:
        gpu_type_lower = gpu_type.lower()
        filtered = {k: v for k, v in available.items() if gpu_type_lower in k.lower()}
    else:
        filtered = available

    total_free = sum(filtered.values())
    best = select_best_gpu(config, partition=partition)

    from .commands import gpu_selection_json
    return json.dumps({
        "available_gpus": filtered,
        "total_free_gpus": total_free,
        "best_available": best,
        "gpu_selection": gpu_selection_json(config, best, 1, partition),
        "filter": {"gpu_type": gpu_type, "partition": partition},
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_submit
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_submit(
    job_script: str,
    partition: Optional[str] = None,
    time: Optional[str] = None,
    gpus: Optional[int] = None,
    desc: Optional[str] = None,
    gpu_type: Optional[str] = None,
    account: Optional[str] = None,
    constraint: Optional[str] = None,
    memory: Optional[str] = None,
    cpus: Optional[int] = None,
) -> str:
    """Submit a job script via sbatch. The script must exist locally.

    Args:
        job_script: Path to the local job script file.
        partition: SLURM partition (default: from config).
        time: Walltime (e.g. '04:00:00').
        gpus: Number of GPUs.
        desc: Job description for the run manifest.
        gpu_type: GPU type for --gres (e.g. 'a100'). Auto-selected if not specified.
        account: SLURM account.
        constraint: SLURM constraint (e.g. 'hopper').
        memory: Memory request (e.g. '32G').
        cpus: Number of CPUs per task.
    """
    from pathlib import Path

    config = _load_cfg()
    script_path = Path(job_script)

    if not script_path.exists():
        return json.dumps({"error": f"Job script not found: {job_script}"})

    sbatch_args: list[str] = []
    if partition:
        sbatch_args.extend(["--partition", partition])
    if time:
        sbatch_args.extend(["--time", time])
    if constraint:
        sbatch_args.extend(["--constraint", constraint])
    if account:
        sbatch_args.extend(["--account", account])
    elif config.default_account:
        sbatch_args.extend(["--account", config.default_account])
    if memory:
        sbatch_args.extend(["--mem", memory])
    if cpus:
        sbatch_args.extend(["--cpus-per-task", str(cpus)])

    # GPU selection
    from .slurm import parse_gpu_count_from_script
    from .commands import gpu_selection_json
    gpu_count = gpus or parse_gpu_count_from_script(script_path)
    gpu_selection_info = None
    if gpu_type:
        sbatch_args.append(f"--gres=gpu:{gpu_type}:{gpu_count}")
    elif not gpus:
        # Auto-select best GPU (queue-aware)
        target_partition = partition or config.default_partition
        best = select_best_gpu(config, partition=target_partition, min_gpus=gpu_count)
        sbatch_args.append(f"--gres=gpu:{best}:{gpu_count}")
        gpu_selection_info = gpu_selection_json(config, best, gpu_count, target_partition)
    else:
        sbatch_args.append(f"--gres=gpu:{gpu_count}")

    from .slurm import submit_job as _submit_job

    try:
        job_id = _submit_job(
            config,
            script_path,
            sbatch_args=sbatch_args,
            job_desc=desc,
        )
    except (SSHError, FileNotFoundError) as exc:
        return json.dumps({"error": f"Submission failed: {exc}"})

    result = {
        "status": "submitted",
        "job_id": job_id,
        "job_script": str(script_path),
        "sbatch_args": sbatch_args,
        "description": desc,
    }
    if gpu_selection_info:
        result["gpu_selection"] = gpu_selection_info
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_resubmit
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_resubmit(job_id: str) -> str:
    """Resubmit a previously recorded job using its stored manifest.

    Args:
        job_id: The job ID of the original submission to retry.
    """
    import re

    config = _load_cfg()
    entry = show_run(config, job_id)
    if not entry:
        return json.dumps({"error": f"No run recorded with job ID {job_id}."})

    sbatch_args: list[str] = entry.get("sbatch_args") or []
    manifest = entry.get("manifest") or entry.get("git") or {}

    # Recover script path
    job_script = manifest.get("job_script") or entry.get("job_script")
    if not job_script and sbatch_args:
        for candidate in reversed(sbatch_args):
            if not candidate.startswith("-"):
                job_script = candidate
                break

    if not job_script:
        return json.dumps({"error": f"Cannot determine job script for job {job_id}."})

    sbatch_cmd = ["sbatch"] + list(sbatch_args) + [job_script]

    try:
        result = run_ssh(config, sbatch_cmd, capture_output=True)
    except SSHError as exc:
        return json.dumps({"error": f"Submission failed: {exc}"})

    output = (result.stdout or "").strip()
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        return json.dumps({"error": f"Unable to parse job ID from: {output}"})

    new_job_id = match.group(1)

    # Record the new run
    try:
        from .runs import record_submission
        record_submission(
            config,
            job_id=new_job_id,
            sbatch_args=sbatch_args,
            manifest=manifest,
            local_job_dir=None,
            remote_job_dir=None,
            description=f"resubmit of {job_id}",
        )
    except Exception:
        pass

    return json.dumps({
        "status": "resubmitted",
        "old_job_id": job_id,
        "new_job_id": new_job_id,
        "sbatch_args": sbatch_args,
        "job_script": job_script,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_diagnose
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_diagnose(job_id: str) -> str:
    """Diagnose why a completed or failed job ended the way it did.

    Queries sacct for exit state and resource usage, reads the last 50 lines
    of stderr, and pattern-matches common failure signatures (OOM, timeout,
    node failure, CUDA OOM, missing modules, etc.) to produce an actionable
    diagnosis with fix suggestions.

    Args:
        job_id: The SLURM job ID to diagnose.
    """
    import math
    import re
    import shlex

    config = _load_cfg()

    # --- Query sacct ---
    sacct_fmt = (
        "JobID,State,ExitCode,MaxRSS,MaxVMSize,Elapsed,Timelimit,"
        "NodeList,Reason,ReqMem,AllocCPUS,AllocTRES%60"
    )
    try:
        result = run_ssh(
            config,
            ["sacct", "-j", str(job_id), f"--format={sacct_fmt}", "--parsable2", "--noheader"],
            capture_output=True,
        )
    except SSHError as exc:
        return json.dumps({"error": f"Failed to query sacct for job {job_id}: {exc}"})

    sacct = None
    for line in result.stdout.strip().splitlines():
        fields = line.strip().split("|")
        if len(fields) < 12:
            continue
        if "." in fields[0].strip():
            continue
        sacct = {
            "job_id": fields[0].strip(),
            "state": fields[1].strip(),
            "exit_code": fields[2].strip(),
            "max_rss_raw": fields[3].strip(),
            "elapsed": fields[5].strip(),
            "timelimit": fields[6].strip(),
            "node": fields[7].strip(),
            "reason": fields[8].strip(),
            "req_mem_raw": fields[9].strip(),
        }
        break

    if sacct is None:
        return json.dumps({"error": f"Job {job_id} not found in sacct."})

    # --- Read stderr ---
    stderr = ""
    try:
        io_paths = get_job_io_paths(config, job_id)
        if io_paths.stderr:
            quoted = shlex.quote(io_paths.stderr)
            r = run_ssh(
                config,
                ["bash", "-lc", f"tail -n 50 {quoted} 2>/dev/null"],
                capture_output=True,
                check=False,
            )
            stderr = r.stdout or ""
    except SSHError:
        pass

    state = sacct["state"]
    exit_code = sacct["exit_code"]

    # Parse peak memory
    mem_multipliers = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    max_rss_mb = None
    raw = sacct["max_rss_raw"].rstrip("nc")
    if raw and raw not in ("0", ""):
        suffix = raw[-1].upper()
        if suffix in mem_multipliers:
            try:
                max_rss_mb = float(raw[:-1]) * mem_multipliers[suffix]
            except ValueError:
                pass
        else:
            try:
                max_rss_mb = float(raw)
            except ValueError:
                pass

    diag = {
        "job_id": sacct["job_id"],
        "state": state,
        "exit_code": exit_code,
        "max_rss_mb": round(max_rss_mb, 1) if max_rss_mb else None,
        "elapsed": sacct["elapsed"],
        "timelimit": sacct["timelimit"],
        "node": sacct["node"],
        "diagnosis": "Unknown failure",
        "root_cause": None,
        "suggestion": None,
        "fix_command": None,
    }

    # --- Pattern matching ---
    if state == "COMPLETED" and exit_code == "0:0":
        diag["diagnosis"] = "Job completed successfully"
        diag["root_cause"] = "No failure detected"
        diag["suggestion"] = "No action needed."
    elif state == "OUT_OF_MEMORY" or exit_code == "0:137":
        diag["diagnosis"] = "Out of Memory (OOM killed)"
        diag["root_cause"] = "Job exceeded allocated RAM"
        if max_rss_mb and max_rss_mb > 0:
            suggested_mb = max_rss_mb * 1.3
            if suggested_mb >= 1024:
                diag["suggestion"] = f"Peak RSS was {max_rss_mb:.0f}M. Request at least {suggested_mb / 1024:.1f}G."
                diag["fix_command"] = f"--mem={math.ceil(suggested_mb / 1024)}G"
            else:
                diag["suggestion"] = f"Peak RSS was {max_rss_mb:.0f}M. Request at least {suggested_mb:.0f}M."
                diag["fix_command"] = f"--mem={math.ceil(suggested_mb)}M"
        else:
            diag["suggestion"] = "Increase --mem."
    elif state == "TIMEOUT":
        diag["diagnosis"] = "Walltime exceeded"
        diag["root_cause"] = "Job did not finish within the allocated time limit"
        diag["suggestion"] = f"Elapsed {sacct['elapsed']} hit limit {sacct['timelimit']}. Use --chain or increase --time."
        diag["fix_command"] = "koa submit --chain"
    elif state == "NODE_FAIL":
        node = sacct["node"]
        diag["diagnosis"] = "Node failure"
        diag["root_cause"] = f"Node {node} failed during execution"
        diag["suggestion"] = f"Exclude faulty node and resubmit."
        diag["fix_command"] = f"koa resubmit {sacct['job_id']} -- --exclude={node}"
    elif "CANCELLED" in state:
        diag["diagnosis"] = "Job was cancelled"
        diag["root_cause"] = f"State: {state}, reason: {sacct.get('reason', 'unknown')}"
        diag["suggestion"] = "Check if cancellation was intentional."
        diag["fix_command"] = f"koa resubmit {sacct['job_id']}"
    elif stderr and ("CUDA out of memory" in stderr or ("out of memory" in stderr.lower() and "cuda" in stderr.lower())):
        diag["diagnosis"] = "GPU VRAM exhausted"
        diag["root_cause"] = "Model or batch does not fit in GPU memory"
        diag["suggestion"] = "Reduce batch size, enable gradient checkpointing, use mixed precision, or request a GPU with more VRAM."
    elif stderr and "NCCL" in stderr and ("error" in stderr.lower() or "warn" in stderr.lower()):
        diag["diagnosis"] = "NCCL communication error"
        diag["root_cause"] = "Distributed training communication failure"
        diag["suggestion"] = "Set NCCL_DEBUG=INFO. Try NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1."
    elif stderr:
        mod_match = re.search(r"ModuleNotFoundError:\s*No module named '([^']+)'", stderr)
        path_match = re.search(r"No such file or directory:\s*['\"]?([^'\"\\n]+)", stderr)
        perm_match = re.search(r"Permission denied", stderr, re.IGNORECASE)
        if mod_match:
            module = mod_match.group(1)
            diag["diagnosis"] = f"Missing module: {module}"
            diag["root_cause"] = f"Module '{module}' not installed"
            diag["suggestion"] = f"pip install {module}"
            diag["fix_command"] = f"pip install {module}"
        elif path_match:
            missing = path_match.group(1).strip()
            diag["diagnosis"] = f"Missing file: {missing}"
            diag["root_cause"] = f"Path does not exist: {missing}"
            diag["suggestion"] = "Verify data was staged before submission."
        elif perm_match:
            diag["diagnosis"] = "Permission denied"
            diag["root_cause"] = "Insufficient filesystem permissions"
            diag["suggestion"] = "Check file permissions on the target path."
        elif "FAILED" in state:
            last_lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
            diag["diagnosis"] = f"Job failed with exit code {exit_code}"
            diag["root_cause"] = last_lines[-1][:200] if last_lines else "Unknown"
            diag["suggestion"] = "Check the full stderr log."
    elif "FAILED" in state:
        diag["diagnosis"] = f"Job failed with exit code {exit_code}"
        diag["root_cause"] = "Non-zero exit code"
        diag["suggestion"] = "Check the job's stderr log."

    return json.dumps(diag, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_validate
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_validate(
    script_path: str,
    partition: Optional[str] = None,
    gpus: Optional[int] = None,
) -> str:
    """Pre-flight validation of a SLURM job script.

    Parses #SBATCH directives, queries the cluster for partition/GPU/QOS info,
    and checks for common misconfigurations (wrong partition, GPU type, count,
    memory, walltime, missing GPU code, output path issues).

    Args:
        script_path: Local path to the job script to validate.
        partition: Override partition to validate against (optional).
        gpus: Override GPU count for validation (optional).
    """
    from pathlib import Path
    from .commands.validate import run_checks, _parse_sbatch_directives

    config = _load_cfg()
    p = Path(script_path)
    if not p.exists():
        return json.dumps({"error": f"Script not found: {script_path}"})

    script_text = p.read_text(encoding="utf-8")
    directives = _parse_sbatch_directives(script_text)
    results = run_checks(
        script_text,
        directives,
        config,
        partition_override=partition,
        gpus_override=gpus,
    )
    return json.dumps({"checks": results}, indent=2)



# ---------------------------------------------------------------------------
# Tool: koa_env_freeze
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_env_freeze() -> str:
    """Freeze the local Python environment to a YAML lockfile.

    Detects environment type (pip/conda), captures all installed packages,
    Python version, CUDA version, OS info, and relevant environment variables.
    Writes the result to koa-env.lock.yaml in the current directory.
    """
    from .commands.env import (
        _detect_env_type,
        _get_cuda_version,
        _get_env_vars,
        _get_local_packages,
    )
    import platform
    from datetime import datetime, timezone
    from pathlib import Path
    import yaml

    env_type = _detect_env_type()
    packages = _get_local_packages(env_type)
    python_version = platform.python_version()
    cuda_version = _get_cuda_version()
    env_vars = _get_env_vars()

    lockfile = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "python_version": python_version,
        "cuda_version": cuda_version or "not detected",
        "os": platform.platform(),
        "env_type": env_type,
        "packages": packages,
    }
    if env_vars:
        lockfile["env_vars"] = env_vars

    output_path = Path("koa-env.lock.yaml")
    output_path.write_text(yaml.safe_dump(lockfile, sort_keys=False), encoding="utf-8")

    return json.dumps({
        "status": "frozen",
        "env_type": env_type,
        "python_version": python_version,
        "cuda_version": cuda_version or "not detected",
        "package_count": len(packages),
        "output_file": str(output_path),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool: koa_env_diff
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_env_diff() -> str:
    """Compare the local koa-env.lock.yaml lockfile against the remote cluster environment.

    Reads the local lockfile and runs pip freeze on the remote cluster,
    then reports matching, mismatched, local-only, and remote-only packages.
    """
    from pathlib import Path
    import yaml
    from .commands.env import _parse_package_name_version, _remote_pip_freeze

    lockfile_path = Path("koa-env.lock.yaml")
    if not lockfile_path.exists():
        return json.dumps({"error": "No koa-env.lock.yaml found. Run koa_env_freeze first."})

    lock = yaml.safe_load(lockfile_path.read_text(encoding="utf-8")) or {}
    local_packages = lock.get("packages", [])
    local_map = dict(_parse_package_name_version(p) for p in local_packages)

    config = _load_cfg()
    remote_pkgs = _remote_pip_freeze(config)
    remote_map = dict(_parse_package_name_version(p) for p in remote_pkgs)

    all_names = sorted(set(local_map) | set(remote_map))
    rows = []
    n_match = n_mismatch = n_added = n_removed = 0

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
            status = "local_only"
            n_added += 1
        else:
            status = "remote_only"
            n_removed += 1
        rows.append({
            "package": name,
            "local": local_ver or "",
            "remote": remote_ver or "",
            "status": status,
        })

    return json.dumps({
        "packages": rows,
        "summary": {
            "matching": n_match,
            "mismatched": n_mismatch,
            "local_only": n_added,
            "remote_only": n_removed,
            "total": len(all_names),
        },
    }, indent=2)

# ---------------------------------------------------------------------------
# Tool: koa_anywhere
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_anywhere(script_path: str, time: str = "04:00:00") -> str:
    """Compare estimated start times across all configured backends.

    Probes every backend in the global config via sbatch --test-only and
    returns a ranked comparison so the agent can recommend the fastest cluster.

    Args:
        script_path: Path to the job script (remote path on each cluster).
        time: Walltime to simulate (default: 04:00:00).
    """
    from .commands.anywhere import _run_anywhere

    results, recommendation = _run_anywhere(
        script_path, time, config_path=None, output_format="json",
    )

    json_results = []
    for entry in results:
        out = dict(entry)
        out.pop("start_dt", None)
        json_results.append(out)

    return json.dumps({
        "backends": json_results,
        "recommendation": recommendation["backend"] if recommendation else None,
    }, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool: koa_budget
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_budget(days: int = 30) -> str:
    """Track GPU-hours consumption, burn rate, waste, and projected allocation exhaustion.

    Queries sacct for GPU-hour usage over the given period and sacctmgr for
    allocation limits. Returns a breakdown by job state and partition, daily
    burn rate, and projected exhaustion date if an allocation cap exists.

    Args:
        days: Lookback period in days (default: 30).
    """
    from datetime import datetime, timedelta

    config = _load_cfg()
    days = int(max(1, days))

    # 1. Query sacct for GPU-hours
    sacct_cmd = (
        f"sacct -u {shlex.quote(config.user)} "
        f"--format=JobID,JobName%30,Partition,State%20,Elapsed,AllocTRES%60,Start,End "
        f"-P -n "
        f"--starttime=$(date -d '{days} days ago' +%Y-%m-%d 2>/dev/null || date -v-{days}d +%Y-%m-%d)"
    )

    try:
        result = run_ssh(config, ["bash", "-lc", sacct_cmd], capture_output=True)
    except SSHError as exc:
        return json.dumps({"error": f"Failed to query sacct: {exc}"})

    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return json.dumps({"period_days": days, "total_gpu_hours": 0, "jobs": [], "message": "No jobs found."})

    # 2. Parse jobs
    jobs: list[dict] = []
    total_gpu_hours = 0.0
    by_state: dict[str, float] = {}
    by_partition: dict[str, float] = {}

    def _parse_time(t: str) -> Optional[float]:
        if not t or t in ("", "UNLIMITED", "Partition_Limit"):
            return None
        t = t.strip()
        d = 0
        if "-" in t:
            dp, t = t.split("-", 1)
            try:
                d = int(dp)
            except ValueError:
                return None
        parts = t.split(":")
        try:
            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            elif len(parts) == 2:
                h, m, s = 0, int(parts[0]), float(parts[1])
            elif len(parts) == 1:
                h, m, s = 0, 0, float(parts[0])
            else:
                return None
        except ValueError:
            return None
        return d * 86400 + h * 3600 + m * 60 + s

    def _gpu_count(tres: str) -> int:
        for entry in tres.split(","):
            entry = entry.strip()
            if entry.startswith("gres/gpu="):
                try:
                    return int(entry.split("=")[1])
                except (ValueError, IndexError):
                    pass
        return 0

    def _state_bucket(s: str) -> str:
        s = s.upper()
        if "COMPLETED" in s:
            return "COMPLETED"
        if "FAIL" in s or "TIMEOUT" in s:
            return "FAILED"
        if "CANCEL" in s:
            return "CANCELLED"
        if "RUNNING" in s:
            return "RUNNING"
        return "OTHER"

    for line in lines:
        fields = line.split("|")
        if len(fields) < 8:
            continue
        job_id = fields[0].strip()
        if "." in job_id:
            continue
        job_name = fields[1].strip()
        partition = fields[2].strip()
        state_raw = fields[3].strip()
        elapsed_raw = fields[4].strip()
        tres_raw = fields[5].strip()
        start_raw = fields[6].strip()
        end_raw = fields[7].strip()

        elapsed_s = _parse_time(elapsed_raw)
        if elapsed_s is None or elapsed_s <= 0:
            continue
        gpus = _gpu_count(tres_raw)
        if gpus <= 0:
            continue

        gpu_hours = (elapsed_s * gpus) / 3600.0
        bucket = _state_bucket(state_raw)

        total_gpu_hours += gpu_hours
        by_state[bucket] = by_state.get(bucket, 0.0) + gpu_hours
        by_partition[partition] = by_partition.get(partition, 0.0) + gpu_hours

        jobs.append({
            "job_id": job_id,
            "job_name": job_name,
            "partition": partition,
            "state": bucket,
            "gpu_count": gpus,
            "gpu_hours": round(gpu_hours, 2),
            "start": start_raw,
            "end": end_raw,
        })

    if not jobs:
        return json.dumps({"period_days": days, "total_gpu_hours": 0, "jobs": [], "message": "No GPU jobs found."})

    # 3. Query allocation limit
    allocation_limit: Optional[float] = None
    try:
        assoc_result = run_ssh(
            config,
            ["bash", "-lc",
             f"sacctmgr show assoc user={shlex.quote(config.user)} format=Account,GrpTRESMins --parsable2 --noheader"],
            capture_output=True,
        )
        for aline in assoc_result.stdout.strip().splitlines():
            aline = aline.strip()
            if not aline:
                continue
            parts = [p.strip() for p in aline.split("|")]
            if len(parts) >= 2:
                for entry in parts[1].split(","):
                    entry = entry.strip()
                    if "gpu" in entry.lower() and "=" in entry:
                        try:
                            mins = float(entry.split("=")[1])
                            limit_h = mins / 60.0
                            if allocation_limit is None or limit_h > allocation_limit:
                                allocation_limit = limit_h
                        except (ValueError, IndexError):
                            pass
    except SSHError:
        pass

    # 4. Compute metrics
    burn_rate = total_gpu_hours / days
    wasted = by_state.get("FAILED", 0.0) + by_state.get("CANCELLED", 0.0)

    projected_exhaustion: Optional[str] = None
    days_remaining: Optional[float] = None
    if allocation_limit is not None and burn_rate > 0:
        remaining = allocation_limit - total_gpu_hours
        if remaining > 0:
            days_remaining = remaining / burn_rate
            projected_exhaustion = (datetime.now() + timedelta(days=days_remaining)).strftime("%Y-%m-%d")
        else:
            days_remaining = 0.0
            projected_exhaustion = "EXHAUSTED"

    return json.dumps({
        "period_days": days,
        "total_gpu_hours": round(total_gpu_hours, 2),
        "by_state": {k: round(v, 2) for k, v in sorted(by_state.items())},
        "by_partition": {k: round(v, 2) for k, v in sorted(by_partition.items())},
        "burn_rate_per_day": round(burn_rate, 2),
        "wasted_gpu_hours": round(wasted, 2),
        "allocation_limit": round(allocation_limit, 2) if allocation_limit is not None else None,
        "projected_exhaustion": projected_exhaustion,
        "days_remaining": round(days_remaining, 1) if days_remaining is not None else None,
        "jobs": jobs,
    }, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool: koa_distributed_show
# ---------------------------------------------------------------------------


@mcp.tool()
def koa_distributed_show(
    script_path: str,
    nodes: int = 2,
    gpus_per_node: int = 1,
) -> str:
    """Show the distributed training configuration that would be injected for a script.

    Auto-detects the framework (PyTorch, DeepSpeed, Horovod) and returns the
    environment variables, sbatch flags, and launcher command that
    ``koa submit --distributed`` would inject.

    Args:
        script_path: Path to the training script (local).
        nodes: Number of nodes (default: 2).
        gpus_per_node: GPUs per node (default: 1).
    """
    from pathlib import Path as _Path
    from .commands.distributed import build_distributed_config

    script = _Path(script_path).expanduser()
    if not script.exists():
        return json.dumps({"error": f"Script not found: {script_path}"})

    cfg = build_distributed_config(
        script,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
    )
    return json.dumps(cfg, indent=2)


@mcp.tool()
def koa_jupyter(
    time: str = "04:00:00",
    gpus: int = 1,
    gpu_type: Optional[str] = None,
    partition: Optional[str] = None,
    mem: str = "16G",
    lab: bool = True,
    conda_env: Optional[str] = None,
) -> str:
    """Submit a Jupyter notebook server on a GPU compute node.

    Returns connection info (job ID, node, URL with token) as JSON.
    NOTE: The MCP server cannot hold a persistent SSH tunnel, so the caller
    must establish their own tunnel using the returned node and port info,
    or instruct the user to run ``koa jupyter`` from the CLI for the
    full automated experience.

    Args:
        time: Walltime limit (default: 04:00:00).
        gpus: Number of GPUs (default: 1).
        gpu_type: GPU type (default: auto-select best available).
        partition: Partition (default: from config).
        mem: Memory allocation (default: 16G).
        lab: True for jupyter lab, False for jupyter notebook.
        conda_env: Conda environment to activate before starting Jupyter.
    """
    import secrets
    import random

    config = _load_cfg()
    part = partition or config.default_partition or "kill-shared"

    # Auto-select GPU (queue-aware)
    from .commands import gpu_selection_json
    selected_gpu = gpu_type or select_best_gpu(config, part, min_gpus=gpus)

    # Generate token and random remote port
    token = secrets.token_hex(24)
    remote_port = random.randint(49152, 65535)

    # Build SLURM script
    jupyter_cmd = "jupyter lab" if lab else "jupyter notebook"
    conda_block = ""
    if conda_env:
        conda_block = (
            '\neval "$(conda shell.bash hook)"\n'
            f"conda activate {conda_env}\n"
        )

    script_content = (
        "#!/bin/bash\n"
        f"{conda_block}"
        f"{jupyter_cmd} \\\n"
        f"    --no-browser \\\n"
        f"    --ip=0.0.0.0 \\\n"
        f"    --port={remote_port} \\\n"
        f"    --ServerApp.token='{token}' \\\n"
        f"    --ServerApp.allow_origin='*'\n"
    )

    script_hex = secrets.token_hex(8)
    remote_script = f"/tmp/koa-jupyter-{script_hex}.sh"

    try:
        run_ssh(
            config,
            f"cat > {remote_script} << 'KOAEOF'\n{script_content}KOAEOF",
            capture_output=True,
        )
    except SSHError as exc:
        return json.dumps({"error": f"Failed to write remote script: {exc}"})

    # Submit
    gres = f"gpu:{selected_gpu}:{gpus}"
    sbatch_cmd = [
        "sbatch",
        "--partition", part,
        "--gres", gres,
        "--mem", mem,
        "--time", time,
        "--job-name", "koa-jupyter",
        "--output", f"/tmp/koa-jupyter-{script_hex}.log",
        remote_script,
    ]

    try:
        result = run_ssh(config, sbatch_cmd, capture_output=True)
    except SSHError as exc:
        return json.dumps({"error": f"sbatch failed: {exc}"})

    import re
    output = (result.stdout or "").strip()
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        return json.dumps({"error": f"Could not parse job ID: {output}"})

    job_id = match.group(1)

    result_data = {
        "job_id": job_id,
        "gpu_type": selected_gpu,
        "gpus": gpus,
        "partition": part,
        "remote_port": remote_port,
        "token": token,
        "remote_script": remote_script,
        "status": "submitted",
        "gpu_selection": gpu_selection_json(config, selected_gpu, gpus, part),
        "instructions": (
            f"Job {job_id} submitted. To connect:\n"
            f"1. Wait for the job to start: squeue -j {job_id}\n"
            f"2. Find the node: squeue -j {job_id} -h -o '%N'\n"
            f"3. Open SSH tunnel: ssh -N -L 8888:<node>:{remote_port} {config.login}\n"
            f"4. Open in browser: http://localhost:8888/?token={token}\n"
            f"\nOr use the CLI for automatic tunnel: koa jupyter"
        ),
    }
    return json.dumps(result_data, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
