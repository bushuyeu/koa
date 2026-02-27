from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import Config
from .ssh import SSHError, copy_to_remote, run_ssh

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")
DEFAULT_PARTITION = "kill-shared"

# GPU priority ranking — higher score = more desirable
GPU_PRIORITY: Dict[str, int] = {
    "h200": 110,
    "h100": 100,
    "a100": 90,
    "a40": 80,
    "l40": 75,
    "a30": 70,
    "v100": 60,
    "rtx_a6000": 55,
    "rtx_a5000": 50,
    "rtx2080ti": 40,
    "t4": 30,
}

# Map detected GPU names from sinfo to SLURM GRES names
GPU_NAME_MAP: Dict[str, str] = {
    "h200": "h200",
    "h100": "h100",
    "a100": "a100",
    "a40": "a40",
    "l40": "l40",
    "a30": "a30",
    "v100": "v100",
    "v100s": "v100",
    "rtx_a6000": "rtx_a6000",
    "rtx_a5000": "rtx_a5000",
    "rtx2080ti": "rtx2080ti",
    "rtx_2080_ti": "rtx2080ti",
    "t4": "t4",
}


@dataclass
class JobIOPaths:
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def _has_partition_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg in {"--partition", "-p"}:
            return True
        if arg.startswith("--partition="):
            return True
        if arg.startswith("-p") and arg != "-p":
            return True
    return False


def _has_output_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg == "--output":
            return True
        if arg.startswith("--output="):
            return True
    return False


def _has_error_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg == "--error":
            return True
        if arg.startswith("--error="):
            return True
    return False


def ensure_remote_workspace(config: Config) -> None:
    run_ssh(config, ["mkdir", "-p", str(config.remote_code_dir)])
    if config.remote_results_dir:
        run_ssh(config, ["mkdir", "-p", str(config.remote_results_dir)])
    if config.shared_env_dir:
        run_ssh(config, ["mkdir", "-p", str(config.shared_env_dir)])


def submit_job(
    config: Config,
    local_job_script: Path,
    *,
    sbatch_args: Optional[Iterable[str]] = None,
    script_sbatch_args: Optional[Iterable[str]] = None,
    remote_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
    job_desc: Optional[str] = None,
) -> str:
    if not local_job_script.exists():
        raise FileNotFoundError(f"Job script not found: {local_job_script}")

    ensure_remote_workspace(config)

    remote_script = config.remote_code_dir / (remote_name or local_job_script.name)
    copy_to_remote(config, local_job_script, remote_script)

    env_vars: list[str] = [
        f"KOA_ML_CODE_ROOT={config.remote_code_dir}",
        f"KOA_BACKEND={config.cluster_name}",
        f"KOA_REMOTE_ROOT={config.remote_root}",
    ]
    if config.cuda_minor_version:
        env_vars.append(f"CUDA_MINOR_VERSION={config.cuda_minor_version}")
    remote_results_root: Optional[Path] = config.remote_results_dir
    if remote_results_root:
        env_vars.append(f"KOA_ML_RESULTS_ROOT={remote_results_root}")
    if config.remote_project_root:
        env_vars.append(f"KOA_PROJECT_ROOT={config.remote_project_root}")
    if config.shared_env_dir:
        env_vars.append(f"KOA_SHARED_ENV={config.shared_env_dir}")

    run_dir_str: Optional[str] = None
    if run_dir:
        run_dir_str = str(run_dir)
        env_vars.append(f"KOA_RUN_DIR={run_dir_str}")
        env_vars.append(f"KOA_RUN_METADATA_DIR={run_dir_str}/run_metadata")
    if job_desc:
        env_vars.append(f"KOA_JOB_DESC={job_desc}")

    args = ["env", *env_vars, "sbatch"]
    sbatch_args_list = list(sbatch_args or [])
    script_sbatch_args_list = list(script_sbatch_args or [])

    default_partition = config.default_partition or DEFAULT_PARTITION
    if not _has_partition_flag(sbatch_args_list) and not _has_partition_flag(script_sbatch_args_list):
        args.extend(["--partition", default_partition])

    if run_dir_str:
        if not _has_output_flag(sbatch_args_list):
            args.extend(["--output", f"{run_dir_str}/job.log"])
        if not _has_error_flag(sbatch_args_list):
            args.extend(["--error", f"{run_dir_str}/job.err"])
    elif remote_results_root:
        results_root = str(remote_results_root)
        if not _has_output_flag(sbatch_args_list):
            args.extend(["--output", f"{results_root}/%j/job.log"])
        if not _has_error_flag(sbatch_args_list):
            args.extend(["--error", f"{results_root}/%j/job.err"])

    if sbatch_args_list:
        args.extend(sbatch_args_list)

    args.append(str(remote_script))

    result = run_ssh(config, args, capture_output=True)
    output = result.stdout.strip() if result.stdout else ""
    match = SBATCH_JOB_ID_PATTERN.search(output)
    if not match:
        raise SSHError(f"Unable to parse sbatch output for job id: {output}")
    return match.group(1)


def cancel_job(config: Config, job_id: str) -> None:
    run_ssh(config, ["scancel", job_id])


def list_jobs(config: Config) -> str:
    result = run_ssh(
        config,
        [
            "squeue",
            "-u",
            config.user,
            "-o",
            r"%i|%j|%T|%M|%l|%D|%R",
        ],
        capture_output=True,
    )
    return result.stdout


def run_health_checks(config: Config) -> str:
    result = run_ssh(
        config,
        [
            "bash",
            "-lc",
            (
                "set -euo pipefail;"
                "echo '== hostname =='; hostname;"
                "echo '== sinfo =='; sinfo -o '%P %a %l %D %G %m'"
            ),
        ],
        capture_output=True,
    )
    return result.stdout


def get_job_io_paths(config: Config, job_id: str) -> JobIOPaths:
    """Return the stdout/stderr paths configured for the given job."""
    result = run_ssh(
        config,
        ["scontrol", "show", "job", str(job_id)],
        capture_output=True,
    )
    stdout_path = None
    stderr_path = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("StdOut="):
            stdout_path = line.split("=", 1)[1] or None
        elif line.startswith("StdErr="):
            stderr_path = line.split("=", 1)[1] or None
    return JobIOPaths(stdout=stdout_path, stderr=stderr_path)


def get_available_gpus(config: Config, partition: Optional[str] = None) -> Dict[str, int]:
    """Query sinfo for available GPUs on idle/mix nodes. Returns {gpu_type: count}."""
    part = partition or config.default_partition or DEFAULT_PARTITION
    result = run_ssh(
        config,
        [
            "sinfo",
            "-p", part,
            "--Format=nodehost,gres:30,statecompact",
            "--noheader",
        ],
        capture_output=True,
    )

    available: Dict[str, int] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        gres_field = parts[1].strip()
        state = parts[2].strip().lower().rstrip("*")
        if state not in ("idle", "mix", "mixed"):
            continue
        # Parse gres like "gpu:h100:4" or "gpu:a100:2(S:0-1)"
        for gres_entry in gres_field.split(","):
            gres_entry = gres_entry.strip()
            if not gres_entry.startswith("gpu:"):
                continue
            gres_parts = gres_entry.split(":")
            if len(gres_parts) < 3:
                continue
            gpu_name = gres_parts[1].lower()
            count_str = gres_parts[2].split("(")[0]
            try:
                count = int(count_str)
            except ValueError:
                continue
            normalized = GPU_NAME_MAP.get(gpu_name, gpu_name)
            available[normalized] = available.get(normalized, 0) + count

    return available


def select_best_gpu(config: Config, partition: Optional[str] = None) -> str:
    """Select the highest-priority available GPU on the given partition.

    Returns the GRES GPU type name (e.g. 'h100', 'a100').
    Falls back to 'rtx2080ti' if nothing is detected.
    """
    available = get_available_gpus(config, partition)
    if not available:
        return "rtx2080ti"

    best_type = max(available, key=lambda g: GPU_PRIORITY.get(g, 0))
    return best_type


def parse_gpu_count_from_script(script_path: Path) -> int:
    """Extract the GPU count from #SBATCH --gres=gpu:N directives in a script."""
    try:
        text = script_path.read_text(encoding="utf-8")
    except Exception:
        return 1

    pattern = re.compile(r"#SBATCH\s+--gres=gpu(?::[\w]+)?:(\d+)")
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if match:
            return int(match.group(1))
    return 1


def queue_status(config: Config, partition: Optional[str] = None) -> str:
    """Get the full cluster queue. Returns raw pipe-delimited squeue output."""
    cmd: List[str] = [
        "squeue",
        "-o", r"%i|%u|%j|%T|%M|%l|%D|%C|%m|%R",
        "--sort=P,t,-p",
    ]
    if partition:
        cmd.extend(["-p", partition])

    result = run_ssh(config, cmd, capture_output=True)
    return result.stdout


def get_cluster_availability(config: Config, partition: Optional[str] = None) -> str:
    """Query sinfo for per-node GPU inventory. Returns raw pipe-delimited output."""
    cmd: List[str] = [
        "sinfo",
        "-N",
        "-o", "%N|%P|%G|%T|%C|%m",
        "--noheader",
    ]
    if partition:
        cmd.extend(["-p", partition])

    result = run_ssh(config, cmd, capture_output=True)
    return result.stdout
