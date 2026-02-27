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
# Based on actual KOA cluster inventory (sinfo output)
GPU_PRIORITY: Dict[str, int] = {
    "nvidia_h200_nvl": 110,       # Hopper, 141GB HBM3e
    "nvidia_h100_nvl": 105,       # Hopper, 94GB HBM3
    "nvidia_h100_pcie": 100,      # Hopper, 80GB HBM3
    "NV-H100": 100,               # Hopper
    "NV-L40": 85,                 # Ada Lovelace, 48GB GDDR6
    "NV-A30": 75,                 # Ampere, 24GB HBM2
    "nvidia_a30_2g.12gb": 65,     # A30 MIG 2g slice, 12GB
    "NV-V100-SXM2": 60,           # Volta, 32GB HBM2
    "NV-RTX-A4000": 50,           # Ampere, 16GB GDDR6
    "NV-RTX5000": 45,             # Turing, 16GB GDDR6
    "NV-RTX2080Ti": 35,           # Turing, 11GB GDDR6
    "NV-RTX2070": 25,             # Turing, 8GB GDDR6
    "nvidia_a30_1g.6gb": 20,      # A30 MIG 1g slice, 6GB
}

# GPU VRAM in GB (for adequacy checks)
GPU_VRAM_GB: Dict[str, int] = {
    "nvidia_h200_nvl": 141,
    "nvidia_h100_nvl": 94,
    "nvidia_h100_pcie": 80,
    "NV-H100": 80,
    "NV-L40": 48,
    "NV-V100-SXM2": 32,
    "NV-A30": 24,
    "NV-RTX-A4000": 16,
    "NV-RTX5000": 16,
    "nvidia_a30_2g.12gb": 12,
    "NV-RTX2080Ti": 11,
    "NV-RTX2070": 8,
    "nvidia_a30_1g.6gb": 6,
}

# Map lowercased GPU names from sinfo to SLURM GRES names (preserving case)
GPU_NAME_MAP: Dict[str, str] = {
    "nvidia_h200_nvl": "nvidia_h200_nvl",
    "nv-h100": "NV-H100",
    "nvidia_h100_nvl": "nvidia_h100_nvl",
    "nvidia_h100_pcie": "nvidia_h100_pcie",
    "nv-l40": "NV-L40",
    "nv-a30": "NV-A30",
    "nvidia_a30_2g.12gb": "nvidia_a30_2g.12gb",
    "nvidia_a30_1g.6gb": "nvidia_a30_1g.6gb",
    "nv-v100-sxm2": "NV-V100-SXM2",
    "nv-rtx5000": "NV-RTX5000",
    "nv-rtx-a4000": "NV-RTX-A4000",
    "nv-rtx2080ti": "NV-RTX2080Ti",
    "nv-rtx2070": "NV-RTX2070",
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


def get_max_gpus_per_node(config: Config, partition: Optional[str] = None) -> Dict[str, int]:
    """Query sinfo for the max GPUs per node for each type. Returns {gpu_type: max_count}."""
    part = partition or config.default_partition or DEFAULT_PARTITION
    result = run_ssh(
        config,
        [
            "sinfo",
            "-p", part,
            "--Format=nodehost,gres:30",
            "--noheader",
        ],
        capture_output=True,
        check=False,
    )
    if not result.stdout:
        return {}

    max_per_node: Dict[str, int] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        gres_field = fields[1].strip()
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
            max_per_node[normalized] = max(max_per_node.get(normalized, 0), count)

    return max_per_node


def get_pending_gpu_counts(
    config: Config, partition: Optional[str] = None
) -> Dict[str, int]:
    """Query squeue for pending jobs per GPU type. Returns {gpu_type: pending_job_count}."""
    part = partition or config.default_partition or DEFAULT_PARTITION
    result = run_ssh(
        config,
        ["squeue", "-t", "PD", "-p", part, "-h", "-o", "%b"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {}

    pending: Dict[str, int] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Parse GRES like "gpu:NV-A30:1" or "gres/gpu:NV-A30:1"
        if "gpu:" not in line:
            continue
        gpu_part = line[line.index("gpu:") :]
        gres_parts = gpu_part.split(":")
        if len(gres_parts) < 3:
            continue
        gpu_name = gres_parts[1]
        normalized = GPU_NAME_MAP.get(gpu_name.lower(), gpu_name)
        pending[normalized] = pending.get(normalized, 0) + 1

    return pending


def select_best_gpu(
    config: Config,
    partition: Optional[str] = None,
    *,
    queue_aware: bool = True,
    min_gpus: int = 1,
) -> str:
    """Select the best available GPU, balancing compute power and queue depth.

    When queue_aware is True (default), each GPU type is scored as:
        score = priority / (1 + pending_jobs)

    This means an H100 (priority 100) with 1 pending job scores 50,
    beating an RTX 2070 (priority 25) with no queue. But an RTX 2080 Ti
    (priority 35) with no queue beats an A30 (priority 75) with 4 pending.

    When min_gpus > 1, GPU types that have fewer than min_gpus per node
    are filtered out (can't satisfy the request on a single node).

    Returns the GRES GPU type name (e.g. 'nvidia_h200_nvl', 'NV-A30').
    Falls back to 'NV-RTX2080Ti' if nothing is detected.
    """
    available = get_available_gpus(config, partition)
    if not available:
        return "NV-RTX2080Ti"

    # Filter by min GPUs per node
    if min_gpus > 1:
        max_per_node = get_max_gpus_per_node(config, partition)
        candidates = {
            g: c for g, c in available.items()
            if max_per_node.get(g, 0) >= min_gpus
        }
        if candidates:
            available = candidates

    if not queue_aware:
        return max(available, key=lambda g: GPU_PRIORITY.get(g, 0))

    pending = get_pending_gpu_counts(config, partition)

    def _score(g: str) -> float:
        return GPU_PRIORITY.get(g, 0) / (1 + pending.get(g, 0))

    return max(available, key=_score)


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
