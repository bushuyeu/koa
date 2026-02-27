"""
koa_cli package entrypoint.

Lightweight CLI and helpers for submitting and managing KOA SLURM jobs.
"""

from .config import Config, discover_config_path, load_config
from .manifest import update_manifest_metadata, write_run_manifest
from .runs import list_runs, record_submission, show_run, sync_statuses
from .slurm import (
    JobIOPaths,
    cancel_job,
    get_available_gpus,
    get_cluster_availability,
    get_job_io_paths,
    list_jobs,
    parse_gpu_count_from_script,
    queue_status,
    run_health_checks,
    select_best_gpu,
    submit_job,
)

__all__ = [
    "Config",
    "discover_config_path",
    "load_config",
    "write_run_manifest",
    "update_manifest_metadata",
    "record_submission",
    "list_runs",
    "show_run",
    "sync_statuses",
    "submit_job",
    "cancel_job",
    "list_jobs",
    "queue_status",
    "run_health_checks",
    "get_job_io_paths",
    "get_available_gpus",
    "get_cluster_availability",
    "select_best_gpu",
    "parse_gpu_count_from_script",
    "JobIOPaths",
]
