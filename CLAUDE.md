# KOA CLI — AI Agent Instructions

## Overview
KOA CLI manages HPC jobs on SLURM clusters (University of Hawaii KOA and compatible systems). It connects via SSH to execute SLURM commands remotely.

## Key Commands (all support --format json for machine-readable output)

### Scheduling Weapons
- `koa optimize <script>` — Find fastest GPU config via dry-run scheduling simulation
- `koa audit [--days N]` — Right-size jobs from history; shows resource waste
- `koa why <job_id>` — Explain why a job is pending with actionable advice
- `koa diagnose <job_id>` — Auto-diagnose failed/completed jobs (OOM, timeout, NCCL, missing deps)
- `koa validate <script>` — Pre-flight checks (GPU/partition/memory/time validation)
- `koa budget [--days N]` — GPU-hours used, burn rate, allocation tracking
- `koa submit <script> --chain N` — Auto job chaining for long training runs
- `koa submit <script> --off-peak` — Schedule for off-peak hours (23:00)

### Queue Intelligence
- `koa limits` — QOS/quota/fair-share visibility
- `koa spy [-p PARTITION]` — Queue depth, next GPUs freeing, wait history
- `koa priority [--all]` — Fair-share score and queue ranking
- `koa efficiency <job_id>` — Live GPU utilization and waste detection

### Core Operations
- `koa jobs` — Your active jobs
- `koa queue [-p PARTITION]` — Full cluster queue (your jobs highlighted)
- `koa availability [-p PARTITION]` — GPU/node inventory
- `koa submit <script> [options]` — Submit jobs with auto GPU selection
- `koa submit <script> --distributed` — Multi-node training with auto NCCL/env config
- `koa submit <script> --anywhere` — Cross-cluster routing (submit to fastest backend)
- `koa cancel <job_id>` — Cancel a job
- `koa logs <job_id> [--follow]` — Stream job logs

### Environment Management
- `koa env freeze` — Snapshot Python env, CUDA, packages to koa-env.lock.yaml
- `koa env deploy` — Deploy frozen environment to remote cluster
- `koa env diff` — Compare local lockfile vs remote environment

### Automation
- `koa resubmit <job_id>` — Replay previous submission from manifest
- `koa sweep <script> --params FILE` — Parameter sweep via job arrays
- `koa watch --gpu-type TYPE` — Monitor for GPU availability
- `koa notify start --all` — Job state change alerts via webhooks
- `koa anywhere <script>` — Compare all backends without submitting
- `koa distributed show <script>` — Preview multi-node config without submitting

### Run Management
- `koa runs list` — List recorded submissions
- `koa runs sync` — Update statuses from cluster
- `koa runs show <job_id>` — Show run metadata

## MCP Server
Configure in Claude Code settings (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "koa": {
      "command": "koa-mcp",
      "args": []
    }
  }
}
```

## Working with this codebase
- Source: `src/koa_cli/`
- Commands: `src/koa_cli/commands/` — each module has `register_parser()` and `handle()`
- SLURM interaction: `src/koa_cli/ssh.py` via `run_ssh(config, cmd_list, capture_output=True)`
- Config: `~/.config/koa/config.yaml` (global) and `koa-config.yaml` (project-level)
- MCP server: `src/koa_cli/mcp_server.py` — FastMCP-based, exposes all commands as tools
- Rich library used for terminal formatting; all commands support `--format json`
- Tests: `pytest` (run from project root)

## Workflow for AI agents
1. Use `--format json` for all queries to get structured data
2. Start with `koa availability --format json` to understand cluster state
3. Use `koa validate` to check script before submitting
4. Use `koa optimize` or `koa anywhere` to find the fastest config/cluster
5. After submission, use `koa why` for pending jobs and `koa efficiency` for running jobs
6. If a job fails, use `koa diagnose` for automatic root cause analysis
7. Use `koa audit` and `koa budget` periodically to right-size requests and track usage
8. Use `koa env freeze` before submitting to ensure environment reproducibility
