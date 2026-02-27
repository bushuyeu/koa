# koa

A lightweight command-line companion for the University of Hawai'i KOA cluster. The tool focuses on one job: making it easy to sync code, submit jobs, and monitor their life cycle from your local workstation.

---

## Highlights
- **Job submission pipeline** – copy a script, infer sensible defaults, and hand it to `sbatch` with a single command.
- **Auto GPU selection** – automatically detects the best available GPU type on the target partition and injects the right `--gres` flag (ranked H200 NVL > H100 NVL > H100 PCIe > L40 > A30 > V100 SXM2 > RTX A4000 > RTX 5000 > RTX 2080 Ti > RTX 2070).
- **Scheduling optimizer** – dry-run `sbatch --test-only` across every GPU type and partition to find the fastest start time (`koa optimize`).
- **Job auditing** – analyze historical resource usage via `sacct` and right-size future requests (`koa audit`).
- **Job chaining** – split long training runs into auto-dependent segments with `koa submit --chain N`.
- **Queue intelligence** – pending-reason decoder (`koa why`), queue depth and wait-time estimates (`koa spy`), fair-share score tracker (`koa priority`).
- **Live efficiency** – detect idle GPUs on running jobs via `nvidia-smi` over SSH (`koa efficiency`).
- **Automation** – one-click resubmit from manifest, parameter sweeps via job arrays, GPU availability watch, Slack/Discord webhook alerts.
- **Cluster queue view** – see the full cluster queue with your own jobs highlighted in bold, others dimmed, all in a Rich-formatted table.
- **GPU availability** – real-time GPU/node inventory color-coded by state (green=idle, yellow=mixed, red=down) with a summary of GPU counts by type.
- **AI agent integration** – MCP server (`koa-mcp`), Claude Code slash commands, and `--format json` on all commands for machine-readable output.
- **Workspace snapshots** – every submission bundles the exact repo state so jobs run with reproducible code and configs.
- **Run catalog** – every submission is indexed locally so you can list, sync, and inspect historical runs.
- **Plain Python package** – small dependency footprint and no bundled training/evaluation code.

---

# Prerequisites

Add this to `~/.bashrc` in HPC environment in order to avoid using up disk quota.
```bash
export XDG_CACHE_HOME=<path to your scratch folder. eg. /mnt/lustre/koa/scratch/yosubs/cache>
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCH_HOME=$XDG_CACHE_HOME/torch
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TMPDIR=$XDG_CACHE_HOME/tmp
export APPTAINER_CACHEDIR=$XDG_CACHE_HOME/apptainer
```

Setup ssh config at `~/.ssh/config` so that it doesn't keep asking for multi-factor authentication.
```bash
Host koa koa.its.hawaii.edu
  HostName koa.its.hawaii.edu
  User <username: e.g. yosubs>
  IdentitiesOnly yes
  IdentityFile <path to secret key: e.g. ~/.ssh/koa_key>
  ControlMaster auto
  ControlPath ~/.ssh/control-%r@%h:%p
  ControlPersist 25h
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

For automated daily DUO authentication (tap "Approve" on your phone once per day), see `scripts/koa-auth-setup.sh`.

## Installation

```bash
# Recommended: uv tool keeps koa isolated but available everywhere
uv tool install git+https://github.com/bushuyeu/koa.git

# Development: editable install
uv tool install --force --editable .
```

The CLI installs the entry point `koa`.

---

## Configure access

1. Run `koa setup` (once per machine) to capture your KOA username, host, and the global workspace roots on KOA and locally. The global config lives at `~/.config/koa/config.yaml`.
   - The global config now supports multiple Slurm backends via a top-level `backends:` list. Each entry records the connection + workspace settings for a `cluster_name` (e.g. `koa` or `delta`). `koa setup` targets the default `koa` backend; pass `--backend delta` to add or update another cluster.
   - Use `--default-constraint hopper` (or leave blank) to set per-cluster Slurm constraints; this keeps KOA on `hopper` nodes while clusters like Delta can omit constraints entirely.
   - Use `--default-gres gpu:a100:1` (or similar) if you want a default `--gres` line added to submissions for a backend.
   - Use `--cuda-version 12.4` (or leave blank for 12.8) to set the CUDA Toolkit minor version that should be installed automatically for that backend.
   - Provide `--dashboard-base-url https://<ondemand-host>/pun/sys/dashboard/files/fs` if you want web links in the dashboard to jump straight into your OnDemand file browser.
2. Inside each repository, run `koa init` to generate a minimal `koa-config.yaml` plus helper scripts. Use `--cuda-version` if this project needs a different CUDA Toolkit minor version from the backend default.
3. (Optional) Update `env_watch` and `snapshot_excludes` in `koa-config.yaml` if your project needs custom lockfiles or directories you want to skip during snapshots.

The CLI automatically discovers `koa-config.yaml` by walking up from the current working directory. Environment variables such as `KOA_USER`, `KOA_HOST`, `KOA_IDENTITY_FILE`, `KOA_REMOTE_ROOT`, `KOA_LOCAL_ROOT`, `KOA_DEFAULT_PARTITION`, `KOA_DEFAULT_CONSTRAINT`, `KOA_CUDA_VERSION`, `KOA_ENV_WATCH`, `KOA_PROXY_COMMAND`, and `KOA_DASHBOARD_BASE_URL` can override the saved configuration at runtime.

Example `koa-config.yaml` generated by `koa init`:

```yaml
project: my-awesome-project

default_backend: koa

cuda_minor_version: 12.8

env_watch:
  - scripts/setup_env.sh
  - requirements.txt
  - pyproject.toml

# Always forward these env vars when submitting (if set locally)
env_pass:
  - MODEL_NAME
  - DATA_ROOT
```

### Sample multi-backend ~/.config/koa/config.yaml

```yaml
default_backend: koa

backends:
  - cluster_name: koa
    user: yosubs
    host: koa.its.hawaii.edu
    remote_root: /mnt/lustre/koa/scratch/yosubs/koa
    local_root: ~/koa-projects
    default_partition: kill-shared
    default_constraint: hopper
    default_gres: gpu:NV-A30:1
    cuda_minor_version: 12.8

  - cluster_name: delta
    user: yosubs
    host: login.delta.ncsa.illinois.edu
    remote_root: /projects/yosubs/koa
    local_root: ~/delta-projects
    default_partition: gpuA100x4  # leave blank if cluster picks a default
    default_constraint: ""        # unset to avoid hopper-only behavior
    default_gres: gpu:NV-A30:1
    cuda_minor_version: 12.4
```

Run `koa setup --backend delta` to add or update the `delta` block; omit flags to keep existing values. Use `--backend <name>` on commands (or `KOA_BACKEND`) to pick the cluster when submitting.
Set per-cluster constraints by adding `default_constraint: hopper` under the relevant backend in `~/.config/koa/config.yaml` (leave it unset for clusters that don't need a constraint). Project-level `koa-config.yaml` can override or clear it if needed.
Add optional `snapshot_excludes:` if you want to skip additional files or directories during submission snapshots (e.g., raw datasets or build artifacts). Patterns without a slash match any basename (`data` matches every `data/` folder). Patterns with a slash match repo-relative paths (`data/hf` matches only that path). Prefix a basename with `./` or `/` to target only the repo root (`./data` excludes the top-level `data/` only).



---

## Everyday workflow

```bash
# 0. (First time) Configure your KOA defaults
koa setup --user $USER

# 0b. (Per project) Bootstrap a repo with config + scripts
koa init

# 1. Check connectivity and cluster health
koa check

# 2. Find the optimal GPU config before submitting
koa optimize scripts/basic_job.slurm --time 04:00:00
# Tests all GPU types × partitions, shows estimated start times

# 3. Submit a job script
koa submit scripts/basic_job.slurm --time 01:00:00 --desc "baseline"
# Auto-selected GPU: h100 x1
# Submitted KOA job 123456

# Submit a long training run as chained 4-hour segments
koa submit scripts/train.slurm --chain 5 --chain-time 04:00:00

# Parameter sweep over a grid
koa sweep scripts/hparam_search.slurm --params sweep_config.json

# 4. Monitor jobs
koa jobs                        # your active jobs
koa queue -p sandbox            # full cluster queue
koa availability                # GPU/node inventory
koa why <job-id>                # explain why a job is pending
koa efficiency <job-id>         # live GPU utilization
koa efficiency <job-id> --watch 30  # refresh every 30s

# 5. Queue intelligence
koa spy                         # queue depth, next GPUs freeing, wait history
koa priority                    # your fair-share score and ranking
koa limits                      # QOS quotas and account associations

# 6. Audit and optimize future jobs
koa audit --days 14             # right-sizing advice from recent history

# 7. Automation
koa resubmit <job-id>           # replay a previous submission
koa notify setup --slack-url https://hooks.slack.com/...  # configure alerts
koa notify start --all          # start background job-state alerting
koa watch --gpu-type h100       # alert when H100s become available

# 8. Run management
koa logs <job-id> --follow
koa cancel <job-id>
koa runs list
koa runs sync
koa dashboard
```

Every submitted job includes a `run_metadata/` folder under its results directory containing `manifest.json`, `git_head.txt`, `git_status.txt`, `env_hashes.json`, and any untracked files that were present locally when you launched the run.

---

## CLI reference

### Core commands

- `check` – run a quick SSH round-trip and display `sinfo` output.
- `setup` – configure global defaults (user, workspace roots, default CUDA Toolkit version).
- `init` – scaffold project config and helper scripts using global defaults.
- `jobs` – list your queued and running jobs in a Rich-formatted table with color-coded states (green = running, yellow = pending, red = failed).
- `queue` – display the full cluster queue as a Rich table. Your own jobs are **bold and color-coded**; other users' jobs are dimmed. Use `--partition`/`-p` to filter by partition.
- `availability` – show a real-time GPU/node inventory table. Rows are color-coded by state (green=idle, yellow=mixed/allocated, red=down/drained). Includes a summary footer with GPU counts grouped by type and state. Use `--partition`/`-p` to filter.
- `dashboard` – open the Streamlit dashboard with job history, logs, and GPU node views.
- `submit` – copy a script and call `sbatch`; use `--sbatch-arg` for raw overrides. Add flags like `--gpus` (generic count), `--constraint hopper`, or `--desc` to control resources and the timestamped results folder name. Forward env vars with `--env NAME` or `--env NAME=value`, and set defaults in `env_pass` within `koa-config.yaml`.
  - **Auto GPU selection** is enabled by default. The CLI queries `sinfo` for idle/mix nodes on the target partition, ranks available GPU types by priority (H200 NVL > H100 NVL > H100 PCIe > L40 > A30 > V100 SXM2 > RTX A4000 > RTX 5000 > RTX 2080 Ti > RTX 2070), and injects the appropriate `--gres=gpu:<type>:<count>` flag. The GPU count is read from `#SBATCH --gres=gpu:N` in your script (defaults to 1).
  - Pass `--no-auto-gpu` to disable this behavior, or use `--gpus` / `--gres` to override manually.
  - **`--chain N`** – split the job into N dependent segments. Each segment runs for `--chain-time` (default: 4h) and automatically continues the next with `--dependency=afterok`. Jobs receive `SLURM_CHAIN_LINK` and `SLURM_CHAIN_TOTAL` environment variables. Add `--off-peak` to schedule the first segment at 23:00.
  - **`--distributed`** – multi-node training helper. Auto-detects framework (PyTorch, DeepSpeed, Horovod), injects `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, NCCL env vars, and correct `--nodes`/`--ntasks-per-node` flags. Use `--nodes N`, `--gpus-per-node G`, `--framework` to configure.
  - **`--anywhere`** – cross-cluster smart routing. Probes all configured backends in parallel via `sbatch --test-only`, shows a comparison table of estimated start times, and submits to the fastest cluster.
- `cancel` – stop a job by ID with `scancel`.
- `logs` – stream or inspect a job's stdout/stderr in real time via `tail` (stored at `<remote results dir>/<job-id>/job.log` and `job.err`).
- `jupyter` – launch Jupyter Lab (or Notebook) on a GPU compute node with automatic SSH tunnel. Auto-selects the best GPU, submits a SLURM job, waits for allocation, opens the tunnel, and prints a ready-to-paste URL for VS Code. Ctrl+C cleanly cancels the job and closes the tunnel. Options: `--time`, `--gpus`, `--gpu-type`, `--partition`, `--port`, `--lab`/`--notebook`, `--conda-env`, `--mem`.
- `runs` – sync and inspect the local catalog of submitted jobs.
  - `koa runs list` shows recent submissions (most recent first).
  - `koa runs sync` updates Slurm status and downloads completed runs into the local mirror automatically.
  - `koa runs show <job-id>` prints the recorded metadata (git commit, env hashes, locations) for a single run.

### Scheduling commands

- `optimize` – find the fastest GPU configuration by running `sbatch --test-only` across all GPU types and partitions. Shows a ranked table of estimated start times so you can pick the config that gets your job running soonest. Use `--time`, `--partition`, `--gpu-type` to constrain the search space.
- `audit` – analyze recent job history via `sacct` and identify resource waste. Reports memory, time, and CPU efficiency per job with color-coded ratings, suggests right-sized values (130% of actual peak), and summarizes total wasted compute-hours. Use `--days N` to control the lookback window (default: 7).
- `why` – explain why a job is pending. Decodes SLURM reason codes (Priority, Resources, QOSMaxJobsPerUser, etc.) into plain-language explanations with actionable advice. Shows queue position and estimated wait context.
- `diagnose` – automatic failure diagnosis for completed/failed jobs. Reads `sacct` exit codes, stderr logs, and matches against known failure patterns (OOM, CUDA OOM, timeout, node failure, NCCL errors, missing files, missing modules). Provides plain-language root cause and actionable fix commands. Use `koa diagnose <job_id>`.
- `validate` – pre-flight resource validation. Parses `#SBATCH` directives from your script and checks against cluster capabilities: partition exists, GPU type available, memory adequate for GPU, time within QOS limits, GPU code present in script. Automatically runs as advisory warnings during `koa submit`.
- `budget` – resource allocation tracking. Shows GPU-hours consumed over a period, broken down by state (completed/failed/cancelled) and partition. Displays burn rate, projected allocation exhaustion, and per-job breakdown. Use `--days N` and `--breakdown` for detail.

### Queue intelligence commands

- `limits` – display your QOS limits, account associations, and fair-share information in three Rich tables. Shows max jobs, max GPUs, max wall time, and current usage against each quota.
- `spy` – queue intelligence dashboard. Shows queue depth per partition, the next GPUs to free (sorted by end time), historical wait-time statistics (median/avg/min/max), and a partition overview. Use `--partition`/`-p` to focus on a specific partition.
- `priority` – display your fair-share score, queue ranking, and priority factor breakdown from `sprio` and `sshare`. Includes recovery advice when your fair-share is low. Use `--all` to see all users.
- `efficiency` – live GPU waste detector. Queries `nvidia-smi` on the compute node via SSH hop and correlates with `sstat` CPU/memory data. Flags GPUs below 20% utilization. Use `--watch N` for continuous monitoring with N-second refresh.

### Automation commands

- `resubmit` – re-run a previous job using its stored manifest. Recovers the exact script path and sbatch arguments from the run catalog. Use `--dry-run` to preview without submitting.
- `notify` – Slack/Discord webhook alerts for job state changes.
  - `koa notify setup --slack-url URL` or `--discord-url URL` to configure.
  - `koa notify start [--job-id ID | --all]` to begin background monitoring.
  - `koa notify status` to check the monitor process.
- `sweep` – parameter sweep via SLURM job arrays. Provide a JSON or YAML params file defining the sweep grid; KOA computes the cartesian product, uploads a params mapping, and submits with `--array=0-{N-1}`. The job script reads `KOA_SWEEP_PARAMS_FILE` to look up its combination. Use `--max-concurrent N` to throttle.
- `watch` – monitor the cluster for GPU availability. Polls `sinfo` for idle nodes matching `--gpu-type` and `--partition`, sends a webhook alert when GPUs appear. Use `--once` for a single check, or let it run continuously with Rich Live display. Use `--interval N` to set poll frequency.

### Multi-cluster and environment commands

- `anywhere` – standalone cross-cluster comparison. Probes all configured backends in parallel and shows a ranked table of estimated start times. Use `koa anywhere <script> [--time T]` for comparison without submitting.
- `distributed show` – preview multi-node training configuration. Shows detected framework, injected env vars, sbatch flags, and launcher command suggestion without submitting. Use `koa distributed show <script>`.
- `env` – environment snapshot management.
  - `koa env freeze` – capture current Python environment, CUDA version, and system info into `koa-env.lock.yaml`.
  - `koa env deploy` – deploy the frozen environment to the remote cluster via SSH.
  - `koa env diff` – compare local lockfile against the remote environment, showing package mismatches.

All new commands support `--format json` for machine-readable output, `--config` for alternate config files, and `--backend` for multi-cluster targeting.

---

## Auto GPU Selection

When you run `koa submit`, the CLI automatically selects the best available GPU type on the target partition. This eliminates the need to manually check node availability or remember GRES names.

**How it works:**

1. Queries `sinfo` for nodes in `idle` or `mix` state on the target partition.
2. Parses the GRES field to identify available GPU types and counts.
3. Ranks GPUs by priority:

   | Rank | GPU | GRES Name | Priority Score |
   |------|-----|-----------|---------------|
   | 1 | H200 NVL | `nvidia_h200_nvl` | 110 |
   | 2 | H100 NVL | `nvidia_h100_nvl` | 105 |
   | 3 | H100 PCIe | `nvidia_h100_pcie` / `NV-H100` | 100 |
   | 4 | L40 | `NV-L40` | 85 |
   | 5 | A30 | `NV-A30` | 75 |
   | 6 | A30 MIG 2g | `nvidia_a30_2g.12gb` | 65 |
   | 7 | V100 SXM2 | `NV-V100-SXM2` | 60 |
   | 8 | RTX A4000 | `NV-RTX-A4000` | 50 |
   | 9 | RTX 5000 | `NV-RTX5000` | 45 |
   | 10 | RTX 2080 Ti | `NV-RTX2080Ti` | 35 |
   | 11 | RTX 2070 | `NV-RTX2070` | 25 |
   | 12 | A30 MIG 1g | `nvidia_a30_1g.6gb` | 20 |

4. Injects `--gres=gpu:<best_type>:<count>` into the `sbatch` command.

**GPU count** is read from `#SBATCH --gres=gpu:N` in your script. If no such directive exists, it defaults to 1.

**Disable** auto selection with `--no-auto-gpu`, or override it with `--gpus` or an explicit `--gres` via `--sbatch-arg`.

```bash
# Auto-selects best GPU (default)
koa submit scripts/basic_job.slurm

# Disable auto selection
koa submit scripts/basic_job.slurm --no-auto-gpu

# Manual override (also disables auto selection)
koa submit scripts/basic_job.slurm --gpus 4
```

---

## Cluster Queue View

Use `koa queue` to see all jobs across the cluster at a glance. Your own jobs are highlighted in **bold** with color-coded state indicators, while other users' jobs appear dimmed.

```bash
# Full cluster queue
koa queue

# Filter by partition
koa queue --partition sandbox
koa queue -p kill-shared
```

The table includes: Job ID, User, Name, State, Time, Time Limit, Nodes, CPUs, Min Memory, and Node List. A caption at the bottom shows how many of your jobs are currently in the queue.

The `koa jobs` command also now uses Rich-formatted output with the same color coding (green for running, yellow for pending, red for failed/cancelled).

---

## Dashboard

KOA ships with a built-in Streamlit dashboard; no extra install steps are required.

Launch the Streamlit UI via:

```bash
koa dashboard
```

Features:

- **Job catalog** – combines the local `runs.json` history with live `squeue`/`sacct` data so you can review submission, start/end times, GPU allocations, Slurm reasons, and recorded manifests.
- **Log viewer** – fetches stdout/stderr from the local mirror when available and falls back to streaming the remote log path via SSH.
- **Cluster links** – set `dashboard_base_url` (or `KOA_DASHBOARD_BASE_URL`) to your Open OnDemand file-browser prefix to unlock per-job shortcuts to the run folder.
- **Run notes** – the optional `--desc` flag is saved with each submission, and you can edit or clear descriptions (or delete stale runs) directly inside the dashboard.
- **Resource usage** – surfaces reported TRES allocations/usage, Max RSS, and live `sstat` samples for running jobs when available.
- **GPU inventory** – mirrors `sinfo -N -o "%N|%G|%T|%C|%P"` so you can see node states, partitions, and GPU models at a glance.

Use the sidebar’s refresh button to force a reread of Slurm data; otherwise the dashboard automatically refreshes on every interaction (with remote calls cached for ~30 seconds).

---

## Directory layout

`koa setup` captures two roots:

- **Remote root** (e.g. `/mnt/lustre/koa/scratch/<user>/koa`)
- **Local root** (e.g. `~/.koa`)

For a project named `<project>`, the CLI derives:

```
Remote: <remote_root>/projects/<project>/jobs/<timestamp[_desc]>/{repo,run_metadata,results,job.log,job.err}
Local : <local_root>/projects/<project>/jobs/<timestamp[_desc]>/{repo,run_metadata,results,job.log,job.err}
```

- `repo/` contains the exact snapshot submitted with the job.
- `run_metadata/` holds manifests, git info, and environment hashes.
- `results/` is where your job writes outputs; `koa runs sync` copies the entire run directory (including logs) back to the local mirror automatically once the job completes.

---

## Sample SLURM script

Minimal templates live under `src/koa/templates/`. Start from `basic_job.slurm` and adapt the resources, modules, and commands to your workload (including any `#SBATCH --gres` lines you require). The CLI sets `KOA_ML_RESULTS_ROOT` automatically so jobs can collect outputs in the directory that `koa runs sync` mirrors locally once they finish.

Running `koa init` also drops a project-specific `scripts/basic_job.slurm` and `scripts/setup_env.sh` that you can customise; they mirror the global defaults captured by `koa setup`. The default config watches files like `scripts/setup_env.sh`, `requirements.txt`, and `pyproject.toml`, so changing any of them automatically triggers a virtualenv rebuild on the next submission.

---

## AI Agent Integration

KOA is designed to be operated by AI agents as well as humans. Three integration points are provided:

### MCP Server

The `koa-mcp` entry point exposes all commands as [Model Context Protocol](https://modelcontextprotocol.io/) tools, letting AI assistants manage your HPC jobs directly.

Configure in `~/.claude/settings.json`:

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

Available MCP tools: `koa_cluster_status`, `koa_jobs`, `koa_queue`, `koa_availability`, `koa_cancel`, `koa_logs`, `koa_optimize`, `koa_why`, `koa_audit`, `koa_limits`, `koa_spy`, `koa_priority`, `koa_efficiency`, `koa_watch_once`, `koa_submit`, `koa_resubmit`, `koa_jupyter`, and more.

### Claude Code Slash Commands

Pre-built workflows in `.claude/commands/`:

- `/koa-submit <script>` – find optimal config, submit, report result
- `/koa-monitor` – comprehensive status report with recommendations
- `/koa-audit` – efficiency analysis with right-sizing suggestions
- `/koa-optimize <script> [flags]` – test scheduling across GPU configurations
- `/koa-status` – cluster and job status summary

### JSON Output

All commands support `--format json` for structured, machine-parseable output:

```bash
koa jobs --format json
koa spy --format json
koa optimize scripts/train.slurm --format json
```

---

## Development

```bash
uv tool install --force --editable .
ruff check
pytest
```

Contributions are welcome via pull request.
