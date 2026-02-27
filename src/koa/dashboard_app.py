from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

try:  # Support running as `python dashboard_app.py` or via Streamlit.
    from .config import Config, DEFAULT_BACKEND_NAME, DEFAULT_CONFIG_PATH, load_config  # type: ignore
    from .dashboard_data import (
        collect_job_records,
        fetch_gpu_nodes,
        get_job_log_tail,
        job_record_to_dict,
        last_updated_timestamp,
    )
    from .runs import delete_run_entry, set_run_description  # type: ignore
except ImportError:  # pragma: no cover - fallback when executed outside package context
    import sys
    from pathlib import Path

    PACKAGE_ROOT = Path(__file__).resolve().parent
    if str(PACKAGE_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT.parent))
    from koa.config import Config, DEFAULT_BACKEND_NAME, DEFAULT_CONFIG_PATH, load_config  # type: ignore
    from koa.dashboard_data import (
        collect_job_records,
        fetch_gpu_nodes,
        get_job_log_tail,
        job_record_to_dict,
        last_updated_timestamp,
    )
    from koa.runs import delete_run_entry, set_run_description  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=Path, default=None, help="Path to koa-config.yaml")
    parser.add_argument("--backend", default=None, help="Backend name to load from the global config.")
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=0,
        help="How long (in seconds) job + GPU queries stay cached; set to 0 to disable auto-expiry (default: 0).",
    )
    return parser.parse_args()


def _load_available_backends(config_path: Optional[Path]) -> list[str]:
    path = config_path or DEFAULT_CONFIG_PATH
    try:
        data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return [DEFAULT_BACKEND_NAME]
    names: list[str] = []
    backends = data.get("backends")
    if isinstance(backends, list):
        for entry in backends:
            if isinstance(entry, dict):
                name = entry.get("cluster_name")
                if isinstance(name, str) and name:
                    if name not in names:
                        names.append(name)
    legacy = data.get("default_backend")
    if legacy and legacy not in names:
        names.insert(0, legacy)
    if not names:
        names = [DEFAULT_BACKEND_NAME]
    return names


def _format_timestamp(value: Optional[str]) -> str:
    if not value:
        return "—"
    try:
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt.astimezone()
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _status_bucket(status: Optional[str]) -> str:
    if not status:
        return "unknown"
    status_upper = status.upper()
    if status_upper.startswith("RUN"):
        return "running"
    if status_upper.startswith("PEND"):
        return "pending"
    if status_upper.startswith("COMP"):
        return "completed"
    if status_upper.startswith("FAIL") or status_upper.startswith("CANCEL") or "PREEMPT" in status_upper:
        return "needs_attention"
    return "other"


def _job_signature(config: Config) -> str:
    return f"{config.user}@{config.host}:{config.cluster_name}:{config.project_name}"


def _build_cache_loaders(config: Config, cache_ttl: int):
    ttl_value = None if cache_ttl <= 0 else max(5, cache_ttl)

    @st.cache_data(ttl=ttl_value, show_spinner=False)
    def _load_jobs(signature: str) -> dict:
        records = [job_record_to_dict(record) for record in collect_job_records(config)]
        return {"records": records, "fetched_at": last_updated_timestamp()}

    @st.cache_data(ttl=ttl_value, show_spinner=False)
    def _load_gpu(signature: str) -> dict:
        nodes = fetch_gpu_nodes(config)
        return {"rows": nodes, "fetched_at": last_updated_timestamp()}

    return _load_jobs, _load_gpu


def _render_status_metrics(job_rows: List[Dict[str, Any]]) -> None:
    counts = Counter(_status_bucket(row.get("status")) for row in job_rows)
    cols = st.columns(4)
    cols[0].metric("Running", counts.get("running", 0))
    cols[1].metric("Pending", counts.get("pending", 0))
    cols[2].metric("Completed", counts.get("completed", 0))
    cols[3].metric("Attention", counts.get("needs_attention", 0))


def _render_job_table(job_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    import pandas as pd

    table: List[Dict[str, Any]] = []
    for row in job_rows:
        actual_end = row.get("ended_at")
        expected_end = row.get("expected_end")
        if actual_end:
            ended_display = _format_timestamp(actual_end)
        elif expected_end:
            ended_display = f"{_format_timestamp(expected_end)} (scheduled)"
        else:
            ended_display = "—"
        table.append(
            {
                "Job ID": row["job_id"],
                "Name": row.get("job_name") or "—",
                "Status": row.get("status") or "UNKNOWN",
                "Submitted": _format_timestamp(row.get("submitted_at")),
                "Started": _format_timestamp(row.get("started_at")),
                "Ended": ended_display,
                "GPUs": row.get("gpu_summary") or "—",
                "Description": row.get("description") or "—",
                "Reason / Node": row.get("reason") or "—",
            }
        )
    df = pd.DataFrame(table)
    builder = GridOptionsBuilder.from_dataframe(df)
    builder.configure_selection(selection_mode="single", use_checkbox=False, pre_selected_rows=[0])
    builder.configure_pagination(paginationAutoPageSize=True)
    builder.configure_default_column(resizable=True, sortable=True, filter=True)
    grid_options = builder.build()
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=500,
    )
    selected_rows = grid_response.get("selected_rows")
    if selected_rows is None or selected_rows.empty:
        return None
    records = selected_rows.to_dict("records")
    return records[0] if records else None


def _render_job_details(config: Config, job: Dict[str, Any]) -> None:
    st.subheader(f"Details · {job['job_id']}")
    cols = st.columns(3)
    cols[0].markdown(f"**Status**: {job.get('status') or 'UNKNOWN'}")
    cols[0].markdown(f"**Reason**: {job.get('reason') or '—'}")
    cols[0].markdown(f"**Description**: {job.get('description') or '—'}")
    cols[1].markdown(f"**Submitted**: {_format_timestamp(job.get('submitted_at'))}")
    cols[1].markdown(f"**Started**: {_format_timestamp(job.get('started_at'))}")
    ended_at = job.get("ended_at")
    expected_end = job.get("expected_end")
    if ended_at:
        cols[1].markdown(f"**Ended**: {_format_timestamp(ended_at)}")
    elif expected_end:
        cols[1].markdown(f"**Ends (scheduled)**: {_format_timestamp(expected_end)}")
    else:
        cols[1].markdown("**Ended**: —")
    cols[2].markdown(f"**GPU(s)**: {job.get('gpu_summary') or '—'}")
    cols[2].markdown(f"**Nodes**: {job.get('node_list') or job.get('nodes') or '—'}")
    cols[2].markdown(f"**Partition**: {job.get('partition') or '—'}")

    dash_url = job.get("dashboard_url")
    remote_dir = job.get("remote_job_dir")
    local_dir = job.get("local_job_dir")

    if dash_url:
        st.markdown(f"[Open in cluster dashboard]({dash_url})")
    else:
        st.caption("Dashboard link unavailable for this job.")

    with st.expander("Edit", expanded=False):
        desc_key = f"desc-input-{job['job_id']}"
        can_edit = bool(job.get("has_local_record"))
        new_desc = st.text_area(
            "Description",
            value=job.get("description") or "",
            key=desc_key,
            disabled=not can_edit,
        )
        col_save, col_delete = st.columns([1, 1])
        if col_save.button("Save description", key=f"save-desc-{job['job_id']}", disabled=not can_edit):
            success = set_run_description(config, job["job_id"], new_desc.strip() or None)
            if success:
                st.success("Description updated.")
                st.rerun()
            else:
                st.error("Failed to update description.")
        if col_delete.button(
            "Remove run entry",
            key=f"delete-run-{job['job_id']}",
            disabled=not can_edit,
        ):
            if delete_run_entry(config, job["job_id"]):
                st.success("Run entry deleted.")
                st.rerun()
            else:
                st.error("Failed to delete run entry.")
        if not can_edit:
            st.caption("Notes require a local run recorded by this backend (via `koa submit`).")

    st.markdown("**Resource requests**")
    res_cols = st.columns(2)
    res_cols[0].json(job.get("alloc_tres") or {}, expanded=True)
    res_cols[1].json(job.get("req_tres") or {}, expanded=True)
    st.caption("Alloc vs requested TRES (cpu/mem/gpu counts reported by Slurm).")

    usage = job.get("tres_usage_tot") or job.get("sstat_usage", {}).get("tres_usage_ave")
    if usage:
        st.markdown("**Runtime usage (Slurm TRES)**")
        st.json(usage, expanded=False)

    max_rss = job.get("max_rss") or job.get("sstat_usage", {}).get("max_rss")
    ave_rss = job.get("sstat_usage", {}).get("ave_rss")
    ave_cpu = job.get("sstat_usage", {}).get("ave_cpu")
    if max_rss or ave_rss or ave_cpu:
        stats = []
        if max_rss:
            stats.append(f"Max RSS: {max_rss}")
        if ave_rss:
            stats.append(f"Ave RSS: {ave_rss}")
        if ave_cpu:
            stats.append(f"Ave CPU: {ave_cpu}")
        st.info(" · ".join(stats))

    if job.get("sbatch_args"):
        with st.expander("sbatch args"):
            for arg in job["sbatch_args"]:
                st.code(arg, language="")

    st.subheader("Logs")
    if not (local_dir or remote_dir):
        st.info("Log paths unavailable yet for this job.")
    else:
        stream_defs = [("stdout", "Stdout"), ("stderr", "Stderr")]
        tabs = st.tabs([label for _, label in stream_defs])
        for (stream, label), tab in zip(stream_defs, tabs):
            with tab:
                lines_key = f"log-lines-{job['job_id']}-{stream}"
                lines = st.session_state.get(lines_key, 50)
                if st.button("Load more", key=f"log-more-{job['job_id']}-{stream}"):
                    lines += 50
                    st.session_state[lines_key] = lines
                else:
                    st.session_state.setdefault(lines_key, lines)

                version = f"{job.get('status')}|{job.get('ended_at')}|{job.get('elapsed')}|{lines}|{stream}"
                cache_key = f"log-cache-{job['job_id']}-{stream}"
                cache_state = st.session_state.get(cache_key)
                if cache_state is None or cache_state.get("version") != version:
                    try:
                        payload = get_job_log_tail(
                            config,
                            job["job_id"],
                            local_job_dir=local_dir,
                            remote_job_dir=remote_dir,
                            stream=stream,
                            lines=lines,
                        )
                        cache_state = {"version": version, "payload": payload}
                    except Exception as exc:  # pragma: no cover - streamlit runtime feedback
                        cache_state = {"version": version, "error": str(exc)}
                    st.session_state[cache_key] = cache_state

                st.caption(f"Showing last {lines} lines of {label.lower()}.")
                if cache_state:
                    if "error" in cache_state:
                        st.error(cache_state["error"])
                    else:
                        payload = cache_state.get("payload", {})
                        st.caption(payload.get("source") or "")
                        st.code(payload.get("content") or "<no output>", language="")


def _render_gpu_tab(gpu_payload: dict) -> None:
    rows = gpu_payload.get("rows") or []
    if not rows:
        st.info("No GPU node information returned (sinfo output was empty).")
        return
    refreshed = gpu_payload.get("fetched_at")
    st.caption(f"Last refreshed: {_format_timestamp(refreshed) if refreshed else 'n/a'}")
    table: List[dict] = []
    for row in rows:
        cpus = row.get("cpus") or {}
        table.append(
            {
                "Node": row.get("node"),
                "State": row.get("state"),
                "Partition": row.get("partition"),
                "GPUs": row.get("gpu_summary") or row.get("gres"),
                "CPU alloc/idle/other/total": f"{cpus.get('allocated')}/{cpus.get('idle')}/{cpus.get('other')}/{cpus.get('total')}",
            }
        )
    st.dataframe(table, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="KOA Dashboard", layout="wide")
    args = _parse_args()
    available_backends = _load_available_backends(args.config)
    default_backend = args.backend or available_backends[0]

    st.sidebar.header("Cluster")
    selected_backend = st.sidebar.selectbox(
        "Backend",
        options=available_backends,
        index=available_backends.index(default_backend) if default_backend in available_backends else 0,
    )

    config = load_config(args.config, backend_name=selected_backend)
    cache_ttl = args.cache_ttl or 0
    load_jobs, load_gpu = _build_cache_loaders(config, cache_ttl)
    signature = _job_signature(config)

    st.sidebar.write(f"User: `{config.user}`")
    st.sidebar.write(f"Host: `{config.host}`")
    st.sidebar.write(f"Backend: `{config.cluster_name}`")
    st.sidebar.write(f"Project: `{config.project_name}`")
    if cache_ttl <= 0:
        st.sidebar.markdown("Cache TTL: **manual (no auto-expiry)**")
    else:
        st.sidebar.markdown(f"Cache TTL: **{cache_ttl}s**")
    if st.sidebar.button("Refresh now", width="stretch"):
        load_jobs.clear()
        load_gpu.clear()
        st.rerun()

    st.title("KOA Dashboard")

    tabs = st.tabs(["Jobs", "GPU Nodes"])

    with tabs[0]:
        try:
            jobs_payload = load_jobs(signature)
            job_rows = jobs_payload.get("records", [])
        except Exception as exc:  # pragma: no cover - runtime feedback
            st.error(f"Unable to load jobs: {exc}")
            job_rows = []
            jobs_payload = {"fetched_at": None}

        refreshed = jobs_payload.get("fetched_at")
        st.caption(f"Last refreshed: {_format_timestamp(refreshed) if refreshed else 'n/a'}")

        if not job_rows:
            st.info("No recorded jobs yet. Submit with `koa submit` to populate the catalog.")
        else:
            _render_status_metrics(job_rows)
            selected_row = _render_job_table(job_rows)
            job_lookup = {row["job_id"]: row for row in job_rows}
            if selected_row:
                selected_job_id = selected_row.get("Job ID")
            else:
                selected_job_id = job_rows[0]["job_id"]
            st.markdown("### Job details")
            _render_job_details(config, job_lookup.get(selected_job_id, job_rows[0]))

    with tabs[1]:
        try:
            gpu_payload = load_gpu(signature)
        except Exception as exc:  # pragma: no cover - runtime feedback
            st.error(f"Unable to load GPU nodes: {exc}")
            gpu_payload = {"rows": []}
        _render_gpu_tab(gpu_payload)


if __name__ == "__main__":  # pragma: no cover
    main()
