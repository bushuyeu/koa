from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Optional


def _run_git(args: Iterable[str], cwd: Path) -> tuple[int, str, str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _detect_git_root(start: Path) -> Optional[Path]:
    returncode, stdout, _ = _run_git(["rev-parse", "--show-toplevel"], start)
    if returncode != 0:
        return None
    return Path(stdout).resolve()


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_untracked_files(repo_root: Path, status_lines: Iterable[str], dest_dir: Path) -> list[str]:
    untracked: list[str] = []
    for line in status_lines:
        if line.startswith("?? "):
            rel_path = line[3:].strip()
            if not rel_path:
                continue
            src = repo_root / rel_path
            if not src.exists():
                continue
            dest_path = dest_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest_path)
            untracked.append(rel_path)
    return untracked


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_run_manifest(destination: Path, *, workspace: Optional[Path] = None, env_watch: Optional[Iterable[str]] = None) -> dict:
    """
    Populate ``destination`` with metadata describing the current workspace state.
    """
    destination.mkdir(parents=True, exist_ok=True)
    workspace = workspace or Path.cwd()
    metadata: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
    }

    repo_root = _detect_git_root(workspace)
    if repo_root is None:
        _write_text_file(destination / "git_status.txt", "Not a git repository.\n")
        metadata["git"] = {"present": False}
    else:
        metadata["git"] = {"present": True, "root": str(repo_root)}

        head_rc, head_stdout, _ = _run_git(["rev-parse", "HEAD"], repo_root)
        if head_rc == 0:
            metadata["git"]["head"] = head_stdout
            _write_text_file(destination / "git_head.txt", f"{head_stdout}\n")
        else:
            metadata["git"]["head"] = None

        status_rc, status_stdout, status_stderr = _run_git(
            ["status", "--short"], repo_root
        )
        status_path = destination / "git_status.txt"
        if status_rc == 0:
            _write_text_file(status_path, status_stdout + ("\n" if status_stdout else ""))
            status_lines = status_stdout.splitlines()
        else:
            combined = status_stdout or status_stderr or "Failed to read git status."
            _write_text_file(status_path, combined + ("\n" if combined else ""))
            status_lines = []

        untracked_dir = destination / "untracked"
        untracked_files = _copy_untracked_files(repo_root, status_lines, untracked_dir)
        if untracked_files:
            _write_text_file(
                destination / "untracked_files.txt", "\n".join(untracked_files) + "\n"
            )
        metadata["git"]["untracked_files"] = untracked_files
        metadata["git"]["status_lines"] = status_lines

    env_hashes: dict[str, str] = {}
    if env_watch:
        for rel_path in env_watch:
            rel = rel_path.strip()
            if not rel:
                continue
            candidate = workspace / rel
            if candidate.exists() and candidate.is_file():
                try:
                    env_hashes[rel] = _hash_file(candidate)
                except OSError:
                    env_hashes[rel] = "<error>"
            else:
                env_hashes[rel] = "<missing>"
        metadata["env_hashes"] = env_hashes
        _write_text_file(destination / "env_hashes.json", json.dumps(env_hashes, indent=2))

    metadata_path = destination / "manifest.json"
    _write_text_file(metadata_path, json.dumps(metadata, indent=2))
    return metadata


def update_manifest_metadata(destination: Path, **updates: object) -> None:
    metadata_path = destination / "manifest.json"
    if metadata_path.exists():
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        current = {}
    current.update(updates)
    _write_text_file(metadata_path, json.dumps(current, indent=2))
