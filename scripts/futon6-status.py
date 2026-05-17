#!/usr/bin/env python3
"""Summarize the live futon6/mark2 lane from local and Linode evidence."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REMOTE_PROBE = r"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MARK2_HOME = Path(sys.argv[1])
ROB_HOME = Path(sys.argv[2])
VERIFY_HASHES = sys.argv[3] == "1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256sum(path: Path) -> str | None:
    if not VERIFY_HASHES:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256sum(path),
    }


def list_files(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [file_entry(p) for p in sorted(path.iterdir()) if p.is_file()]


def manifest_counts(path: Path) -> tuple[int | None, int | None]:
    if not path.exists():
        return None, None
    db = sqlite3.connect(path)
    try:
        pending = db.execute(
            "SELECT COUNT(*) FROM papers "
            "WHERE latest=1 AND include=1 AND status='pending'"
        ).fetchone()[0]
        total = db.execute(
            "SELECT COUNT(*) FROM papers WHERE latest=1 AND include=1"
        ).fetchone()[0]
        return pending, total
    finally:
        db.close()


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"batches": {}, "next_batch": 1}
    return json.loads(path.read_text())


def build_lock_info(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return {}
    info: dict[str, str] = {}
    for field in text.split():
        if "=" in field:
            key, value = field.split("=", 1)
            info[key] = value
    if not info:
        info["raw"] = text
    return info


def active_builder(lock_info: dict[str, str] | None) -> tuple[int | None, str | None]:
    if lock_info:
        pid_text = lock_info.get("pid")
        if pid_text and pid_text.isdigit():
            pid = int(pid_text)
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode().strip()
            except FileNotFoundError:
                cmdline = ""
            if "mark2" in cmdline and (" fill " in f" {cmdline} " or " build " in f" {cmdline} "):
                return pid, cmdline

    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        if pid_text.isdigit() and "mark2" in args and (" fill " in f" {args} " or " build " in f" {args} "):
            return int(pid_text), args
    return None, None


def latest_fill_log(path: Path) -> dict[str, object] | None:
    log_dir = path / "logs"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("fill*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None
    log = logs[0]
    lines = [line.rstrip() for line in log.read_text(errors="replace").splitlines() if line.strip()]
    tail = lines[-5:]
    return {
        "path": str(log),
        "mtime": datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc).isoformat(),
        "tail": tail,
    }


state = load_state(MARK2_HOME / "state.json")
pending, total = manifest_counts(MARK2_HOME / "arxiv_manifest.sqlite")
lock_info = build_lock_info(MARK2_HOME / "build.lock")
builder_pid, builder_cmd = active_builder(lock_info)

result = {
    "checked_at": now_iso(),
    "hostname": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
    "mark2_home": str(MARK2_HOME),
    "rob_home": str(ROB_HOME),
    "manifest": {"pending": pending, "total": total},
    "state": state,
    "build_lock": lock_info,
    "active_builder": {"pid": builder_pid, "cmd": builder_cmd},
    "latest_fill_log": latest_fill_log(MARK2_HOME),
    "inbox_files": list_files(MARK2_HOME / "inbox"),
    "outbox_files": list_files(MARK2_HOME / "outbox"),
    "rob_files": [
        file_entry(path)
        for path in sorted(ROB_HOME.iterdir())
        if path.is_file() and path.suffixes[-2:] == [".tar", ".gz"]
    ] if ROB_HOME.exists() else [],
}

print(json.dumps(result))
"""


def load_script_module(script_path: Path, module_name: str):
    loader = importlib.machinery.SourceFileLoader(module_name, str(script_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return repo_root().parent


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def local_search_roots() -> list[Path]:
    roots = [
        workspace_root() / "storage",
        workspace_root() / "_linode_reclaimed",
        Path.home() / "LenovoBackup",
        Path("/media/joe/LenovoBackup"),
        Path("/mnt/LenovoBackup"),
    ]
    return [root for root in roots if root.exists()]


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_local_file(path: Path, verify_hashes: bool) -> dict[str, Any]:
    st = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size": st.st_size,
        "sha256": _sha256_file(path) if verify_hashes else None,
    }


def find_local_mirrors(remote_file: dict[str, Any], search_roots: list[Path], verify_hashes: bool) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[Path] = set()
    basename = remote_file["name"]
    for root in search_roots:
        for candidate in root.rglob(basename):
            if not candidate.is_file() or candidate in seen:
                continue
            seen.add(candidate)
            local = inventory_local_file(candidate, verify_hashes)
            evidence: list[str] = []
            if local["size"] == remote_file["size"]:
                evidence.append("size")
            if verify_hashes and local["sha256"] and remote_file.get("sha256") and local["sha256"] == remote_file["sha256"]:
                evidence.append("sha256")
            if evidence:
                local["evidence"] = evidence
                matches.append(local)
    return matches


def candidate_delete_records(remote_files: list[dict[str, Any]], search_roots: list[Path], verify_hashes: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for remote_file in remote_files:
        mirrors = find_local_mirrors(remote_file, search_roots, verify_hashes)
        if not mirrors:
            continue
        verified = [mirror for mirror in mirrors if "sha256" in mirror["evidence"]]
        if verified:
            mirrors = verified
        records.append({
            "remote": remote_file,
            "mirrors": mirrors,
        })
    return records


def ssh_probe(host: str, remote_mark2_home: str, remote_rob_home: str, verify_hashes: bool) -> dict[str, Any]:
    proc = subprocess.run(
        ["ssh", host, "python3", "-", remote_mark2_home, remote_rob_home, "1" if verify_hashes else "0"],
        input=REMOTE_PROBE,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"ssh probe failed for {host}")
    return json.loads(proc.stdout)


def summarize_batches(state: dict[str, Any]) -> dict[str, Any]:
    batches = state.get("batches", {})
    counts = Counter(batch.get("status", "?") for batch in batches.values())
    latest = []
    for num_str in sorted(batches.keys(), key=int)[-5:]:
        batch = dict(batches[num_str])
        batch["num"] = int(num_str)
        latest.append(batch)
    return {
        "counts": dict(counts),
        "latest": latest,
        "next_batch": state.get("next_batch", 1),
    }


def render_file_line(file_info: dict[str, Any]) -> str:
    return f"{file_info['path']} ({human_size(int(file_info['size']))})"


def render_report(report: dict[str, Any]) -> str:
    remote = report["remote"]
    lines: list[str] = []

    lines.append("Futon6 Status")
    lines.append(f"checked:      {report['checked_at']}")
    lines.append(f"host:         {remote['hostname']} via {report['host_alias']}")

    manifest = remote["manifest"]
    pending = manifest.get("pending")
    total = manifest.get("total")
    if pending is None or total is None:
        manifest_line = "missing"
    else:
        manifest_line = f"{pending:,}/{total:,} pending"
    lines.append(f"manifest:     {manifest_line}")

    batch_summary = report["batch_summary"]
    ready = batch_summary["counts"].get("inbox", 0)
    results_ready = batch_summary["counts"].get("results-ready", 0)
    lines.append(f"next batch:   batch-{int(batch_summary['next_batch']):03d}")
    lines.append(f"ready for Rob:{ready}")
    lines.append(f"posted by Rob:{results_ready}")

    active = remote["active_builder"]
    lock_info = remote.get("build_lock")
    if active.get("pid"):
        lines.append(f"builder:      pid={active['pid']} {active['cmd']}")
    elif lock_info:
        pid_text = lock_info.get("pid", "?")
        lines.append(f"builder:      none active; stale build.lock pid={pid_text}")
    else:
        lines.append("builder:      none active")

    latest_log = remote.get("latest_fill_log")
    if latest_log:
        tail = latest_log.get("tail") or []
        last_line = tail[-1] if tail else "(empty log)"
        lines.append(f"last fill:    {Path(latest_log['path']).name} :: {last_line}")

    lines.append("")
    lines.append("Remote Ready")
    if remote["inbox_files"]:
        for file_info in remote["inbox_files"]:
            lines.append(f"- {render_file_line(file_info)}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Remote Results")
    if remote["outbox_files"]:
        for file_info in remote["outbox_files"]:
            lines.append(f"- {render_file_line(file_info)}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Recent Batches")
    for batch in batch_summary["latest"]:
        when = batch.get("collected_at") or batch.get("returned_at") or batch.get("pulled_at") or batch.get("created_at") or "?"
        lines.append(
            f"- batch-{batch['num']:03d} {batch.get('status', '?')} "
            f"ok={batch.get('ok', '?')} failed={batch.get('failed', '?')} when={when}"
        )

    lines.append("")
    lines.append("Delete Candidates")
    any_candidates = False
    for section_name, records in (
        ("mark2 live lane", report["delete_candidates"]["mark2"]),
        ("rob-home transfer archives", report["delete_candidates"]["rob"]),
    ):
        lines.append(f"{section_name}:")
        if not records:
            lines.append("- none")
            continue
        any_candidates = True
        for record in records:
            remote_file = record["remote"]
            privilege = "sudo/rob required" if remote_file["path"].startswith("/home/rob/") else "joe can remove"
            lines.append(f"- {render_file_line(remote_file)} [{privilege}]")
            for mirror in record["mirrors"]:
                evidence = "+".join(mirror["evidence"])
                lines.append(f"  mirror: {mirror['path']} [{evidence}]")
    if not any_candidates:
        lines.append("- no mirrored remote files found")

    delete_commands = report["delete_commands"]
    lines.append("")
    lines.append("Delete Commands")
    if delete_commands["joe"]:
        lines.append(f"- joe: {delete_commands['joe']}")
    if delete_commands["sudo"]:
        lines.append(f"- sudo: {delete_commands['sudo']}")
    if not delete_commands["joe"] and not delete_commands["sudo"]:
        lines.append("- none")

    return "\n".join(lines)


def build_delete_commands(host: str, delete_candidates: dict[str, list[dict[str, Any]]]) -> dict[str, str | None]:
    joe_paths: list[str] = []
    sudo_paths: list[str] = []
    for records in delete_candidates.values():
        for record in records:
            remote_path = record["remote"]["path"]
            if remote_path.startswith("/home/rob/"):
                sudo_paths.append(remote_path)
            else:
                joe_paths.append(remote_path)

    joe_cmd = None
    sudo_cmd = None
    if joe_paths:
        joined = " ".join(shlex.quote(path) for path in joe_paths)
        joe_cmd = f"ssh {shlex.quote(host)} rm -v -- {joined}"
    if sudo_paths:
        joined = " ".join(shlex.quote(path) for path in sudo_paths)
        sudo_cmd = f"ssh -t {shlex.quote(host)} sudo rm -v -- {joined}"
    return {"joe": joe_cmd, "sudo": sudo_cmd}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    remote = ssh_probe(args.host, args.remote_mark2_home, args.remote_rob_home, not args.no_hash)
    batch_summary = summarize_batches(remote["state"])
    search_roots = local_search_roots()
    mark2_delete_candidates = candidate_delete_records(
        remote["inbox_files"] + remote["outbox_files"],
        search_roots,
        not args.no_hash,
    )
    rob_delete_candidates = candidate_delete_records(
        remote["rob_files"],
        search_roots,
        not args.no_hash,
    )
    delete_candidates = {"mark2": mark2_delete_candidates, "rob": rob_delete_candidates}
    return {
        "checked_at": remote["checked_at"],
        "host_alias": args.host,
        "remote": remote,
        "batch_summary": batch_summary,
        "delete_candidates": delete_candidates,
        "delete_commands": build_delete_commands(args.host, delete_candidates),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="futon6-status.py",
        description="Summarize the live futon6/mark2 lane from local and Linode evidence.",
    )
    ap.add_argument("--host", default=os.environ.get("FUTON6_STATUS_HOST", "linode-chicago"))
    ap.add_argument("--remote-mark2-home", default="/home/joe/mark2")
    ap.add_argument("--remote-rob-home", default="/home/rob")
    ap.add_argument("--no-hash", action="store_true", help="Skip sha256 verification and match on size only.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
