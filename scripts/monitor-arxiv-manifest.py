#!/usr/bin/env python3
"""Monitor arXiv manifest harvest/download progress.

Reads futon6 manifest SQLite + optional harvest.log and prints:
- current harvest progress snapshot
- processing rates
- ETA (if a goal row count is provided)
- downloader queue state from the DB
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "arxiv-manifest" / "arxiv_manifest.sqlite"
DEFAULT_LOG = REPO_ROOT / "data" / "arxiv-manifest" / "harvest.log"

HARVEST_RE = re.compile(
    r"\[harvest\]\s+page=(\d+)\s+records=(\d+)\s+seen=(\d+)\s+"
    r"written=(\d+)\s+skipped=(\d+)\s+token=(yes|no)"
)
DONE_RE = re.compile(r"\[harvest\]\s+done\s+run_id=(\d+)\s+seen=(\d+)\s+written=(\d+)\s+skipped=(\d+)")


@dataclass
class HarvestSnapshot:
    page: int
    records: int
    seen: int
    written: int
    skipped: int
    token_yes: bool


def parse_sqlite_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    ts = ts.strip()
    if not ts:
        return None
    # sqlite datetime('now') yields "YYYY-MM-DD HH:MM:SS" (UTC)
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # fallback for ISO strings
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d > 0:
        return f"{d}d {h:02d}h {m:02d}m"
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def tail_lines(path: Path, max_bytes: int = 512_000) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    return text.splitlines()


def parse_log_snapshot(log_path: Path) -> tuple[HarvestSnapshot | None, int | None]:
    lines = tail_lines(log_path)
    snap: HarvestSnapshot | None = None
    done_run_id: int | None = None
    for ln in lines:
        d = DONE_RE.search(ln)
        if d:
            done_run_id = int(d.group(1))
        m = HARVEST_RE.search(ln)
        if m:
            snap = HarvestSnapshot(
                page=int(m.group(1)),
                records=int(m.group(2)),
                seen=int(m.group(3)),
                written=int(m.group(4)),
                skipped=int(m.group(5)),
                token_yes=(m.group(6) == "yes"),
            )
    return snap, done_run_id


def safe_rate(numer: float, denom: float) -> float | None:
    if denom <= 0:
        return None
    return numer / denom


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor arXiv manifest progress from DB/log")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="Path to arxiv_manifest.sqlite")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="Path to harvest.log")
    ap.add_argument(
        "--goal-latest",
        type=int,
        default=0,
        help="Optional target count for latest rows (enables ETA)",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    log_path = Path(args.log)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    now = datetime.now(timezone.utc)
    with sqlite3.connect(db_path) as conn:
        latest_run = conn.execute(
            """
            SELECT run_id, started_at, finished_at, records_seen, rows_written, rows_skipped, notes
            FROM harvest_runs
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()

        total_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        latest_rows = conn.execute("SELECT COUNT(*) FROM papers WHERE latest=1").fetchone()[0]
        latest_included = conn.execute("SELECT COUNT(*) FROM papers WHERE latest=1 AND include=1").fetchone()[0]
        queue_pending = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE latest=1 AND include=1 AND status IN ('pending','error')"
        ).fetchone()[0]
        queue_downloaded = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE latest=1 AND include=1 AND status='downloaded'"
        ).fetchone()[0]

        token_row = conn.execute(
            "SELECT state_value, updated_at FROM harvest_state WHERE state_key='oai_resumption_token'"
        ).fetchone()

    snap, done_run_id = parse_log_snapshot(log_path)

    print("== Harvest Monitor ==")
    print(f"db: {db_path}")
    print(f"log: {log_path} ({'exists' if log_path.exists() else 'missing'})")
    print()

    if latest_run is None:
        print("No harvest_runs rows yet.")
        return 0

    run_id, started_at_s, finished_at_s, run_seen, run_written, run_skipped, notes = latest_run
    started_at = parse_sqlite_ts(started_at_s)
    finished_at = parse_sqlite_ts(finished_at_s)
    elapsed = None
    if started_at:
        end_time = finished_at or now
        elapsed = (end_time - started_at).total_seconds()
    active = finished_at is None and done_run_id != run_id

    print(f"run_id: {run_id}")
    print(f"started_at_utc: {started_at_s}")
    print(f"finished_at_utc: {finished_at_s or 'running'}")
    print(f"elapsed: {fmt_duration(elapsed)}")
    print(f"status: {'running' if active else 'finished/unknown'}")

    if snap:
        print(
            f"log_snapshot: page={snap.page} seen={snap.seen} written={snap.written} "
            f"skipped={snap.skipped} token={'yes' if snap.token_yes else 'no'}"
        )
    else:
        print("log_snapshot: n/a")

    if token_row:
        tok_val, tok_updated = token_row
        tok_state = "set" if tok_val else "empty"
        print(f"resumption_token: {tok_state} (updated {tok_updated})")

    print()
    print("== DB Counts ==")
    print(f"papers_total_rows: {total_rows}")
    print(f"papers_latest_rows: {latest_rows}")
    print(f"papers_latest_included: {latest_included}")

    # Prefer log snapshot if available during active harvest.
    progress_written = snap.written if (active and snap) else run_written
    progress_seen = snap.seen if (active and snap) else run_seen
    progress_skipped = snap.skipped if (active and snap) else run_skipped

    print()
    print("== Harvest Progress ==")
    print(f"seen_records: {progress_seen}")
    print(f"written_rows: {progress_written}")
    print(f"skipped_records: {progress_skipped}")

    wr_rate = safe_rate(float(progress_written), float(elapsed or 0))
    seen_rate = safe_rate(float(progress_seen), float(elapsed or 0))

    if wr_rate is not None:
        print(f"write_rate: {wr_rate:.3f} rows/s ({wr_rate*3600:.1f} rows/h)")
    else:
        print("write_rate: n/a")

    if seen_rate is not None:
        print(f"scan_rate: {seen_rate:.3f} rec/s ({seen_rate*3600:.1f} rec/h)")
    else:
        print("scan_rate: n/a")

    goal = args.goal_latest if args.goal_latest and args.goal_latest > 0 else None
    if goal:
        remaining = max(0, goal - latest_rows)
        eta_sec = (remaining / wr_rate) if (wr_rate and wr_rate > 0) else None
        print(f"goal_latest_rows: {goal}")
        print(f"remaining_to_goal: {remaining}")
        print(f"eta_to_goal: {fmt_duration(eta_sec)}")
    else:
        print("eta_to_goal: n/a (set --goal-latest)")

    print()
    print("== Download Queue ==")
    print(f"downloaded_latest_included: {queue_downloaded}")
    print(f"pending_or_error_latest_included: {queue_pending}")

    if notes:
        print()
        print("== Run Notes ==")
        print(notes.strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
