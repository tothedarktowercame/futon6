#!/usr/bin/env python3
"""Initialize SQLite manifest DB for arXiv harvesting + downloader queueing.

This schema is designed to interoperate with futon6's existing downloader
(`scripts/fetch-arxiv-eprints.py`), while adding robust queue bookkeeping.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "arxiv-manifest" / "arxiv_manifest.sqlite"

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    id_with_version TEXT NOT NULL,
    created TEXT,
    updated TEXT,
    oai_datestamp TEXT,
    title TEXT,
    abstract TEXT,
    authors_json TEXT,
    categories_json TEXT,
    primary_category TEXT,
    set_specs_json TEXT,
    doi TEXT,
    license TEXT,
    is_withdrawn INTEGER NOT NULL DEFAULT 0,
    abs_url TEXT NOT NULL,
    eprint_url TEXT NOT NULL,
    latest INTEGER NOT NULL DEFAULT 0,
    include INTEGER NOT NULL DEFAULT 1,

    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    fetched_at TEXT,
    http_status INTEGER,
    raw_bytes INTEGER,
    sha256 TEXT,
    local_path TEXT,
    error_code TEXT,
    error_message TEXT,

    source TEXT NOT NULL DEFAULT 'oai',
    harvested_at TEXT NOT NULL DEFAULT (datetime('now')),

    PRIMARY KEY (arxiv_id, version)
);

CREATE INDEX IF NOT EXISTS idx_papers_latest_include_status
    ON papers (latest, include, status);

CREATE INDEX IF NOT EXISTS idx_papers_primary_category
    ON papers (primary_category);

CREATE INDEX IF NOT EXISTS idx_papers_created
    ON papers (created);

CREATE TABLE IF NOT EXISTS harvest_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT,
    base_url TEXT NOT NULL,
    oai_set TEXT NOT NULL,
    metadata_prefix TEXT NOT NULL,
    from_utc TEXT,
    until_utc TEXT,
    latest_only INTEGER NOT NULL,
    include_crosslists INTEGER NOT NULL,
    records_seen INTEGER NOT NULL DEFAULT 0,
    rows_written INTEGER NOT NULL DEFAULT 0,
    rows_updated INTEGER NOT NULL DEFAULT 0,
    rows_skipped INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS harvest_state (
    state_key TEXT PRIMARY KEY,
    state_value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIEW IF NOT EXISTS download_queue AS
SELECT
    arxiv_id,
    version,
    id_with_version,
    created,
    updated,
    title,
    primary_category,
    abs_url,
    eprint_url,
    status,
    attempts,
    last_attempt_at
FROM papers
WHERE include = 1
  AND latest = 1
  AND status IN ('pending', 'error')
ORDER BY created ASC, arxiv_id ASC;
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def main() -> int:
    ap = argparse.ArgumentParser(description="Initialize arXiv manifest SQLite DB")
    ap.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="Path to SQLite DB",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    init_db(db_path)
    print(f"[init] initialized DB: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
