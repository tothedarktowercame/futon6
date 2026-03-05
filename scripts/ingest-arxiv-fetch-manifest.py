#!/usr/bin/env python3
"""Ingest futon6 fetch-manifest.jsonl into arxiv_manifest.sqlite queue state."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "arxiv-manifest" / "arxiv_manifest.sqlite"


def parse_id_version(raw_id: str) -> tuple[str, int | None]:
    m = re.match(r"^(.*)v(\d+)$", raw_id)
    if not m:
        return raw_id, None
    return m.group(1), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest downloader manifest into papers status")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--fetch-manifest", required=True, help="Path to fetch-manifest.jsonl")
    args = ap.parse_args()

    db = Path(args.db)
    mf = Path(args.fetch_manifest)
    if not db.exists():
        raise FileNotFoundError(db)
    if not mf.exists():
        raise FileNotFoundError(mf)

    ok = 0
    err = 0
    miss = 0

    with sqlite3.connect(db) as conn, mf.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            status = row.get("status", "")
            rid = row.get("id", "")
            base_id, version = parse_id_version(rid)

            if not rid:
                continue

            if version is None:
                # fallback: apply to latest row
                q = "SELECT version FROM papers WHERE arxiv_id=? ORDER BY version DESC LIMIT 1"
                got = conn.execute(q, (base_id,)).fetchone()
                if not got:
                    miss += 1
                    continue
                version = int(got[0])

            exists = conn.execute(
                "SELECT 1 FROM papers WHERE arxiv_id=? AND version=?",
                (base_id, version),
            ).fetchone()
            if not exists:
                miss += 1
                continue

            if status == "ok":
                conn.execute(
                    """
                    UPDATE papers
                    SET status='downloaded',
                        attempts=attempts+1,
                        last_attempt_at=?,
                        fetched_at=?,
                        raw_bytes=?,
                        sha256=?,
                        local_path=?,
                        error_code=NULL,
                        error_message=NULL
                    WHERE arxiv_id=? AND version=?
                    """,
                    (
                        row.get("ts", ""),
                        row.get("ts", ""),
                        row.get("bytes"),
                        row.get("sha256"),
                        row.get("path"),
                        base_id,
                        version,
                    ),
                )
                ok += 1
            elif status == "error":
                conn.execute(
                    """
                    UPDATE papers
                    SET status='error',
                        attempts=attempts+1,
                        last_attempt_at=?,
                        http_status=?,
                        error_code='fetch_error',
                        error_message=?
                    WHERE arxiv_id=? AND version=?
                    """,
                    (
                        row.get("ts", ""),
                        row.get("http_status"),
                        row.get("error", ""),
                        base_id,
                        version,
                    ),
                )
                err += 1

        conn.commit()

    print(f"[ingest] ok={ok} error={err} missing={miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
