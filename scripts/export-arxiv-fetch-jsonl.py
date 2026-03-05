#!/usr/bin/env python3
"""Export futon6 downloader input JSONL from manifest DB.

Produces rows compatible with scripts/fetch-arxiv-eprints.py:
required keys are `id` and `eprint_url`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "arxiv-manifest" / "arxiv_manifest.sqlite"
DEFAULT_OUT = REPO_ROOT / "data" / "arxiv-manifest" / "arxiv-math-manifest-fetch.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="Export queue rows to futon6 fetch JSONL")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--status", default="pending,error", help="Comma-separated statuses")
    ap.add_argument("--latest-only", action="store_true", default=True)
    ap.add_argument("--all-versions", action="store_true")
    ap.add_argument("--include-flag", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    db = Path(args.db)
    out = Path(args.out)
    statuses = [s.strip() for s in args.status.split(",") if s.strip()]

    latest_clause = "AND latest = 1" if args.latest_only and not args.all_versions else ""
    ph = ",".join(["?"] * len(statuses))
    query = f"""
        SELECT arxiv_id, version, id_with_version, title, abstract, authors_json,
               categories_json, created, updated, abs_url, eprint_url
        FROM papers
        WHERE include = ?
          AND status IN ({ph})
          {latest_clause}
        ORDER BY created ASC, arxiv_id ASC, version ASC
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn, out.open("w", encoding="utf-8") as f:
        rows = conn.execute(query, (args.include_flag, *statuses)).fetchall()
        for (
            arxiv_id,
            version,
            idv,
            title,
            abstract,
            authors_json,
            categories_json,
            created,
            updated,
            abs_url,
            eprint_url,
        ) in rows:
            authors = json.loads(authors_json or "[]")
            categories = json.loads(categories_json or "[]")
            obj = {
                "id": idv,
                "base_id": arxiv_id,
                "version": version,
                "title": title or "",
                "abstract": abstract or "",
                "authors": authors,
                "categories": categories,
                "date": created or "",
                "updated": updated or "",
                "url": abs_url,
                "eprint_url": eprint_url,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[export] wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
