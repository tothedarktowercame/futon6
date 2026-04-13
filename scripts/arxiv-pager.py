#!/usr/bin/env python3
"""arXiv pager — fetch papers in batches for superpod processing.

Downloads arXiv eprint sources in pages of N papers (default 5000,
the arXiv API courtesy limit). Each page becomes a self-contained
tarball that Rob can scp to the superpod and run through the pipeline.

The manifest DB tracks state: pending → paging → staged → (processed).
Re-running is safe — it picks up where it left off.

Usage:
    # Fetch next batch (default 5000)
    python3 scripts/arxiv-pager.py

    # Smaller batch for testing
    python3 scripts/arxiv-pager.py --page-size 100

    # Continuous mode: fetch pages until manifest is exhausted
    python3 scripts/arxiv-pager.py --continuous

    # Dry run: show what would be fetched
    python3 scripts/arxiv-pager.py --dry-run

Output structure (per batch):
    staging/batch-001/
        batch-001.jsonl          # metadata for pipeline --arxiv-jsonl
        eprints/                 # source tarballs (.tar.gz, .tex, etc.)
    staging/batch-001.tar.gz     # ready for scp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path.home() / "code" / "storage" / "arxiv-manifest" / "arxiv_manifest.sqlite"
DEFAULT_STAGING = REPO_ROOT / "staging"
DEFAULT_PAGE_SIZE = 5000
DEFAULT_RATE_LIMIT = 3.0  # seconds between requests (arXiv courtesy)
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 4
USER_AGENT = "futon6-arxiv-pager/1.0 (+https://github.com/tothedarktowercame/futon6)"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(arxiv_id: str) -> str:
    s = arxiv_id.replace("/", "__")
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def infer_extension(content_type: str, payload: bytes) -> str:
    ct = (content_type or "").lower()
    if "gzip" in ct or "x-gzip" in ct:
        return ".tar.gz"
    if "x-eprint-tar" in ct or "x-tar" in ct or "application/tar" in ct:
        return ".tar"
    if "x-tex" in ct or "text/plain" in ct:
        return ".tex"
    if len(payload) >= 2 and payload[:2] == b"\x1f\x8b":
        return ".tar.gz"
    if len(payload) > 265 and payload[257:262] == b"ustar":
        return ".tar"
    head = payload[:1024].lstrip()
    if head.startswith(b"\\documentclass") or head.startswith(b"\\input"):
        return ".tex"
    return ".bin"


def fetch_with_retries(url: str, timeout: int, retries: int) -> tuple[bytes, str]:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "")
            if ctype.lower().startswith("text/html"):
                head = body[:512].decode("utf-8", errors="ignore").lower()
                if "error" in head or "unavailable" in head or "captcha" in head:
                    raise RuntimeError("received html error payload")
            return body, ctype
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, RuntimeError) as exc:
            last_err = exc
            if attempt >= retries:
                break
            backoff = min(30.0, 2.0 * attempt)
            time.sleep(backoff)
    assert last_err is not None
    raise last_err


# -- Manifest DB operations --------------------------------------------------

def next_batch_number(db: sqlite3.Connection) -> int:
    """Find the next batch number by looking at what's already staged."""
    row = db.execute(
        "SELECT MAX(CAST(SUBSTR(status, 8) AS INTEGER)) FROM papers "
        "WHERE status LIKE 'batch-%'"
    ).fetchone()
    if row[0] is not None:
        return row[0] + 1
    # Also check staging directory for existing batches
    return 1


def claim_page(db: sqlite3.Connection, page_size: int, batch_num: int) -> list[dict]:
    """Claim the next page of pending papers. Returns metadata rows."""
    status_tag = f"batch-{batch_num:03d}"

    # Select pending papers ordered by creation date
    rows = db.execute(
        "SELECT arxiv_id, version, id_with_version, title, abstract, "
        "  authors_json, categories_json, primary_category, "
        "  created, updated, abs_url, eprint_url "
        "FROM papers "
        "WHERE latest = 1 AND include = 1 AND status = 'pending' "
        "ORDER BY created ASC, arxiv_id ASC "
        "LIMIT ?",
        (page_size,),
    ).fetchall()

    if not rows:
        return []

    # Mark them as claimed
    ids = [(r[0], r[1]) for r in rows]
    db.executemany(
        "UPDATE papers SET status = ?, last_attempt_at = ? "
        "WHERE arxiv_id = ? AND version = ?",
        [(status_tag, now_iso(), aid, ver) for aid, ver in ids],
    )
    db.commit()

    # Build metadata dicts
    papers = []
    for (arxiv_id, version, idv, title, abstract, authors_json,
         categories_json, primary_category, created, updated,
         abs_url, eprint_url) in rows:
        papers.append({
            "id": idv,
            "base_id": arxiv_id,
            "version": version,
            "title": title or "",
            "abstract": abstract or "",
            "authors": json.loads(authors_json or "[]"),
            "categories": json.loads(categories_json or "[]"),
            "primary_category": primary_category or "",
            "date": created or "",
            "updated": updated or "",
            "url": abs_url,
            "eprint_url": eprint_url,
        })
    return papers


def mark_fetched(db: sqlite3.Connection, arxiv_id: str, version: int,
                 sha256: str, raw_bytes: int) -> None:
    db.execute(
        "UPDATE papers SET fetched_at = ?, sha256 = ?, raw_bytes = ? "
        "WHERE arxiv_id = ? AND version = ?",
        (now_iso(), sha256, raw_bytes, arxiv_id, version),
    )


def mark_error(db: sqlite3.Connection, arxiv_id: str, version: int,
               error_msg: str) -> None:
    db.execute(
        "UPDATE papers SET attempts = attempts + 1, "
        "  last_attempt_at = ?, error_message = ? "
        "WHERE arxiv_id = ? AND version = ?",
        (now_iso(), error_msg, arxiv_id, version),
    )


def mark_batch_staged(db: sqlite3.Connection, batch_num: int) -> None:
    """Mark all papers in this batch as staged (ready for superpod)."""
    status_tag = f"batch-{batch_num:03d}"
    staged_tag = f"staged-{batch_num:03d}"
    db.execute(
        "UPDATE papers SET status = ? WHERE status = ?",
        (staged_tag, status_tag),
    )
    db.commit()


# -- Batch operations ---------------------------------------------------------

def fetch_batch(
    papers: list[dict],
    eprint_dir: Path,
    db: sqlite3.Connection,
    rate_limit: float,
    timeout: int,
    retries: int,
) -> tuple[int, int]:
    """Download eprint sources for a batch. Returns (ok, failed)."""
    eprint_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed = 0
    last_request_at = 0.0
    t0 = time.time()

    for i, paper in enumerate(papers, start=1):
        arxiv_id = paper["base_id"]
        version = paper["version"]
        eprint_url = paper["eprint_url"]
        sid = safe_id(paper["id"])

        # Rate limit
        wait = rate_limit - (time.time() - last_request_at)
        if wait > 0:
            time.sleep(wait)

        try:
            payload, ctype = fetch_with_retries(eprint_url, timeout, retries)
            last_request_at = time.time()

            ext = infer_extension(ctype, payload)
            out_path = eprint_dir / f"{sid}{ext}"
            tmp_path = eprint_dir / f".{sid}.tmp"
            tmp_path.write_bytes(payload)
            tmp_path.replace(out_path)

            digest = hashlib.sha256(payload).hexdigest()
            mark_fetched(db, arxiv_id, version, digest, len(payload))
            ok += 1

        except Exception as exc:
            last_request_at = time.time()
            mark_error(db, arxiv_id, version, str(exc))
            failed += 1

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"  [{i}/{len(papers)}] ok={ok} failed={failed} "
                  f"({elapsed:.0f}s, {rate:.2f} req/s)")

    db.commit()
    elapsed = time.time() - t0
    print(f"  fetch complete: ok={ok} failed={failed} ({elapsed:.0f}s)")
    return ok, failed


def write_metadata_jsonl(papers: list[dict], path: Path) -> None:
    """Write pipeline-compatible JSONL metadata."""
    with path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")
    print(f"  wrote {len(papers)} rows -> {path}")


def create_tarball(batch_dir: Path, tar_path: Path) -> None:
    """Create a gzipped tarball of the batch directory."""
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(batch_dir, arcname=batch_dir.name)
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"  tarball: {tar_path} ({size_mb:.1f} MB)")


def run_one_page(
    db: sqlite3.Connection,
    staging_dir: Path,
    page_size: int,
    rate_limit: float,
    timeout: int,
    retries: int,
    dry_run: bool,
) -> bool:
    """Fetch one page. Returns True if there are more pages."""
    batch_num = next_batch_number(db)
    batch_name = f"batch-{batch_num:03d}"
    batch_dir = staging_dir / batch_name
    tar_path = staging_dir / f"{batch_name}.tar.gz"

    # Check for already-completed batch
    if tar_path.exists():
        print(f"[pager] {tar_path} already exists, skipping to next")
        return True

    print(f"\n[pager] === {batch_name} ===")

    # Claim papers
    papers = claim_page(db, page_size, batch_num)
    if not papers:
        print("[pager] no more pending papers")
        return False

    remaining = db.execute(
        "SELECT COUNT(*) FROM papers "
        "WHERE latest = 1 AND include = 1 AND status = 'pending'"
    ).fetchone()[0]

    cats = {}
    for p in papers:
        c = p.get("primary_category", "?")
        cats[c] = cats.get(c, 0) + 1
    top_cats = sorted(cats.items(), key=lambda x: -x[1])[:5]
    cat_summary = ", ".join(f"{c}:{n}" for c, n in top_cats)

    print(f"  claimed {len(papers)} papers ({remaining} remaining)")
    print(f"  categories: {cat_summary}")

    if dry_run:
        # Release the claim
        status_tag = f"batch-{batch_num:03d}"
        db.execute(
            "UPDATE papers SET status = 'pending' WHERE status = ?",
            (status_tag,),
        )
        db.commit()
        print("  [dry-run] released claim")
        return remaining > 0

    # Create batch directory
    batch_dir.mkdir(parents=True, exist_ok=True)
    eprint_dir = batch_dir / "eprints"

    # Write metadata JSONL
    jsonl_path = batch_dir / f"{batch_name}.jsonl"
    write_metadata_jsonl(papers, jsonl_path)

    # Fetch eprint sources
    ok, failed = fetch_batch(
        papers, eprint_dir, db,
        rate_limit=rate_limit,
        timeout=timeout,
        retries=retries,
    )

    # Write batch summary
    summary = {
        "batch": batch_name,
        "created_at": now_iso(),
        "total": len(papers),
        "fetched_ok": ok,
        "fetched_failed": failed,
        "categories": cats,
    }
    summary_path = batch_dir / "batch-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Create tarball
    create_tarball(batch_dir, tar_path)

    # Mark batch as staged
    mark_batch_staged(db, batch_num)
    print(f"  {batch_name} staged and ready for scp")

    return remaining > len(papers)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch arXiv papers in batches for superpod processing")
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help="Path to arxiv_manifest.sqlite")
    ap.add_argument("--staging-dir", default=str(DEFAULT_STAGING),
                    help="Directory for batch output")
    ap.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE,
                    help=f"Papers per batch (default: {DEFAULT_PAGE_SIZE})")
    ap.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT,
                    help=f"Seconds between requests (default: {DEFAULT_RATE_LIMIT})")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"Request timeout seconds (default: {DEFAULT_TIMEOUT})")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                    help=f"Retries per download (default: {DEFAULT_RETRIES})")
    ap.add_argument("--continuous", action="store_true",
                    help="Keep fetching pages until manifest is exhausted")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be fetched without downloading")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[pager] manifest DB not found: {db_path}", file=sys.stderr)
        return 1

    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")

    pending = db.execute(
        "SELECT COUNT(*) FROM papers "
        "WHERE latest = 1 AND include = 1 AND status = 'pending'"
    ).fetchone()[0]
    total = db.execute(
        "SELECT COUNT(*) FROM papers WHERE latest = 1 AND include = 1"
    ).fetchone()[0]
    print(f"[pager] manifest: {pending}/{total} pending")
    print(f"[pager] staging:  {staging_dir}")
    print(f"[pager] page size: {args.page_size}")

    if pending == 0:
        print("[pager] nothing to do")
        return 0

    has_more = True
    pages_done = 0
    while has_more:
        has_more = run_one_page(
            db, staging_dir, args.page_size,
            rate_limit=args.rate_limit,
            timeout=args.timeout,
            retries=args.retries,
            dry_run=args.dry_run,
        )
        pages_done += 1
        if not args.continuous:
            break

    db.close()
    print(f"\n[pager] done ({pages_done} page(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
