#!/usr/bin/env python3
"""Fetch arXiv e-print sources from harvested metadata JSONL.

Reads input JSONL rows containing at least `id` and `eprint_url`, then
downloads each source package to a local directory with rate limiting.

Usage:
    python3 scripts/fetch-arxiv-eprints.py
    python3 scripts/fetch-arxiv-eprints.py --max 10
    python3 scripts/fetch-arxiv-eprints.py --rate-limit 3.0 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_INPUT = Path("data/arxiv-math-ct-metadata.jsonl")
DEFAULT_OUT = Path("data/arxiv-math-ct-eprints")
DEFAULT_MANIFEST = "fetch-manifest.jsonl"
DEFAULT_RATE_LIMIT_SECONDS = 3.0
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RETRIES = 4


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
    if "x-tex" in ct:
        return ".tex"
    if "text/plain" in ct:
        return ".tex"

    if len(payload) >= 2 and payload[:2] == b"\x1f\x8b":
        return ".tar.gz"
    if len(payload) > 265 and payload[257:262] == b"ustar":
        return ".tar"

    head = payload[:1024].lstrip()
    if head.startswith(b"\\documentclass") or head.startswith(b"\\input"):
        return ".tex"

    return ".bin"


def read_metadata(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {i}: {exc}") from exc
            if "id" not in obj or "eprint_url" not in obj:
                raise ValueError(f"line {i} missing required keys `id`/`eprint_url`")
            rows.append(obj)
    return rows


def fetch_with_retries(
    url: str,
    timeout: int,
    retries: int,
    user_agent: str,
) -> tuple[bytes, str]:
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "")

            # arXiv sometimes returns html error pages while still 200.
            if ctype.lower().startswith("text/html"):
                head = body[:512].decode("utf-8", errors="ignore").lower()
                if "error" in head or "unavailable" in head or "captcha" in head:
                    raise RuntimeError("received html error payload")

            return body, ctype
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_err = exc
            if attempt >= retries:
                break
            backoff = min(30.0, 2.0 * attempt)
            time.sleep(backoff)

    assert last_err is not None
    raise last_err


def append_manifest(manifest_path: Path, row: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch arXiv e-print sources with rate limiting")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input metadata JSONL")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Directory for downloaded sources")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest file name inside out-dir")
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_RATE_LIMIT_SECONDS,
                        help="Minimum seconds between requests (default: 3.0)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="Retry attempts per item")
    parser.add_argument("--max", type=int, default=0,
                        help="Optional cap for this run (0 means all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip IDs with existing files in out-dir")
    parser.add_argument(
        "--user-agent",
        default="futon6-arxiv-eprint-fetch/1.0 (+https://github.com/tothedarktowercame/futon6)",
        help="HTTP User-Agent",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / args.manifest

    if not input_path.exists():
        print(f"[fetch] input not found: {input_path}", file=sys.stderr)
        return 1
    if args.rate_limit < 0:
        print("[fetch] --rate-limit must be >= 0", file=sys.stderr)
        return 1

    rows = read_metadata(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    existing: dict[str, Path] = {}
    for p in out_dir.glob("*"):
        if p.is_file() and p.name != args.manifest:
            stem = p.name
            for suffix in (".tar.gz", ".tar", ".tex", ".bin"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            existing[stem] = p

    print(f"[fetch] input rows: {total}")
    print(f"[fetch] output dir: {out_dir}")
    print(f"[fetch] rate limit: {args.rate_limit:.2f}s/request")

    done = 0
    skipped = 0
    failed = 0
    requested = 0
    run_limit = args.max if args.max and args.max > 0 else None

    last_request_at = 0.0
    t0 = time.time()

    for i, row in enumerate(rows, start=1):
        arxiv_id = str(row["id"])
        eprint_url = str(row["eprint_url"])
        sid = safe_id(arxiv_id)

        if run_limit is not None and requested >= run_limit:
            break

        if args.resume and sid in existing:
            skipped += 1
            if i % 200 == 0:
                print(f"  progress: {i}/{total} scanned, done={done}, skipped={skipped}, failed={failed}")
            continue

        wait = args.rate_limit - (time.time() - last_request_at)
        if wait > 0:
            time.sleep(wait)

        requested += 1
        try:
            payload, ctype = fetch_with_retries(
                eprint_url,
                timeout=args.timeout,
                retries=args.retries,
                user_agent=args.user_agent,
            )
            last_request_at = time.time()

            ext = infer_extension(ctype, payload)
            out_path = out_dir / f"{sid}{ext}"
            tmp_path = out_dir / f".{sid}.tmp"
            tmp_path.write_bytes(payload)
            tmp_path.replace(out_path)

            digest = hashlib.sha256(payload).hexdigest()
            done += 1
            append_manifest(
                manifest_path,
                {
                    "ts": now_iso(),
                    "status": "ok",
                    "id": arxiv_id,
                    "eprint_url": eprint_url,
                    "path": str(out_path),
                    "bytes": len(payload),
                    "content_type": ctype,
                    "sha256": digest,
                },
            )
        except Exception as exc:
            last_request_at = time.time()
            failed += 1
            append_manifest(
                manifest_path,
                {
                    "ts": now_iso(),
                    "status": "error",
                    "id": arxiv_id,
                    "eprint_url": eprint_url,
                    "error": str(exc),
                },
            )

        if requested % 20 == 0:
            elapsed = time.time() - t0
            rate = requested / elapsed if elapsed > 0 else 0.0
            print(
                f"  requested={requested} done={done} failed={failed} skipped={skipped} "
                f"({elapsed:.0f}s elapsed, {rate:.2f} req/s)")

    elapsed = time.time() - t0
    print("\n[fetch] complete")
    print(f"[fetch] requested={requested} done={done} failed={failed} skipped={skipped}")
    print(f"[fetch] manifest={manifest_path}")
    print(f"[fetch] elapsed={elapsed:.1f}s")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
