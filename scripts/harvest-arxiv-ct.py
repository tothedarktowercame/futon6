#!/usr/bin/env python3
"""Harvest arXiv math.CT metadata via the Atom API.

Downloads title, abstract, authors, categories, and date for all papers
in the math.CT (Category Theory) category. Outputs one JSON object per
line (JSONL), suitable for downstream embedding and corpus-check indexing.

As of Feb 2026 there are ~9,916 papers in math.CT — this takes ~5 minutes
with the required 3-second rate limit between API requests.

Usage:
    python3 scripts/harvest-arxiv-ct.py
    python3 scripts/harvest-arxiv-ct.py --output data/arxiv-ct-metadata.jsonl
    python3 scripts/harvest-arxiv-ct.py --category math.AT  # other categories
    python3 scripts/harvest-arxiv-ct.py --resume  # continue from last page

Output format (one JSON per line):
    {
        "id": "2301.12345",
        "title": "On Higher Categories and ...",
        "abstract": "We study ...",
        "authors": ["Alice", "Bob"],
        "categories": ["math.CT", "math.AT"],
        "date": "2023-01-15",
        "url": "https://arxiv.org/abs/2301.12345",
        "eprint_url": "https://arxiv.org/e-print/2301.12345"
    }

Rate limiting:
    arXiv asks for max 1 request per 3 seconds. This script defaults to
    a 3-second delay. Please do not reduce this.

See: https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}

API_BASE = "http://export.arxiv.org/api/query"
PAGE_SIZE = 100
RATE_LIMIT_SECONDS = 3.0


def fetch_page(category: str, start: int, max_results: int = PAGE_SIZE) -> bytes:
    """Fetch one page of results from the arXiv API."""
    query = urllib.parse.quote(f"cat:{category}")
    url = (
        f"{API_BASE}?search_query={query}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=ascending"
    )
    return urllib.request.urlopen(url, timeout=60).read()


def parse_total(xml_bytes: bytes) -> int:
    """Extract totalResults from an Atom response."""
    root = ET.fromstring(xml_bytes)
    return int(root.findtext("os:totalResults", "0", NS))


def parse_entries(xml_bytes: bytes) -> list[dict]:
    """Parse paper entries from an Atom response."""
    root = ET.fromstring(xml_bytes)
    entries = []
    for entry in root.findall("a:entry", NS):
        raw_id = entry.findtext("a:id", "", NS) or ""
        # Extract arXiv ID from URL like http://arxiv.org/abs/2301.12345v1
        m = re.search(r"/abs/([^\s?]+)", raw_id)
        if not m:
            continue
        arxiv_id = re.sub(r"v\d+$", "", m.group(1))

        title = " ".join((entry.findtext("a:title", "", NS) or "").split())
        abstract = " ".join((entry.findtext("a:summary", "", NS) or "").split())
        date = (entry.findtext("a:published", "", NS) or "")[:10]
        authors = [
            a.findtext("a:name", "", NS)
            for a in entry.findall("a:author", NS)
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("a:category", NS)
        ]

        entries.append({
            "id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": [c for c in categories if c],
            "date": date,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "eprint_url": f"https://arxiv.org/e-print/{arxiv_id}",
        })

    return entries


def count_existing(output_path: Path) -> int:
    """Count lines in an existing JSONL file for resume support."""
    if not output_path.exists():
        return 0
    count = 0
    with open(output_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def harvest(category: str, output_path: Path, resume: bool = False):
    """Harvest all papers in a category, writing JSONL incrementally."""

    # Probe total
    print(f"[harvest] querying arXiv for cat:{category} ...")
    probe = fetch_page(category, 0, 1)
    total = parse_total(probe)
    print(f"[harvest] {total} papers in {category}")

    # Resume support: skip pages we've already fetched
    start = 0
    mode = "w"
    if resume and output_path.exists():
        existing = count_existing(output_path)
        if existing > 0:
            # Round down to nearest page boundary
            start = (existing // PAGE_SIZE) * PAGE_SIZE
            mode = "a"
            print(f"[harvest] resuming from offset {start} "
                  f"({existing} entries already in {output_path})")
    elif output_path.exists() and not resume:
        # Confirm overwrite
        print(f"[harvest] {output_path} exists. Use --resume to continue, "
              f"or delete it to start fresh.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()

    # If resuming, load existing IDs to avoid duplicates
    if mode == "a":
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        seen_ids.add(obj["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    t0 = time.time()
    n_written = len(seen_ids)
    n_pages = 0

    with open(output_path, mode) as fout:
        offset = start
        while offset < total:
            if n_pages > 0:
                time.sleep(RATE_LIMIT_SECONDS)

            try:
                xml_bytes = fetch_page(category, offset, PAGE_SIZE)
                entries = parse_entries(xml_bytes)
            except Exception as e:
                print(f"[harvest] error at offset {offset}: {e}")
                print(f"[harvest] retrying in 10s...")
                time.sleep(10)
                try:
                    xml_bytes = fetch_page(category, offset, PAGE_SIZE)
                    entries = parse_entries(xml_bytes)
                except Exception as e2:
                    print(f"[harvest] retry failed: {e2}. Stopping.")
                    break

            if not entries:
                # API returned empty page — we've reached the end
                break

            for entry in entries:
                if entry["id"] not in seen_ids:
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    seen_ids.add(entry["id"])
                    n_written += 1

            n_pages += 1
            offset += PAGE_SIZE
            elapsed = time.time() - t0
            rate = n_written / elapsed if elapsed > 0 else 0
            eta = (total - n_written) / rate if rate > 0 else 0
            print(f"  page {n_pages}: offset={offset-PAGE_SIZE}, "
                  f"got {len(entries)}, total={n_written}/{total} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            fout.flush()

    elapsed = time.time() - t0
    print(f"\n[harvest] done: {n_written} papers in {elapsed:.0f}s")
    print(f"[harvest] output: {output_path} "
          f"({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Harvest arXiv metadata for a category (default: math.CT)")
    parser.add_argument("--category", default="math.CT",
                        help="arXiv category (default: math.CT)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSONL path (default: data/arxiv-{cat}-metadata.jsonl)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    cat_slug = args.category.replace(".", "-").lower()
    output = Path(args.output or f"data/arxiv-{cat_slug}-metadata.jsonl")

    harvest(args.category, output, resume=args.resume)


if __name__ == "__main__":
    main()
