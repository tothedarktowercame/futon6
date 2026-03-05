#!/usr/bin/env python3
"""arXiv math.CT loader + manifest generator for Tickle.

Phase 1 notes (2026-02-24):
- Raw eprints live at /home/joe/code/futon6/data/arxiv-math-ct-eprints/ (9,795
  gzipped payloads referenced by data/arxiv-math-ct-file-index.jsonl).
- Each index row (9,916 total) looks like:
  {"id": "math/9503217", "safe_id": "math__9503217", "local_file":
   "data/arxiv-math-ct-eprints/math__9503217.tar.gz", ...}.
- Metadata rows live in data/arxiv-math-ct-metadata.jsonl and include
  title/authors/categories/abstract per paper.
- Formats: mostly .tar.gz tarballs containing one or more .tex files, with a
  subset that are simple gzip-compressed .tex payloads (tar headers absent).
  ~50 entries are .bin fetches which we currently skip (body text empty).
- Manifest target lives in data/arxiv-math-ct/entities.json (same schema as
  data/ct-validation/entities.json but Article-focused fields only).

load_arxiv_entries(data_dir) returns paper dicts with:
  entity_id, title, source_file, body_text, body_length, arxiv_id,
  categories, authors. CLI helpers:
    --count      → print # of entries discovered
    --manifest   → materialise data/arxiv-math-ct/entities.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
INDEX_PATH = DATA_ROOT / "arxiv-math-ct-file-index.jsonl"
METADATA_PATH = DATA_ROOT / "arxiv-math-ct-metadata.jsonl"
MANIFEST_PATH = DATA_ROOT / "arxiv-math-ct" / "entities.json"
PREVIEW_LENGTH = 200


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_metadata(path: Path) -> Dict[str, dict]:
    return {row["id"]: row for row in _iter_jsonl(path)}


def _pick_tar_member(members: List[tarfile.TarInfo]) -> Optional[tarfile.TarInfo]:
    if not members:
        return None
    tex_members = [m for m in members if m.name.lower().endswith((".tex", ".ltx"))]
    texty = tex_members or [m for m in members if m.name.lower().endswith((".txt", ".texi", ".md"))]
    candidates = texty or members
    return sorted(candidates, key=lambda m: (-m.size, m.name))[0]


def _read_payload(local_path: Path) -> str:
    if not local_path.exists():
        return ""
    if local_path.suffix == ".bin":
        return ""
    # Try tarball first (tarfile handles gzip transparently)
    try:
        with tarfile.open(local_path, mode="r:*") as tf:
            member = _pick_tar_member([m for m in tf.getmembers() if m.isfile()])
            if not member:
                return ""
            extracted = tf.extractfile(member)
            if not extracted:
                return ""
            return extracted.read().decode("utf-8", errors="replace")
    except tarfile.ReadError:
        pass
    # Fall back to raw gzip/plain text
    try:
        with gzip.open(local_path, mode="rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return local_path.read_text(encoding="utf-8", errors="replace")


@dataclass
class ArxivEntry:
    entity_id: str
    title: str
    source_file: str
    body_text: str
    body_length: int
    arxiv_id: str
    categories: List[str]
    authors: List[str]

    def as_manifest_row(self) -> dict:
        preview = " ".join(self.body_text.split())[:PREVIEW_LENGTH]
        return {
            "entity_id": self.entity_id,
            "source_file": self.source_file,
            "title": self.title,
            "type": "Article",
            "body_length": self.body_length,
            "arxiv_id": self.arxiv_id,
            "categories": self.categories,
            "authors": self.authors,
            "ner_count": 0,
            "scope_count": 0,
            "wire_count": 0,
            "port_count": 0,
            "body_preview": preview,
        }


def iter_arxiv_entries(data_dir: str | Path, *, limit: Optional[int] = None) -> Iterator[ArxivEntry]:
    data_dir = Path(data_dir)
    index_path = data_dir / "arxiv-math-ct-file-index.jsonl" if (data_dir / "arxiv-math-ct-file-index.jsonl").exists() else INDEX_PATH
    metadata_path = data_dir / "arxiv-math-ct-metadata.jsonl" if (data_dir / "arxiv-math-ct-metadata.jsonl").exists() else METADATA_PATH
    metadata = _load_metadata(metadata_path)

    produced = 0
    for row in _iter_jsonl(index_path):
        if limit is not None and produced >= limit:
            break
        if not row.get("has_local_file"):
            continue
        arxiv_id = row["id"]
        safe_id = row.get("safe_id", arxiv_id.replace("/", "__"))
        entity_id = f"arxiv-{safe_id}"
        local_file = row.get("local_file") or row.get("path")
        if not local_file:
            continue
        local_path = Path(local_file)
        if not local_path.is_absolute():
            candidate = data_dir / local_path
            if candidate.exists():
                local_path = candidate
            else:
                local_path = ROOT / local_path
        body_text = _read_payload(local_path)
        body_length = len(body_text)
        meta = metadata.get(arxiv_id, {})
        title = meta.get("title") or row.get("title") or f"arXiv {arxiv_id}"
        categories = meta.get("categories") or row.get("categories") or []
        authors = meta.get("authors") or []

        try:
            rel_path = str(local_path.relative_to(ROOT))
        except ValueError:
            rel_path = str(local_path)

        yield ArxivEntry(
            entity_id=entity_id,
            title=title,
            source_file=rel_path,
            body_text=body_text,
            body_length=body_length,
            arxiv_id=arxiv_id,
            categories=categories,
            authors=authors,
        )
        produced += 1


def load_arxiv_entries(data_dir: str) -> List[dict]:
    """Load arXiv math.CT entries from DATA_DIR -> list[dict]."""
    return [entry.__dict__ for entry in iter_arxiv_entries(data_dir)]


def write_manifest(entries: Iterable[ArxivEntry], manifest_path: Path) -> None:
    manifest = [entry.as_manifest_row() for entry in entries]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[loader] wrote {len(manifest)} rows to {manifest_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="arXiv math.CT loader")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Directory containing arxiv-math-ct-* files")
    parser.add_argument("--count", action="store_true", help="Print number of entries discovered")
    parser.add_argument("--manifest", action="store_true", help="Generate the entities.json manifest")
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH), help="Output path for manifest")
    parser.add_argument("--limit", type=int, help="Process at most N entries (debug)")
    args = parser.parse_args()

    buffer = list(iter_arxiv_entries(args.data_root, limit=args.limit))
    total = len(buffer)

    if args.count:
        print(total)

    if args.manifest:
        write_manifest(buffer, Path(args.manifest_path))

    if not args.count and not args.manifest:
        if buffer:
            sample = buffer[0]
            print(json.dumps(sample.__dict__, indent=2)[:2000])
            if total > 1:
                print(f"\n... ({total - 1} more entries)")
        else:
            print("No entries found.")


if __name__ == "__main__":
    cli()
