#!/usr/bin/env python3
"""Add a validated authority and checksummed inventory to an existing substrate.

Preserves existing regular-file payloads; refuses links and unsafe paths. The
source index is shipped verbatim. Output is deterministic for identical inputs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from concept_authority import ConceptAuthority

AUTHORITY = "futon6/data/background-corpus-index.json"
MANIFEST = "futon6/data/mark7-substrate-manifest.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(base: Path, index: Path, output: Path) -> dict:
    ConceptAuthority(index)  # fail before creating or replacing any archive
    raw_index = index.read_bytes()
    index_doc = json.loads(raw_index)
    payload = {}
    with tarfile.open(base, "r:gz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"non-regular member: {member.name}; dereference at source")
            if member.name in (AUTHORITY, MANIFEST):
                continue
            if member.name in payload:
                raise ValueError(f"duplicate member: {member.name}")
            payload[member.name] = source.extractfile(member).read()
    payload[AUTHORITY] = raw_index
    manifest = {
        "schema-version": 1,
        "base-archive": {"name": base.name, "sha256": digest(base.read_bytes())},
        "authority": {
            "path": AUTHORITY,
            "metadata": {k: v for k, v in index_doc.items()
                         if k not in ("terms", "candidate-terms")},
            "term-keys": len(index_doc["terms"]),
            "builder": "scripts/background_corpus_index.py",
            "provenance-limit": "Original input hashes were not recorded in this index; "
                                "metadata describes the existing materialization, not a new rebuild.",
        },
        "files": {name: {"sha256": digest(data), "bytes": len(data)}
                  for name, data in sorted(payload.items())},
    }
    payload[MANIFEST] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=".substrate-", suffix=".tgz")
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as archive:
                    for name, data in sorted(payload.items()):
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        member.mode = 0o644
                        archive.addfile(member, io.BytesIO(data))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest(output.read_bytes())}  {output.name}\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.base, args.index, args.output)
    print(f"{args.output}: {len(manifest['files'])} files; "
          f"{manifest['authority']['term-keys']} authority terms")


if __name__ == "__main__":
    main()
