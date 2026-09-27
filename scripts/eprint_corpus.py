#!/usr/bin/env python3
"""Where a corpus's e-prints are, and which of them the corpus consists of.

Locating the corpus is its own concern, separate from parsing what is inside an archive:
the spine's stages all need to enumerate and resolve papers, none of them needs the
anatomy sweep to do it.

A corpus is TWO things, each configurable as a file:

- its SOURCES (`FUTON6_EPRINTS`, a directory, or a file listing directories and/or
  archive paths). An acquired corpus arrives as many batch directories. Copying tens of
  thousands of archives into one directory to satisfy a reader is slow, and on a
  filesystem without links (exFAT) it is a full second copy of the corpus -- so the
  archives are named where they lie, ideally one path per paper.
- its IDS (`FUTON6_CORPUS_IDS`, a run manifest). One acquisition holds many subjects;
  the roots say where archives are, the manifest says which of them are this corpus.

Two consequences of a multi-directory corpus are decided here, because nothing else can:
a paper held at several versions is read at its LATEST (mining v1 and v2 would count one
paper twice), and a paper held in several roots is read once.
"""

from __future__ import annotations

import re
import sys as _sys
from collections.abc import Sequence
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import futon6_config as config

from pathlib import Path

EPRINT_SUFFIXES = (".tar.gz", ".gz", ".tar", ".tex", ".bin")
ARCHIVE_SUFFIXES = (".tar.gz", ".tex.gz", ".gz", ".tar", ".bin", ".tex")
_VERSION_SUFFIX = re.compile(r"v(\d+)$")


def strip_archive_suffix(path: Path) -> str:
    name = path.name
    for suffix in ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def paper_key(stem: str) -> str:
    """A paper's identity without its version: `0704.0002v2` and `0704.0002v1` are one paper."""
    return _VERSION_SUFFIX.sub("", stem)


def version_of(stem: str) -> int:
    found = _VERSION_SUFFIX.search(stem)
    return int(found.group(1)) if found else 0


def iter_eprints(eprint_dir: Path) -> list[Path]:
    """Every e-print archive in ONE directory, by name."""
    paths = [p for p in eprint_dir.iterdir() if p.is_file() and p.name.endswith(EPRINT_SUFFIXES)]
    return sorted(paths, key=lambda p: p.name)


_INDEX_CACHE: dict[tuple[tuple[str, ...], frozenset[str] | None], dict[str, Path]] = {}


def corpus_index(roots: Sequence[Path] | None = None,
                 ids: frozenset[str] | None = None) -> dict[str, Path]:
    """paper id -> its archive, built once per process.

    Resolution has to be a lookup, not a search: searching the filesystem per paper costs a
    stat per root per suffix and a glob per root, for every paper, in every pass. The best
    input is a LIST OF ARCHIVES, which the corpus's owner can produce from its own records;
    directories are still accepted and listed once.
    """
    roots = tuple(roots) if roots is not None else config.eprint_roots()
    if ids is None:
        ids = config.corpus_ids()
    key = (tuple(str(root) for root in roots), ids)
    held = _INDEX_CACHE.get(key)
    if held is not None:
        return held
    best: dict[str, Path] = {}
    for entry in roots:
        # An entry naming an archive is that paper, read where it lies: a listed corpus costs
        # no directory walk and no filesystem call until a paper is actually opened. Only an
        # entry naming a directory is listed.
        if entry.name.endswith(EPRINT_SUFFIXES):
            archives = [entry]
        elif entry.is_dir():
            archives = iter_eprints(entry)
        else:
            continue
        for archive in archives:
            stem = strip_archive_suffix(archive)
            paper = paper_key(stem)
            if ids is not None and paper not in ids:
                continue
            previous = best.get(paper)
            if previous is None or version_of(stem) > version_of(strip_archive_suffix(previous)):
                best[paper] = archive
    _INDEX_CACHE[key] = best
    return best


def iter_corpus_eprints(roots: Sequence[Path] | None = None,
                        ids: frozenset[str] | None = None) -> list[Path]:
    """Every e-print OF THE CORPUS, across all of the directories that hold it.

    Ordered by paper id rather than by directory, so a corpus reads the same whether it
    is one directory or a hundred batch directories.
    """
    index = corpus_index(roots, ids)
    return [index[paper] for paper in sorted(index)]


def find_corpus_eprint(paper_id: str, roots: Sequence[Path] | None = None) -> Path | None:
    """The archive for one paper, at its latest version. A map lookup, never a search."""
    return corpus_index(roots).get(paper_key(paper_id))


def corpus_paper_ids(roots: Sequence[Path] | None = None,
                     ids: frozenset[str] | None = None) -> list[str]:
    """The corpus's paper ids, version-stripped, in one stable order."""
    return sorted(corpus_index(roots, ids))
