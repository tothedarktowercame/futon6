#!/usr/bin/env python3
"""Which files in a run directory are the artifacts, and which are sidecars.

Third shared helper extracted for the same reason as `edn_compat` (reader
divergence) and `paper_ids` (id families): a selection rule that must hold in
several places was reimplemented in each of them, and the copies disagreed.

The IATC loop writes two EDN files per proof into one directory:

    <pid>__p<N>.edn         the argument graph        <- an artifact
    <pid>__p<N>.rung2.edn   a rung-2 report           <- a sidecar

Consumers that glob `*.edn` therefore see twice as many "graphs" as exist. The
consequences observed on 2026-08-07 were not crashes but quiet wrongness:
`substance_gate` failed the S3 stage gate on its own sidecars, and
`clean_comprehension` emitted 98 spurious `no-structure` verdict rows that the
capability proof then had to select around by hand. `clean_box_typing` had the
right rule inline; nothing shared it.

Use `proof_graphs()` wherever proof graphs are read. Where a caller genuinely
wants the sidecars (rung-2 analysis), ask for them explicitly.
"""
from __future__ import annotations

import glob
import os

SIDECAR_SUFFIXES = (".rung2.edn",)
ATTEMPT_DIR = ".attempts"


def is_sidecar(path: str) -> bool:
    """True for report/attempt files that live beside proof graphs."""
    base = os.path.basename(path)
    if any(base.endswith(sfx) for sfx in SIDECAR_SUFFIXES):
        return True
    return ATTEMPT_DIR in os.path.normpath(path).split(os.sep)


def proof_graphs(directory: str) -> list[str]:
    """Final proof graphs in `directory`, sidecars and attempts excluded."""
    return sorted(p for p in glob.glob(os.path.join(directory, "*.edn"))
                  if not is_sidecar(p))


def rung2_reports(directory: str) -> list[str]:
    """The sidecars, for callers that actually want them."""
    return sorted(p for p in glob.glob(os.path.join(directory, "*.rung2.edn")))
