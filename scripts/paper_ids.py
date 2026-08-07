#!/usr/bin/env python3
"""Shared arXiv paper-id parsing (E-superpod-hardening H14/H19b, 2026-08-06).

arXiv has two id families and the pipeline's safe-form encodes one of them with
a doubled underscore:

    new style   2311.05789          ->  "2311.05789"
    old style   math/0608040        ->  "math__0608040"   (`/` is not path-safe)

Every ad-hoc `split("_")[0]` or `split("__")[0]` therefore breaks on exactly one
family, and the two failures look nothing alike:

  * `iatc_lexicon_harvest._pid_of` split on `__` and took field 0, so EVERY
    pre-2007 paper became the bare archive name "math"; the missing-eprint
    lookup then aborted the whole S10 stage (H14).
  * `expository_reground.harvest_cues` split on `_` and took field 0 — same
    collapse to "math", same abort, different script (H19b).
  * `mark3_expository_loop`'s region-cap trim grouped by filename prefix and so
    missed NEW-style ids, leaving one paper at 336 uncapped regions (H11b).

math.CT is roughly half old-style ids, so none of these is an edge case. This
module is the single place that knows the encoding.
"""
from __future__ import annotations

import os
import re

# old style first: `math__0608040`, `cond-mat__9901001`, `alg-geom__9601001`
_OLD = re.compile(r"^([a-z][a-z-]*(?:\.[A-Za-z]{2})?__\d{7})")
_NEW = re.compile(r"^(\d{4}\.\d{4,5})")


def paper_id_from_name(name: str) -> str | None:
    """Leading arXiv paper id in a filename, or None if it does not start with one.

    Works for any of the pipeline's naming conventions, because it anchors on the
    id itself rather than on whatever separator a given stage happened to use:

        math__0608040__p0.edn                         -> math__0608040
        0705.0102__p12.edn                            -> 0705.0102
        math__0310337_math__0310337-inflight-0001_L1-2.edn -> math__0310337
        0708.1921_0708.1921-leaf-0001_L231-245.edn    -> 0708.1921
    """
    base = os.path.basename(name)
    for rx in (_OLD, _NEW):
        m = rx.match(base)
        if m:
            return m.group(1)
    return None


def proof_pid_from_graph_name(name: str) -> str:
    """Paper id from an IATC graph filename `<pid>__p<N>.edn` (strips the suffix)."""
    base = os.path.basename(name)
    if base.endswith(".edn"):
        base = base[:-4]
    return base.rsplit("__p", 1)[0]
