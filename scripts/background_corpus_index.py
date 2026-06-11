#!/usr/bin/env python3
"""Build a compact background-corpus lookup index for proof scope audits.

The full background corpus is large (nLab content + 432MB wiring JSON + a
large CT term prior). This module builds a small persisted lookup table keyed
by normalized terms. nLab page names are indexed fully; CT-prior terms are
materialized for requested candidate terms so audit-time lookup stays fast.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any

ROOT = Path(__file__).resolve().parent.parent
NLAB_NAME_ROOT = Path("/home/joe/code/nlab-content/pages")
NLAB_WIRING = ROOT / "data" / "nlab-wiring" / "pages.json"
CT_TERM_PRIOR = ROOT / "data" / "ct-term-prior.json"
DEFAULT_INDEX = ROOT / "data" / "background-corpus-index.json"


def normalize_term(term: str) -> str:
    term = re.sub(r"[`*_{}()\[\],.;:]+", " ", str(term))
    term = re.sub(r"\s+", " ", term).strip().lower()
    return term


def term_variants(term: str) -> set[str]:
    norm = normalize_term(term)
    if not norm:
        return set()
    variants = {norm}
    if norm.endswith("s") and len(norm) > 3:
        variants.add(norm[:-1])
    elif len(norm) > 2:
        variants.add(norm + "s")
    return variants


def _strength(kind: str) -> int:
    return {"definition-site": 3, "page": 2, "ct-prior": 1}.get(kind, 0)


def add_resolution(index: dict[str, Any], term: str, kind: str, target: str, display: str | None = None) -> None:
    if not term:
        return
    entry = {"term": display or term, "resolution-kind": kind, "target": target}
    for variant in term_variants(term):
        cur = index.setdefault(variant, [])
        if entry not in cur:
            cur.append(entry)
            cur.sort(key=lambda e: (-_strength(e["resolution-kind"]), e["target"]))


def definition_site_ids(wiring_path: Path = NLAB_WIRING) -> set[str]:
    if not wiring_path.exists():
        return set()
    data = json.loads(wiring_path.read_text(encoding="utf-8"))
    out = set()
    for row in data:
        env_types = row.get("stats", {}).get("env_types", {})
        if env_types.get("env/definition", 0) > 0:
            out.add(str(row.get("page_id", "")).replace("nlab-", ""))
    return out


def add_nlab_names(index: dict[str, Any], name_root: Path = NLAB_NAME_ROOT, wiring_path: Path = NLAB_WIRING) -> int:
    def_sites = definition_site_ids(wiring_path)
    count = 0
    for name_file in name_root.glob("*/*/*/*/*/name"):
        page_num = name_file.parent.name
        try:
            name = name_file.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not name:
            continue
        kind = "definition-site" if page_num in def_sites else "page"
        add_resolution(index, name, kind, f"nlab-{page_num}", name)
        count += 1
    return count


def add_ct_prior_terms(index: dict[str, Any], candidate_terms: Iterable[str] | None = None,
                       prior_path: Path = CT_TERM_PRIOR) -> int:
    if not prior_path.exists():
        return 0
    wanted = None
    if candidate_terms is not None:
        wanted = set()
        for term in candidate_terms:
            wanted.update(term_variants(term))
    data = json.loads(prior_path.read_text(encoding="utf-8"))
    count = 0
    for bucket in ("unigram_df", "bigram_df"):
        for term, df in data.get(bucket, {}).items():
            variants = term_variants(term)
            if not variants:
                continue
            if wanted is not None and variants.isdisjoint(wanted):
                continue
            add_resolution(index, term, "ct-prior", f"ct-term-prior:{normalize_term(term)}", term)
            count += 1
    return count


def build_index(candidate_terms: Iterable[str] | None = None,
                output: Path = DEFAULT_INDEX,
                name_root: Path = NLAB_NAME_ROOT,
                wiring_path: Path = NLAB_WIRING,
                prior_path: Path = CT_TERM_PRIOR) -> dict[str, Any]:
    terms: dict[str, Any] = {}
    nlab_count = add_nlab_names(terms, name_root, wiring_path)
    ct_count = add_ct_prior_terms(terms, candidate_terms, prior_path)
    doc = {
        "generated-at": datetime.now(timezone.utc).isoformat(),
        "nlab-name-count": nlab_count,
        "ct-prior-count": ct_count,
        "candidate-filtered?": candidate_terms is not None,
        "candidate-terms": sorted({normalize_term(t) for t in candidate_terms or [] if normalize_term(t)}),
        "terms": terms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def load_index(path: Path = DEFAULT_INDEX) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(index: dict[str, Any], term: str) -> dict[str, Any] | None:
    terms = index.get("terms", index)
    for variant in term_variants(term):
        hits = terms.get(variant)
        if hits:
            hit = hits[0].copy()
            hit["query"] = term
            return hit
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--candidate", action="append", default=[])
    args = ap.parse_args(argv)
    doc = build_index(args.candidate or None, args.output)
    print(json.dumps({k: doc[k] for k in ["nlab-name-count", "ct-prior-count", "candidate-filtered?"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
