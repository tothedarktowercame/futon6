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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any

ROOT = Path(__file__).resolve().parent.parent
NLAB_NAME_ROOT = Path("/home/joe/code/nlab-content/pages")
NLAB_WIRING = ROOT / "data" / "nlab-wiring" / "pages.json"
CT_TERM_PRIOR = ROOT / "data" / "ct-term-prior.json"
DEFAULT_INDEX = ROOT / "data" / "background-corpus-index.json"
NNEXUS_DUMP = Path("/home/joe/code/nnexus/archive/snapshot-1-2014.sqlite")


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
    return {"definition-site": 4, "nnexus": 3, "page": 2, "ct-prior": 1}.get(kind, 0)


def _domain_count(entry: dict[str, Any]) -> int:
    return int(entry.get("domain-count") or len(entry.get("domains", [])) or 0)


def add_resolution(index: dict[str, Any], term: str, kind: str, target: str, display: str | None = None) -> None:
    if not term:
        return
    entry = {"term": display or term, "resolution-kind": kind, "target": target}
    for variant in term_variants(term):
        cur = index.setdefault(variant, [])
        if entry not in cur:
            cur.append(entry)
            cur.sort(key=lambda e: (-_strength(e["resolution-kind"]), -_domain_count(e), e["target"]))


def parse_sql_values(payload: str) -> list[Any]:
    """Parse the comma-separated payload inside a simple SQL VALUES(...).

    The NNexus dump uses SQLite-style single-quoted strings with doubled
    apostrophes. We avoid csv because the rows contain SQL quoting, not CSV.
    """
    values: list[Any] = []
    buf: list[str] = []
    in_quote = False
    quoted = False
    i = 0
    while i < len(payload):
        ch = payload[i]
        if in_quote:
            if ch == "'":
                if i + 1 < len(payload) and payload[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_quote = False
            else:
                buf.append(ch)
        else:
            if ch == "'":
                in_quote = True
                quoted = True
            elif ch == ",":
                raw = "".join(buf).strip()
                values.append(raw if quoted else int(raw) if raw else None)
                buf = []
                quoted = False
            else:
                buf.append(ch)
        i += 1
    if in_quote:
        raise ValueError("unterminated SQL string")
    raw = "".join(buf).strip()
    values.append(raw if quoted else int(raw) if raw else None)
    return values


def _concept_insert_payload(line: str) -> str | None:
    prefix = 'INSERT INTO "concepts" VALUES('
    if not line.startswith(prefix):
        return None
    line = line.rstrip()
    if not line.endswith(");"):
        return None
    return line[len(prefix):-2]


def add_nnexus_concepts(index: dict[str, Any], dump_path: Path = NNEXUS_DUMP) -> dict[str, Any]:
    if not dump_path.exists():
        return {"row-count": 0, "domain-counts": {}, "dropped-count": 0}
    aggregates: dict[str, dict[str, Any]] = {}
    domain_counts: Counter[str] = Counter()
    dropped = 0
    with dump_path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            payload = _concept_insert_payload(line)
            if payload is None:
                continue
            try:
                row = parse_sql_values(payload)
            except (TypeError, ValueError):
                dropped += 1
                continue
            if len(row) != 8:
                dropped += 1
                continue
            _, first, rest, msc, _scheme, domain, url, _objectid = row
            term = " ".join(str(part).strip() for part in (first, rest) if str(part).strip())
            norm = normalize_term(term)
            if not norm:
                dropped += 1
                continue
            domain = str(domain)
            domain_counts[domain] += 1
            agg = aggregates.setdefault(norm, {
                "term": term,
                "resolution-kind": "nnexus",
                "target": f"nnexus:{norm}",
                "domains": set(),
                "msc": set(),
                "urls": set(),
            })
            agg["domains"].add(domain)
            if msc:
                agg["msc"].add(str(msc))
            if url:
                agg["urls"].add(str(url))
    for norm, agg in aggregates.items():
        entry = {
            "term": agg["term"],
            "resolution-kind": "nnexus",
            "target": agg["target"],
            "domains": sorted(agg["domains"]),
            "domain-count": len(agg["domains"]),
            "msc": sorted(agg["msc"]),
            "urls": sorted(agg["urls"]),
        }
        for variant in term_variants(norm):
            cur = index.setdefault(variant, [])
            if entry not in cur:
                cur.append(entry)
                cur.sort(key=lambda e: (-_strength(e["resolution-kind"]), -_domain_count(e), e["target"]))
    return {
        "row-count": sum(domain_counts.values()),
        "domain-counts": dict(sorted(domain_counts.items())),
        "dropped-count": dropped,
    }


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
                prior_path: Path = CT_TERM_PRIOR,
                nnexus_path: Path = NNEXUS_DUMP) -> dict[str, Any]:
    terms: dict[str, Any] = {}
    nlab_count = add_nlab_names(terms, name_root, wiring_path)
    nnexus = add_nnexus_concepts(terms, nnexus_path)
    ct_count = add_ct_prior_terms(terms, candidate_terms, prior_path)
    doc = {
        "schema-version": 2,
        "generated-at": datetime.now(timezone.utc).isoformat(),
        "nlab-name-count": nlab_count,
        "nnexus-row-count": nnexus["row-count"],
        "nnexus-domain-counts": nnexus["domain-counts"],
        "nnexus-dropped-count": nnexus["dropped-count"],
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
    print(json.dumps({
        k: doc[k]
        for k in [
            "nlab-name-count", "nnexus-row-count", "nnexus-domain-counts",
            "nnexus-dropped-count", "ct-prior-count", "candidate-filtered?",
        ]
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
