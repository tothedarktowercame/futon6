#!/usr/bin/env python3
"""Build the WARP cross-paper concordance.

The concordance maps a normalized term to paper-local counts split by role:
``defined`` for DP definiendum / let-binder concept subjects, and ``used`` for
all other DP or sweep-classified appearances.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import anatomy_v0_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
DEFAULT_ANATOMY = Path("/home/joe/code/storage/futon6/data/ct-anatomy-v0")
DEFAULT_DP = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
DEFAULT_OUT = ROOT / "data" / "warp" / "concordance.json"

ROLE_DEFINED = "defined"
ROLE_USED = "used"
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "if", "in",
    "into", "is", "it", "of", "on", "or", "over", "such", "that", "the",
    "then", "to", "with",
}


def field_map(mark: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in mark.get("fields") or []:
        if isinstance(row, list) and len(row) >= 2:
            out[str(row[0])] = str(row[1])
    return out


def strip_canon_id(value: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]\s*$", "", value).strip()


def normalize_term(value: str) -> str | None:
    value = value.strip()
    if not value or value in {"-", "\u2013", "\u2014", "\u2014 (unresolved)"}:
        return None
    value = strip_canon_id(value)
    value = re.sub(r"\$([^$]*)\$", r" \1 ", value)
    value = re.sub(r"\\(?:text|mathrm|mathbf|mathcal|mathbb|mathsf|mathfrak)\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\", " ")
    value = re.sub(r"[{}_^~`\"\u201c\u201d\u2018\u2019]", " ", value)
    value = value.replace("-", " ")
    value = re.sub(r"[^A-Za-z0-9'.]+", " ", value)
    words = [w for w in value.split() if w and w.lower() not in STOPWORDS]
    if not words:
        return None
    term = " ".join(words)
    if len(term) == 1 and not term.isupper():
        return None
    return term


def normalize_control(cs: str) -> str | None:
    cs = cs.strip()
    if not cs:
        return None
    if not cs.startswith("\\"):
        cs = "\\" + cs
    return cs


def term_variants(value: str) -> set[str]:
    """Return stable variants useful for concept lookup and coarse gates."""
    base = normalize_term(value)
    if not base:
        return set()
    variants = {base}
    words = base.split()
    for size in (3, 2, 1):
        if len(words) >= size:
            tail = " ".join(words[-size:])
            if len(tail) > 1:
                variants.add(tail)
    for idx, word in enumerate(words):
        low = word.lower()
        if low in {"hopf", "monoidal", "abelian", "braided", "comodule", "module", "algebra", "coalgebra", "category", "functor"}:
            phrase = " ".join(words[idx : idx + 2])
            if phrase:
                variants.add(phrase)
            variants.add(word)
    return {v for v in variants if v and v.lower() not in STOPWORDS}


def concept_from_tip(tip: str) -> str | None:
    m = re.search(r"concept:\s*([^\u00b7]+)", tip)
    return strip_canon_id(m.group(1).strip()) if m else None


def classified_cs_from_tip(tip: str) -> str | None:
    if "\u00b7" in tip:
        head = tip.split("\u00b7", 1)[0].strip()
        if head.startswith("\\"):
            return head
    return None


def mark_text(mark: dict, text: str) -> str:
    try:
        start, end = int(mark["start"]), int(mark["end"])
    except Exception:
        return ""
    if start < 0 or end < start or end > len(text):
        return ""
    return text[start:end]


def add_terms(counts: Counter[tuple[str, str]], terms: Iterable[str], role: str) -> None:
    for term in terms:
        if term:
            counts[(term, role)] += 1


def dp_counts(dp_path: Path) -> tuple[Counter[tuple[str, str]], dict]:
    row = json.loads(dp_path.read_text(encoding="utf-8"))
    text = row.get("text") or ""
    counts: Counter[tuple[str, str]] = Counter()
    kinds = Counter()
    for mark in row.get("marks") or []:
        kind = mark.get("kind")
        kinds[kind] += 1
        fields = field_map(mark)
        if kind == "let-binder":
            add_terms(counts, term_variants(fields.get("as", "")), ROLE_DEFINED)
            canon = fields.get("canon")
            if canon and not canon.startswith("\u2014"):
                add_terms(counts, term_variants(canon), ROLE_DEFINED)
        elif kind == "definiendum":
            term = normalize_term(mark_text(mark, text))
            if term:
                counts[(term, ROLE_DEFINED)] += 1
        elif kind == "definiens":
            add_terms(counts, term_variants(mark_text(mark, text)), ROLE_USED)
        elif kind == "concept-typed":
            concept = concept_from_tip(str(mark.get("tip", "")))
            if concept:
                add_terms(counts, term_variants(concept), ROLE_USED)
            else:
                cs = classified_cs_from_tip(str(mark.get("tip", "")))
                term = normalize_control(cs or mark_text(mark, text))
                if term:
                    counts[(term, ROLE_USED)] += 1
        elif kind == "symbol-grounded":
            bound = fields.get("bound")
            if bound:
                add_terms(counts, term_variants(bound), ROLE_USED)
        elif kind == "classified":
            cs = classified_cs_from_tip(str(mark.get("tip", "")))
            term = normalize_control(cs or mark_text(mark, text))
            if term:
                counts[(term, ROLE_USED)] += 1
    return counts, {"source": "dp", "marks": sum(kinds.values()), "mark-kinds": dict(kinds)}


def anatomy_counts(path: Path) -> tuple[Counter[tuple[str, str]], dict]:
    row = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[tuple[str, str]] = Counter()
    controls = seen_spans = 0
    for span in row.get("token-census", {}).get("spans") or []:
        seen_spans += 1
        for ctrl in span.get("controls") or []:
            if ctrl.get("class") == "UNKNOWN":
                continue
            term = normalize_control(ctrl.get("cs", ""))
            if term:
                counts[(term, ROLE_USED)] += 1
                controls += 1
    return counts, {"source": "anatomy-json", "spans": seen_spans, "classified-controls": controls}


def raw_sweep_counts(eprint_path: Path, roles: dict, plain: set[str]) -> tuple[Counter[tuple[str, str]], dict]:
    files, meta = sweep.read_eprint_files(eprint_path)
    counts: Counter[tuple[str, str]] = Counter()
    if not files:
        return counts, {"source": "raw-sweep", "status": "no-files", "loader": meta}
    macros = sweep.collect_macros(files, roles)
    controls = spans = 0
    for f in files:
        text = sweep.strip_comments(f["text"])
        for _start, _end, _delim, body in sweep.math_spans(text):
            spans += 1
            for cs in sweep.control_sequences(body):
                cls = sweep.classify_cseq(cs, macros, roles, plain)
                if cls.get("class") == "UNKNOWN":
                    continue
                term = normalize_control(cls.get("cs", ""))
                if term:
                    counts[(term, ROLE_USED)] += 1
                    controls += 1
    return counts, {"source": "raw-sweep", "spans": spans, "classified-controls": controls, "loader": meta}


def dp_paper_id(path: Path) -> str:
    name = path.name
    if name.startswith("fable-") and name.endswith("-dp-emacs.json"):
        return name[len("fable-") : -len("-dp-emacs.json")]
    return path.stem


def iter_requested_papers(eprints: Path, dp_dir: Path, limit: int | None) -> list[str]:
    ids = {dp_paper_id(p) for p in dp_dir.glob("fable-*-dp-emacs.json")}
    eprint_ids = [sweep.strip_archive_suffix(p) for p in sweep.iter_eprints(eprints)]
    for paper_id in eprint_ids[:limit] if limit is not None else eprint_ids:
        ids.add(paper_id)
    return sorted(ids)


def find_eprint(eprints: Path, paper_id: str) -> Path | None:
    for suffix in (".tar.gz", ".gz", ".tar", ".tex", ".bin"):
        p = eprints / f"{paper_id}{suffix}"
        if p.exists():
            return p
    return None


def build(args: argparse.Namespace) -> dict:
    start = time.time()
    roles = sweep.load_latexml_roles(sweep.ROLE_TSV)
    plain = sweep.load_plain_cseq(sweep.PLAIN_CSEQ)
    papers = iter_requested_papers(args.eprints, args.dp_dir, args.limit)
    index: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()
    source_counts = Counter()
    failures: list[dict] = []

    for idx, paper_id in enumerate(papers, 1):
        dp_path = args.dp_dir / f"fable-{paper_id}-dp-emacs.json"
        anatomy_path = args.anatomy_dir / f"{paper_id}.json"
        try:
            if dp_path.exists():
                counts, meta = dp_counts(dp_path)
            elif anatomy_path.exists():
                counts, meta = anatomy_counts(anatomy_path)
            else:
                eprint_path = find_eprint(args.eprints, paper_id)
                if eprint_path is None:
                    counts, meta = Counter(), {"source": "missing-eprint"}
                else:
                    counts, meta = raw_sweep_counts(eprint_path, roles, plain)
        except Exception as exc:
            failures.append({"paper": paper_id, "error": repr(exc)})
            stats["failed"] += 1
            continue

        source = meta.get("source", "unknown")
        source_counts[source] += 1
        stats["papers"] += 1
        if counts:
            stats["papers-with-terms"] += 1
        else:
            stats["papers-without-terms"] += 1
        stats["term-role-observations"] += sum(counts.values())
        for (term, role), count in sorted(counts.items()):
            index[term].append({"paper": paper_id, "count": count, "role": role})
        if args.progress and (idx % args.progress == 0 or idx == len(papers)):
            print(
                f"[warp-concordance] {idx}/{len(papers)} source={dict(source_counts)} terms={len(index)}",
                file=sys.stderr,
                flush=True,
            )

    concordance = {term: sorted(rows, key=lambda r: (r["paper"], r["role"])) for term, rows in sorted(index.items())}
    stats_out = dict(stats)
    stats_out.update(
        {
            "requested-papers": len(papers),
            "unique-terms": len(concordance),
            "sources": dict(source_counts),
            "failures": len(failures),
            "elapsed-sec": round(time.time() - start, 3),
        }
    )
    return {
        "schema": "warp-concordance-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stats": stats_out,
        "failures": failures[:100],
        "terms": concordance,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eprints", type=Path, default=DEFAULT_EPRINTS)
    ap.add_argument("--anatomy-dir", type=Path, default=DEFAULT_ANATOMY)
    ap.add_argument("--dp-dir", type=Path, default=DEFAULT_DP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="Limit the eprint batch; DP papers are always included.")
    ap.add_argument("--progress", type=int, default=500, help="Log every N papers; 0 disables progress.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = build(args)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    return 0 if not result["stats"].get("failures") else 1


if __name__ == "__main__":
    raise SystemExit(main())
