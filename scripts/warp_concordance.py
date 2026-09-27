#!/usr/bin/env python3
"""Build the WARP cross-paper concordance.

The concordance maps a normalized term to paper-local counts split by role:
``defined`` for DP definiendum / let-binder concept subjects, and ``used`` for
all other DP or sweep-classified appearances.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import futon6_config as config


import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

import anatomy_v0_sweep as sweep
import build_term_prior as prior
import os
import warp_defined_pass as defined_pass

# june 2026-09-16: hardcoded /home/joe/... paths rewritten to a derived code root
# (the tree that holds futon6 and its siblings). FUTON_CODE_ROOT overrides.
_CODE_ROOT = Path(os.environ.get("FUTON_CODE_ROOT") or Path(__file__).resolve().parents[2])


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPRINTS = config.eprint_roots()
DEFAULT_ANATOMY = config.anatomy()
DEFAULT_DP = config.marks()
DEFAULT_OUT = config.warp() / "concordance.json"
DEFAULT_DEFINED_INDEX = config.warp() / "defined-index.json"

ROLE_DEFINED = "defined"
ROLE_USED = "used"
# The prose layer's used phrases travel from a paper's sweep to `build` under this key;
# `build` folds them into the per-concept counts and never writes them as rows.
PROSE_USED_KEY = "_prose-used-phrases"
# canonical concept -> number of papers using it. Written only when prose comes from
# e-prints; sorts before "terms", which the streaming writer requires to be last.
CONCEPT_USED_PAPERS = "concept_used_papers"
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


def prose_counts(text: str, vocabulary: set[str] | None) -> tuple[Counter[tuple[str, str]], set[str], int]:
    """The prose layer for a paper with no DP marks, from the e-print's own TeX.

    DP marks supply two things the control-sequence sweep cannot: what the paper
    DEFINES, and which concept PHRASES it uses. Both are recoverable from the raw
    source with extractors the spine already owns — the defined-pass's emphasized
    definienda (S2) and the term prior's content-bounded n-grams (S6t) — so an
    unmarked subject still reaches the hit-list, which needs a term to be both
    used and defined.

    Returns the DEFINED rows, and the SET of vocabulary phrases the paper uses. The used
    phrases are not rows: across a corpus that is ~600 per paper, tens of millions of
    rows, and the hit-list only ever counts them -- so `build` counts them per concept.

    The used side is restricted to `vocabulary`, the concepts the corpus defines
    SOMEWHERE. Unrestricted, every n-gram in every paper would be considered, and the
    hit-list discards any term that no paper defines.
    """
    counts: Counter[tuple[str, str]] = Counter()
    body = defined_pass.body_before_references(text)
    defined = defined_pass.mined_concepts(text, exclude_references=True)
    for concept in defined:
        counts[(concept, ROLE_DEFINED)] += 1
    used: set[str] = set()
    if vocabulary:
        words = prior._WORD.findall(body.lower())
        used = {gram for gram in prior.ngrams(words) if gram in vocabulary}
    return counts, used, len(defined)


def raw_sweep_counts(eprint_path: Path, roles: dict, plain: set[str], *,
                     prose_vocabulary: set[str] | None = None,
                     with_prose: bool = False) -> tuple[Counter[tuple[str, str]], dict]:
    files, meta = sweep.read_eprint_files(eprint_path)
    counts: Counter[tuple[str, str]] = Counter()
    if not files:
        return counts, {"source": "raw-sweep", "status": "no-files", "loader": meta}
    macros = sweep.collect_macros(files, roles)
    controls = spans = 0
    stripped = []
    for f in files:
        text = sweep.strip_comments(f["text"])
        stripped.append(text)
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
    record = {"source": "raw-sweep", "spans": spans, "classified-controls": controls,
              "loader": meta}
    if with_prose:
        prose, used, defined = prose_counts("\n".join(stripped), prose_vocabulary)
        counts.update(prose)
        record.update({"source": "raw-sweep+prose", "prose-defined": defined,
                       "prose-used": len(used), PROSE_USED_KEY: used})
    return counts, record


def read_eprint_text(eprints: Sequence[Path], paper_id: str) -> str:
    """The paper's TeX, through the reader the defined-pass and usage stages use."""
    original = defined_pass.EPRINT_ROOTS, defined_pass.EPRINTS
    try:
        defined_pass.EPRINT_ROOTS = tuple(eprints)
        defined_pass.EPRINTS = defined_pass.EPRINT_ROOTS[0]
        return defined_pass.read_text(paper_id) or ""
    finally:
        defined_pass.EPRINT_ROOTS, defined_pass.EPRINTS = original


def prose_vocabulary(args: argparse.Namespace, papers: list[str]) -> set[str]:
    """The concepts this corpus defines somewhere, which bound the prose-used side.

    S2's defined-index is that set, so it is read when it exists. Without it the
    same extractor runs here, at the cost of a second pass over the archives —
    correct either way, rather than silently producing an empty prose layer.
    """
    if args.defined_index and args.defined_index.exists():
        index = json.loads(args.defined_index.read_text(encoding="utf-8"))
        return set(index.get("concept_to_papers", index))
    vocabulary: set[str] = set()
    for idx, paper_id in enumerate(papers, 1):
        text = read_eprint_text(args.eprints, paper_id)
        if text:
            vocabulary |= defined_pass.mined_concepts(text, exclude_references=True)
        if args.progress and idx % args.progress == 0:
            print(f"[warp-concordance] prose vocabulary {idx}/{len(papers)} "
                  f"concepts={len(vocabulary)}", file=sys.stderr, flush=True)
    return vocabulary


def dp_paper_id(path: Path) -> str:
    name = path.name
    if name.startswith("fable-") and name.endswith("-dp-emacs.json"):
        return name[len("fable-") : -len("-dp-emacs.json")]
    return path.stem


def iter_requested_papers(eprints: Sequence[Path], dp_dir: Path, limit: int | None) -> list[str]:
    ids = {dp_paper_id(p) for p in dp_dir.glob("fable-*-dp-emacs.json")}
    eprint_ids = [sweep.strip_archive_suffix(p) for p in sweep.iter_corpus_eprints(eprints)]
    for paper_id in eprint_ids[:limit] if limit is not None else eprint_ids:
        ids.add(paper_id)
    return sorted(ids)


def find_eprint(eprints: Sequence[Path], paper_id: str) -> Path | None:
    return sweep.find_corpus_eprint(paper_id, eprints)


def build(args: argparse.Namespace) -> dict:
    start = time.time()
    roles = sweep.load_latexml_roles(sweep.ROLE_TSV)
    plain = sweep.load_plain_cseq(sweep.PLAIN_CSEQ)
    papers = iter_requested_papers(args.eprints, args.dp_dir, args.limit)
    # One row per (term, paper, role), held as a (paper, count, role) TUPLE rather than a
    # dict: across a corpus that is tens of millions of rows, and a 3-key dict costs about
    # three times a 3-tuple. `write_concordance` turns them into the documented
    # {"count", "paper", "role"} objects as it writes, so the file is unchanged.
    index: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    stats = Counter()
    source_counts = Counter()
    failures: list[dict] = []
    with_prose = args.prose_source == "eprints"
    vocabulary = prose_vocabulary(args, papers) if with_prose else None
    # With prose from e-prints, usage is COUNTED here rather than written as rows: for each
    # canonical concept, the number of distinct papers using any of its surface forms --
    # exactly `len(used-set)` as the hit-list computes it from rows, without the rows.
    concept_used: Counter[str] | None = Counter() if with_prose else None
    if with_prose:
        import warp_hitlist
        canon = warp_hitlist.canon

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
                    counts, meta = raw_sweep_counts(
                        eprint_path, roles, plain,
                        prose_vocabulary=vocabulary, with_prose=with_prose)
        except Exception as exc:
            failures.append({"paper": paper_id, "error": repr(exc)})
            stats["failed"] += 1
            continue

        if concept_used is not None:
            used_here = set(meta.pop(PROSE_USED_KEY, ()))
            used_here.update(term for term, role in counts if role == ROLE_USED)
            concept_used.update({canon(term) for term in used_here} - {""})
        source = meta.get("source", "unknown")
        source_counts[source] += 1
        stats["papers"] += 1
        if counts:
            stats["papers-with-terms"] += 1
        else:
            stats["papers-without-terms"] += 1
        stats["term-role-observations"] += sum(counts.values())
        for (term, role), count in sorted(counts.items()):
            index[term].append((paper_id, count, role))
        if args.progress and (idx % args.progress == 0 or idx == len(papers)):
            print(
                f"[warp-concordance] {idx}/{len(papers)} source={dict(source_counts)} terms={len(index)}",
                file=sys.stderr,
                flush=True,
            )

    # Sorted in place, not copied: a sorted copy of the whole index doubled peak memory.
    for rows in index.values():
        rows.sort(key=lambda row: (row[0], row[2]))
    stats_out = dict(stats)
    stats_out.update(
        {
            "requested-papers": len(papers),
            "unique-terms": len(index),
            "eprint-roots": [Path(root).as_posix() for root in args.eprints],
            "prose-source": args.prose_source,
            "prose-vocabulary": len(vocabulary) if vocabulary is not None else 0,
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
        "terms": index,
        **({CONCEPT_USED_PAPERS: dict(sorted(concept_used.items()))} if concept_used is not None else {}),
    }


def write_concordance(result: dict, handle) -> None:
    """Write `result` exactly as `json.dumps(result, indent=2, sort_keys=True) + "\\n"`
    would, but one term at a time.

    Serializing the whole concordance to one string first held a second, larger copy of it
    in memory at the very end of the run. "terms" sorts last among the top-level keys, so
    everything before it is written by `json` itself and the terms follow, each row turned
    into its {"count", "paper", "role"} object only as it is written.
    """
    head = {key: value for key, value in result.items() if key != "terms"}
    head["terms"] = {}
    opening = json.dumps(head, indent=2, sort_keys=True)
    tail = '"terms": {}\n}'
    if not opening.endswith(tail):
        raise ValueError("concordance keys changed: 'terms' no longer sorts last")
    terms = result["terms"]
    if not terms:
        handle.write(opening + "\n")
        return
    handle.write(opening[: -len("{}\n}")] + "{")
    for position, term in enumerate(sorted(terms)):
        rows = [{"count": count, "paper": paper, "role": role} for paper, count, role in terms[term]]
        body = json.dumps(rows, indent=2, sort_keys=True).replace("\n", "\n    ")
        handle.write(("," if position else "") + "\n    " + json.dumps(term) + ": " + body)
    handle.write("\n  }\n}\n")


def iter_concordance_terms(path: Path, *, chunk: int = 1 << 20, meta: dict | None = None):
    """Yield (term, rows) from a concordance file one term at a time; every OTHER top-level
    value is parsed into `meta` when given (they all precede "terms").

    `json.load` on a corpus-scale concordance costs about four times the file size in
    memory -- tens of GB for a subject mined from raw e-prints -- although a reader needs
    only one term's rows at a time. This is an ordinary JSON parse, done incrementally
    with the standard decoder: it depends on JSON syntax, never on how the file was
    indented, so any valid concordance reads the same as through `json.load`.
    """
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, at = "", 0

        def fill() -> bool:
            nonlocal buffer, at
            more = handle.read(chunk)
            if not more:
                return False
            buffer, at = buffer[at:] + more, 0
            return True

        def skip(characters: str = " \t\r\n") -> None:
            nonlocal at
            while True:
                while at < len(buffer) and buffer[at] in characters:
                    at += 1
                if at < len(buffer) or not fill():
                    return

        def expect(token: str) -> None:
            nonlocal at
            skip()
            if at >= len(buffer) or buffer[at] != token:
                raise ValueError(f"concordance: expected {token!r} at offset {at}")
            at += 1

        def value():
            nonlocal at
            skip()
            while True:
                try:
                    parsed, end = decoder.raw_decode(buffer, at)
                except json.JSONDecodeError:
                    if not fill():
                        raise
                    continue
                # A number can end exactly at the chunk boundary and still decode.
                if end == len(buffer) and fill():
                    continue
                at = end
                return parsed

        fill()
        expect("{")
        while True:
            skip()
            if buffer[at] == "}":
                return
            key = value()
            expect(":")
            if key != "terms":
                parsed = value()
                if meta is not None:
                    meta[key] = parsed
            else:
                expect("{")
                skip()
                if buffer[at] == "}":
                    at += 1
                else:
                    while True:
                        term = value()
                        expect(":")
                        yield term, value()
                        skip()
                        if buffer[at] == ",":
                            at += 1
                            continue
                        expect("}")
                        break
            skip()
            if buffer[at] == ",":
                at += 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eprints", type=Path, action="append", default=None,
                    help="An e-print directory; repeatable. Defaults to the configured roots "
                         "(FUTON6_EPRINTS, which may name a file listing them).")
    ap.add_argument("--anatomy-dir", type=Path, default=DEFAULT_ANATOMY)
    ap.add_argument("--dp-dir", type=Path, default=DEFAULT_DP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prose-source", choices=config.PROSE_SOURCES, default=config.prose_source(),
                    help="Where an unmarked paper's prose terms come from. `marks` (the "
                         "default) indexes only classified control sequences for such a "
                         "paper, as this spine always has.")
    ap.add_argument("--defined-index", type=Path, default=DEFAULT_DEFINED_INDEX,
                    help="S2's defined-index, which bounds the prose-used vocabulary; "
                         "recomputed in an extra pass when absent.")
    ap.add_argument("--limit", type=int, default=None, help="Limit the eprint batch; DP papers are always included.")
    ap.add_argument("--progress", type=int, default=500, help="Log every N papers; 0 disables progress.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.eprints = tuple(args.eprints) if args.eprints else DEFAULT_EPRINTS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = build(args)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        write_concordance(result, handle)
    tmp.replace(args.out)
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    return 0 if not result["stats"].get("failures") else 1


if __name__ == "__main__":
    raise SystemExit(main())
