#!/usr/bin/env python3
"""Classical corpus-wide DEFINED-pass (Joe's plan, 2026-06-14, step 1).

For every math.CT eprint, extract the concepts it DEFINES — emphasized
definienda (math-paper convention: a term is italicised/bolded on definition),
definition environments, and "is called the X" — building a
concept -> [defining papers] index. Cheap: regex over text, single process,
no full DP markup, no agents, no per-paper heavy reloads.

Each defined concept IS a definition-SCOPE: grounding it in one paper helps
every other paper that USES it (the cross-paper propagation that compounds,
PageRank-style, over the definition-dependency graph).

    warp_defined_pass.py [--limit N] [--probe id1,id2]
        -> data/warp/defined-index.json  {concept: [defining papers]}
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import futon6_config as config


import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anatomy_v0_sweep as sweep
import markup_strategies as markup

EPRINT_ROOTS = config.eprint_roots()
EPRINTS = EPRINT_ROOTS[0]
OUT = config.warp() / 'defined-index.json'

# emphasized defined-term markers: a term italicised/bolded (the math-paper
# convention for "this is the definition"). High-recall, cheap.
EMPH = re.compile(
    r"\\(?:emph|textit|textbf|textsl|defn|define|dfn|term)\s*\{([^{}]{2,60})\}"
    r"|\{\\(?:em|it|bf|sl)\s+([^{}]{2,60})\}")
DEFENV = re.compile(
    r"\\begin\{(?:definition|defn|define|dfn|defi)\*?\}(.*?)"
    r"\\end\{(?:definition|defn|define|dfn|defi)\*?\}", re.S)
CALL = re.compile(
    r"(?:is called|we call(?:\s+it)?|is termed|known as|is defined to be|"
    r"is defined as)\s+(?:an?|the)\s+([A-Za-z][A-Za-z\- ]{2,40})", re.I)


def concept_norm(s: str):
    s = re.sub(r"\$[^$]*\$|\\[A-Za-z]+|[{}]", " ", s)   # strip math/macros
    s = re.sub(r"[^A-Za-z\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    words = s.split()
    if not (1 <= len(words) <= 4):       # concepts are 1-4 words
        return None
    return s if 3 <= len(s) <= 40 else None


def defined_concepts(text: str):
    out = set()
    def add(raw):
        c = concept_norm(raw or "")
        if c:
            out.add(c)
    for m in EMPH.finditer(text):
        add(m.group(1) or m.group(2))
    for m in DEFENV.finditer(text):
        for em in EMPH.finditer(m.group(1)):
            add(em.group(1) or em.group(2))
    for m in CALL.finditer(text):
        add(m.group(1))
    return out


def body_before_references(text: str) -> str:
    """The paper without its reference list, where `markup_strategies` puts the line.

    A bibliography italicises journal and publisher names, so emphasis there is a
    citation, not a definiendum.
    """
    return text[:markup.bibliography_at(text, [], markup.RAW_BIBLIOGRAPHY_MARKERS)]


def mined_concepts(text: str, *, exclude_references: bool) -> set[str]:
    """What a paper defines. Mining an UNMARKED e-print has to drop the reference
    list and the journal names in it; the marked-up path is unchanged, so this
    checkout's math.CT defined-index is exactly what it always was.
    """
    if not exclude_references:
        concepts = defined_concepts(text)
    else:
        concepts = {c for c in defined_concepts(body_before_references(text))
                    if not markup.bibliographic(c)}
    return _unfiltered(concepts)


def _unfiltered(concepts: set[str]) -> set[str]:
    """Drop what the run's concept filter (if any) says is not a concept. Applied here, where a
    concept is first recorded, it never reaches the concordance's vocabulary or the hit-list."""
    import concept_filter

    active = concept_filter.configured()
    if active is None:
        return concepts
    import warp_hitlist

    return {c for c in concepts if active.match(warp_hitlist.canon(c)) is None}


def corpus_paper_ids() -> list[str]:
    """Every paper of the corpus, across all of its e-print directories.

    Version-stripped, because that is the form a run manifest and the S2 substrate gate
    use: an index keyed `0704.0011v3` would match no run id.
    """
    return sweep.corpus_paper_ids(active_roots())


def active_roots() -> tuple[Path, ...]:
    """The corpus directories this module is reading.

    `EPRINTS` remains a single directory a caller may rebind to select another corpus
    (the term prior does exactly that); when it is untouched, the configured root list
    is used in full.
    """
    return EPRINT_ROOTS if EPRINTS == EPRINT_ROOTS[0] else (EPRINTS,)


def read_text(paper_id: str):
    archive = sweep.find_corpus_eprint(paper_id, active_roots())
    if archive is None:
        return None
    try:
        files, _meta = sweep.read_eprint_files(archive)
    except Exception:
        return None
    if isinstance(files, dict):
        return "\n".join(files.values())
    if isinstance(files, list):
        parts = []
        for f in files:
            if isinstance(f, dict):
                parts.append(f.get("text", ""))
            elif isinstance(f, (list, tuple)) and len(f) > 1:
                parts.append(str(f[1]))
            else:
                parts.append(str(f))
        return "\n".join(parts)
    return files if isinstance(files, str) else None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    limit = None
    # The prose source says whether these papers are marked up; an unmarked corpus
    # is mined out of the raw source, references excluded.
    exclude_references = config.prose_source() == "eprints"
    if "--keep-references" in argv:
        exclude_references = False
    if "--exclude-references" in argv:
        exclude_references = True
    # A corpus-scale pass is otherwise a silent hour: say where it is as it goes.
    progress = int(argv[argv.index("--progress") + 1]) if "--progress" in argv else 500
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    if "--probe" in argv:
        ids = argv[argv.index("--probe") + 1].split(",")
    else:
        ids = corpus_paper_ids()
        if limit:
            ids = ids[:limit]
    concept2papers: dict[str, list] = {}
    done = skip = 0
    for index, pid in enumerate(ids, 1):
        t = read_text(pid)
        if not t:
            skip += 1
            continue
        for c in mined_concepts(t, exclude_references=exclude_references):
            concept2papers.setdefault(c, []).append(pid)
        done += 1
        if progress and (index % progress == 0 or index == len(ids)):
            print(f"[warp-defined-pass] {index}/{len(ids)} read={done} skipped={skip} "
                  f"concepts={len(concept2papers)}", file=sys.stderr, flush=True)
    idx = {c: sorted(set(ps)) for c, ps in concept2papers.items()}
    if "--probe" not in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "schema": "defined-index-v1", "papers_scanned": done, "skipped": skip,
            "references_excluded": exclude_references,
            "unique_concepts": len(idx), "concept_to_papers": idx}))
    print(f"scanned {done} papers ({skip} skipped); {len(idx)} unique defined-concepts")
    for probe in ["homotopy colimit", "hopf algebra", "comodule",
                  "monoidal category", "operad", "galois object"]:
        print(f"  {probe!r}: defined in {len(idx.get(probe, []))} papers "
              f"{idx.get(probe, [])[:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
