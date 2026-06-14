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

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anatomy_v0_sweep as sweep

EPRINTS = sweep.DEFAULT_EPRINTS
OUT = Path("/home/joe/code/futon6/data/warp/defined-index.json")

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


def read_text(paper_id: str):
    for suf in (".tar.gz", ".gz", ".tar", ".tex"):
        p = EPRINTS / f"{paper_id}{suf}"
        if p.exists():
            try:
                files, _ = sweep.read_eprint_files(p)
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
    return None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    if "--probe" in argv:
        ids = argv[argv.index("--probe") + 1].split(",")
    else:
        ids = sorted(p.name[:-len(".tar.gz")] for p in EPRINTS.glob("*.tar.gz"))
        if limit:
            ids = ids[:limit]
    concept2papers: dict[str, list] = {}
    done = skip = 0
    for pid in ids:
        t = read_text(pid)
        if not t:
            skip += 1
            continue
        for c in defined_concepts(t):
            concept2papers.setdefault(c, []).append(pid)
        done += 1
    idx = {c: sorted(set(ps)) for c, ps in concept2papers.items()}
    if "--probe" not in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "schema": "defined-index-v1", "papers_scanned": done, "skipped": skip,
            "unique_concepts": len(idx), "concept_to_papers": idx}))
    print(f"scanned {done} papers ({skip} skipped); {len(idx)} unique defined-concepts")
    for probe in ["homotopy colimit", "hopf algebra", "comodule",
                  "monoidal category", "operad", "galois object"]:
        print(f"  {probe!r}: defined in {len(idx.get(probe, []))} papers "
              f"{idx.get(probe, [])[:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
