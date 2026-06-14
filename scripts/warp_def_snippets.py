#!/usr/bin/env python3
"""Definition-snippet capture (Joe's plan, step 3 prerequisite + GPU corpus).

One corpus scan capturing, per concept, the DEFINITION CONTEXT WINDOW around each
hit (not just the paper-id). Two consumers from one artifact:
  - #3 definition-dependency graph: scan a concept's snippet for OTHER concepts.
  - GPU stage: the snippets are the weak-labeled paraphrase training corpus.

Canon-keyed (dash/case/space unified) but the raw surface form per hit is kept.
Capped per concept to bound size.

    warp_def_snippets.py [--cap N] -> data/warp/def-snippets.json
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warp_defined_pass as dp   # reuse EMPH/DEFENV/CALL/concept_norm/read_text

EPRINTS = dp.EPRINTS
OUT = Path("/home/joe/code/futon6/data/warp/def-snippets.json")
DASH = re.compile(r"[‐-―−-]")


def canon(t):
    t = DASH.sub(" ", t.lower())
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def window(text, a, b, pad=180):
    s = max(0, a - pad)
    e = min(len(text), b + pad)
    return re.sub(r"\s+", " ", text[s:e]).strip()


def hits_with_snippets(text):
    """(canon_concept, surface, snippet) for each definition hit."""
    out = []
    for m in dp.EMPH.finditer(text):
        raw = m.group(1) or m.group(2) or ""
        c = dp.concept_norm(raw)
        if c:
            out.append((canon(raw), raw.strip(), window(text, m.start(), m.end())))
    for m in dp.DEFENV.finditer(text):
        for em in dp.EMPH.finditer(m.group(1)):
            raw = em.group(1) or em.group(2) or ""
            c = dp.concept_norm(raw)
            if c:
                gs = m.start(1) + em.start()
                out.append((canon(raw), raw.strip(), window(text, gs, gs + len(raw))))
    for m in dp.CALL.finditer(text):
        c = dp.concept_norm(m.group(1))
        if c:
            out.append((canon(m.group(1)), m.group(1).strip(),
                        window(text, m.start(), m.end())))
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cap = int(argv[argv.index("--cap") + 1]) if "--cap" in argv else 6
    # only capture for hitlist concepts (the set #3 + GPU care about)
    hl = json.load(open(Path("/home/joe/code/futon6/data/warp/hitlist.json")))
    keep = {h["concept"] for h in hl["hitlist"]}
    snips = defaultdict(list)
    ids = sorted(p.name[:-len(".tar.gz")] for p in EPRINTS.glob("*.tar.gz"))
    done = 0
    for pid in ids:
        t = dp.read_text(pid)
        if not t:
            continue
        done += 1
        seen = set()
        for c, surface, snip in hits_with_snippets(t):
            if c in keep and len(snips[c]) < cap and (c, pid) not in seen:
                snips[c].append({"paper": pid, "surface": surface, "snippet": snip})
                seen.add((c, pid))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"schema": "def-snippets-v1", "papers_scanned": done,
                               "concepts_with_snippets": len(snips),
                               "snippets": dict(snips)}))
    print(f"scanned {done} papers; {len(snips)} concepts have definition snippets")
    for probe in ["monoidal category", "operad", "natural transformation"]:
        s = snips.get(probe, [])
        print(f"  {probe!r}: {len(s)} snippets; e.g. {s[0]['snippet'][:120] if s else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
