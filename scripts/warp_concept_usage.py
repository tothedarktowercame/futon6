#!/usr/bin/env python3
"""Corpus-wide concept-USAGE scan (the classical extension for all-paper coverage).

The concordance recorded concept *phrases* only for the ~261 DP-marked papers
(cseq tokens for the rest), so papers couldn't be placed by concept profile.
This scans ALL 9742 eprints for occurrences of the hitlist concept phrases
(1-3-gram match against the canon concept set) -> paper -> [concepts used].
Cheap classical pass (regex tokenize + set lookup), no agents, no superpod —
the complement to the defined-index (definition) that unlocks the all-paper
landscape.

    warp_concept_usage.py -> data/warp/concept-usage.json {paper: [concepts]}
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warp_defined_pass as dp

W = Path("/home/joe/code/futon6/data/warp")
EPRINTS = dp.EPRINTS
DASH = re.compile(r"[‐-―−-]")
STOPW = set("the a an of to in on for and or is are be by with we that this it as at".split())


def canon_toks(text):
    text = DASH.sub(" ", text.lower())
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return [w for w in text.split() if w not in STOPW]


def main():
    concepts = {h["concept"] for h in json.load(open(W / "hitlist.json"))["hitlist"]}
    maxn = max(len(c.split()) for c in concepts)
    paper_use = {}
    ids = sorted(p.name[:-len(".tar.gz")] for p in EPRINTS.glob("*.tar.gz"))
    done = 0
    for pid in ids:
        t = dp.read_text(pid)
        if not t:
            continue
        done += 1
        w = canon_toks(t)
        found = set()
        # n-gram membership against the concept set
        for n in range(1, min(maxn, 3) + 1):
            for i in range(len(w) - n + 1):
                g = " ".join(w[i:i + n])
                if g in concepts:
                    found.add(g)
        if found:
            paper_use[pid] = sorted(found)
    (W / "concept-usage.json").write_text(json.dumps(
        {"schema": "concept-usage-v1", "papers_scanned": done,
         "papers_with_concepts": len(paper_use), "paper_concepts": paper_use}))
    import statistics
    cnts = [len(v) for v in paper_use.values()]
    print(f"scanned {done} papers; {len(paper_use)} use >=1 hitlist concept; "
          f"median concepts/paper {statistics.median(cnts) if cnts else 0:.0f}; "
          f">=2 concepts: {sum(1 for c in cnts if c >= 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
