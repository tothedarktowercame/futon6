#!/usr/bin/env python3
"""E-iatc-expository-alignment, Pass A — cue-scan the existing expository-scope vote
for IATC perf/value/meta phrasings.

Reads the agent proposal JSONL files (the close-reading vote over the gh200 papers),
and for each IATC performative / value / meta category, counts how many proposal QUOTES
(real arXiv expository sentences the agents flagged) contain a cue phrase for that
category. This is a CPU, lexical FIRST PASS — crude and tunable, not ground truth — that
measures which IATC categories already appear in published expository prose vs which are
(near-)absent, against the pre-registered crosswalk prediction.

struct[...] and rel[...] are NOT scanned: that content/inferential side is already owned
by the formal scopes (Joe, 2026-06-17), so the open question is only perf/value/meta.

Usage:
  python scripts/iatc_alignment_passA.py [--proposals DIR] [--examples N]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

DEFAULT_PROPOSALS = "/home/joe/code/futon3c/holes/excursions/close-reading/proposals"

# (family, category, prediction, [cue regexes]) — prediction is the pre-registered call.
# Cues are deliberately conservative; lowercased-quote matching.
CUES = [
    # --- performatives (dialogical) ---
    ("perf", "Agree", "diverge", [r"\bwe agree\b", r"\bin agreement with\b",
        r"\bas .{0,25}? (notes|observes|remarks|points out)\b", r"\bwe concur\b"]),
    ("perf", "Challenge", "diverge", [r"\bthis is false\b", r"\bwe disagree\b",
        r"\bon the contrary\b", r"\bcontrary to\b", r"\bis incorrect\b", r"\bis simply not true\b"]),
    # NB: bare "retract" excluded — it is the math term (section-retraction); require the speech act.
    ("perf", "Retract", "diverge", [r"\bwe (were wrong|retract|now correct)\b", r"\bon second thought\b",
        r"\bas it turns out,? .{0,25}?(wrong|incorrect|mistaken)\b", r"\bcorrection:\b"]),
    ("perf", "Suggest", "slot-in", [r"\bwe suggest\b", r"\bthe trick\b", r"\bone approach\b",
        r"\bthe idea is to\b", r"\bit (is|might be) (natural|tempting|helpful|convenient) to\b",
        r"\bwe propose to\b", r"\ba natural strategy\b", r"\bone (is|may be) tempted\b"]),
    ("perf", "Query", "duplicate", [r"\bis it true that\b", r"\bwe ask whether\b",
        r"\bit is natural to ask\b", r"\bone (may|might) (wonder|ask)\b", r"\bremains? (open|unknown)\b",
        r"\bit is (unknown|unclear) whether\b", r"\bopen (question|problem)\b", r"\bdo not know whether\b"]),
    # --- value (aesthetic / epistemic) ---
    # NB: trivial/elementary/immediate/routine excluded — math-polysemous (trivial group, elementary
    # topos, immediate consequence). Keep the prototypical authorial hedge only.
    ("value", "easy", "diverge", [r"\beasy to (see|show|prove|check|verify|deduce|find)\b",
        r"\beasily (seen|shown|checked|verified|computed)\b", r"\bstraightforward(ly)?\b",
        r"\bit is (clear|obvious) that\b"]),
    ("value", "plausible", "diverge", [r"\bplausib", r"\bit seems (likely|reasonable|natural)\b",
        r"\bpresumably\b", r"\bconjectural", r"\bheuristical", r"\bwe expect that\b",
        r"\bit is (reasonable|natural) to expect\b"]),
    # NB: nice/striking/surprising/remarkable/pretty excluded — too broad/non-aesthetic. True aesthetic only.
    ("value", "beautiful", "diverge", [r"\bbeautiful\b", r"\belegant(ly)?\b", r"\bexquisite\b",
        r"\baesthetic", r"\bmathematically (elegant|beautiful)\b"]),
    # NB: helpful/invaluable dropped — acknowledgement-prone ("thank X for helpful discussions").
    ("value", "useful", "slot-in", [r"\buseful\b", r"\bconvenient\b",
        r"\bpowerful (tool|technique|method)\b", r"\bkey (tool|ingredient|step)\b",
        r"\binstrumental\b"]),
    # --- meta (reasoning tactics) ---
    ("meta", "goal", "slot-in", [r"\bour goal\b", r"\bthe goal is\b", r"\bwe aim to\b",
        r"\bwe want to (show|prove|establish)\b", r"\bwe seek to\b", r"\bour (aim|objective)\b",
        r"\bin order to (show|prove|establish)\b", r"\bwe wish to (show|prove)\b"]),
    ("meta", "strategy", "slot-in", [r"\bstrateg", r"\bthe (main )?idea (is|behind)\b",
        r"\bour approach\b", r"\bwe proceed by\b", r"\bthe approach\b", r"\bthe key idea\b",
        r"\boutline of the proof\b", r"\bsketch of the proof\b", r"\bproof strategy\b"]),
    ("meta", "auxiliary", "diverge", [r"\bauxiliary\b", r"\bwe first (prove|establish|need)\b",
        r"\bthe following lemma\b", r"\ba (technical|key|preparatory) lemma\b", r"\bpreparatory\b"]),
    # NB: "in the same way" (definitional), parallels/mirrors (mirror symmetry), "similarly to"
    # (broad) dropped; genuine analogy is dominated by analog(y|ous|ue).
    ("meta", "analogy", "duplicate", [r"\banalog(y|ous|ue)\b", r"\bby analogy\b",
        r"\bin the same (spirit|vein)\b", r"\breminiscent of\b", r"\bakin to\b"]),
    ("meta", "implements", "diverge", [r"\bimplements\b", r"\bcarries out\b",
        r"\brealiz(es|ing) the (strategy|idea|programme|program)\b", r"\binto practice\b"]),
    # NB: "extends to"/"can be extended" dropped — usually CONTENT (a map/functor extends to …),
    # not the meta[generalise] tactic. Genuine signal: generali[sz] + "more generally".
    ("meta", "generalise", "duplicate", [r"\bgenerali[sz]", r"\bmore generally\b",
        r"\bin (greater|full) generality\b", r"\bbroader generality\b"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default=DEFAULT_PROPOSALS)
    ap.add_argument("--examples", type=int, default=1, help="example matched quotes per category")
    a = ap.parse_args()

    compiled = [(fam, cat, pred, [re.compile(p) for p in pats]) for fam, cat, pred, pats in CUES]
    prop_hits = collections.Counter()           # category -> # proposals with a cue
    paper_hits = collections.defaultdict(set)    # category -> {papers}
    examples = collections.defaultdict(list)
    kind_dist = collections.Counter()
    all_papers = set()
    n = 0

    for f in sorted(glob.glob(f"{a.proposals}/*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            paper = r.get("paper")
            all_papers.add(paper)
            kind_dist[r.get("kind")] += 1
            q = (r.get("quote") or "").lower()
            if not q:
                continue
            for fam, cat, pred, regs in compiled:
                if any(rx.search(q) for rx in regs):
                    prop_hits[cat] += 1
                    paper_hits[cat].add(paper)
                    if len(examples[cat]) < a.examples:
                        examples[cat].append((r.get("quote") or "")[:120])

    P = len(all_papers)
    print(f"Pass A cue-scan — {n} proposals, {P} distinct papers, {len(glob.glob(f'{a.proposals}/*.jsonl'))} agents\n")
    print(f"{'family':6} {'category':11} {'pred':9} {'props':>6} {'papers':>7} {'%paper':>7}  verdict")
    print("-" * 72)
    for fam, cat, pred, _ in CUES:
        pp = len(paper_hits[cat])
        pct = 100.0 * pp / P if P else 0.0
        verdict = "PRESENT" if pct >= 10 else ("rare" if pct >= 1 else "absent")
        flag = ""
        # flag prediction surprises
        if pred == "diverge" and verdict == "PRESENT":
            flag = "  <-- predicted diverge but PRESENT"
        if pred in ("slot-in", "duplicate") and verdict == "absent":
            flag = "  <-- predicted present but ABSENT"
        print(f"{fam:6} {cat:11} {pred:9} {prop_hits[cat]:>6} {pp:>7} {pct:>6.1f}%  {verdict}{flag}")

    print("\nExisting expository kind distribution (top 12):")
    for k, v in kind_dist.most_common(12):
        print(f"  {v:>6}  {k}")

    print("\nExample matched quotes (sanity-check the cues):")
    for fam, cat, pred, _ in CUES:
        if examples[cat]:
            print(f"  [{cat}] {examples[cat][0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
