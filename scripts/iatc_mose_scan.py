#!/usr/bin/env python3
"""E-iatc-expository-alignment, MO/SE baseline + alternative-phrasing harvest.

The arXiv Pass A (iatc_alignment_passA.py) found certain IATC categories rare in published
expository prose. The claim is "stripped on publication, not nonexistent" — which only holds
if those categories are COMMON in dialogue. This scans the MathOverflow / math.SE sample
threads (IATC's native habitat, an original IATC data source) with the SAME cues, so the
arXiv-vs-dialogue gap is apples-to-apples, and breaks out the COMMENT layer (the multi-agent
back-and-forth where Agree/Challenge/Retract live).

It also (a) runs an informal-dialogical probe — MO/SE-native phrasings the arXiv-tuned cues
miss — and (b) harvests example matched sentences per category. Both feed the LLM step:
generalise these into alternative phrasings of each cue (improves Pass A recall + Pass B's
exemplar bank).

Usage:
  python scripts/iatc_mose_scan.py [--glob 'PATH/*.jsonl'] [--examples 3] [--dump OUT.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iatc_alignment_passA import CUES  # single source of truth for the cue lexicon

DEFAULT_GLOB = "/home/joe/code/futon5/data/stackexchange-samples/*.jsonl"

# audited arXiv Pass A %papers, for the side-by-side gap column
ARXIV_REF = {
    "Agree": 1.0, "Challenge": 1.6, "Retract": 0.0, "Suggest": 21.2, "Query": 16.6,
    "easy": 51.3, "plausible": 13.0, "beautiful": 7.8, "useful": 54.9, "goal": 39.4,
    "strategy": 21.2, "auxiliary": 26.9, "analogy": 52.3, "implements": 3.1, "generalise": 72.5,
}

# MO/SE-native dialogical phrasings the arXiv-tuned cues miss (the harvest's first fruit)
INFORMAL = {
    "Agree": [r"\bi agree\b", r"\bgood (point|question|catch)\b", r"\byou'?re right\b",
              r"\bthat'?s right\b", r"\bexactly\b", r"\bfair enough\b"],
    "Challenge": [r"\bthis is (wrong|false)\b", r"\bthat'?s not (right|correct|true)\b",
                  r"\bi don'?t think (this|that|so)\b", r"\bwhy would\b", r"\bbut surely\b",
                  r"\bthat can'?t be\b"],
    "Retract": [r"\boops\b", r"\bmy (mistake|bad)\b", r"\bi was wrong\b", r"\bnever ?mind\b",
                r"\bi stand corrected\b", r"\bon second thought\b", r"\bedit:\b", r"\bcorrection\b"],
}

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def units_of(thread: dict):
    """Yield (unit_type, text) for question/answers/comments."""
    q = thread.get("question") or {}
    if q.get("body_text"):
        yield "question", q["body_text"]
    for a in thread.get("answers") or []:
        if a.get("body_text"):
            yield "answer", a["body_text"]
    cs = thread.get("comments") or {}
    # comments.question is a list; comments.answers is a dict {answer_id: [comment, ...]}
    buckets = []
    qcs = cs.get("question")
    if isinstance(qcs, list):
        buckets.append(qcs)
    acs = cs.get("answers")
    if isinstance(acs, dict):
        buckets.extend(v for v in acs.values() if isinstance(v, list))
    elif isinstance(acs, list):
        buckets.append(acs)
    for lst in buckets:
        for c in lst:
            if isinstance(c, dict) and c.get("text"):
                yield "comment", c["text"]


def first_sentence_with(text: str, rx: re.Pattern) -> str:
    for s in SENT_SPLIT.split(text):
        if rx.search(s.lower()):
            return s.strip()[:160]
    return text.strip()[:160]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--dump", default=None, help="write harvested examples to JSON")
    a = ap.parse_args()

    compiled = [(fam, cat, pred, [re.compile(p) for p in pats]) for fam, cat, pred, pats in CUES]
    informal = {c: [re.compile(p) for p in pats] for c, pats in INFORMAL.items()}

    thread_hits = collections.defaultdict(set)       # cat -> {thread}
    unit_hits = collections.Counter()                 # cat -> # units
    comment_hits = collections.defaultdict(set)       # cat -> {thread} via a comment
    informal_threads = collections.defaultdict(set)   # cat -> {thread} (informal cues)
    examples = collections.defaultdict(list)
    n_threads = 0
    unit_count = collections.Counter()

    for f in sorted(glob.glob(a.glob)):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            tid = t.get("thread_id") or id(t)
            n_threads += 1
            for utype, text in units_of(t):
                unit_count[utype] += 1
                low = text.lower()
                for fam, cat, pred, regs in compiled:
                    if any(rx.search(low) for rx in regs):
                        unit_hits[cat] += 1
                        thread_hits[cat].add(tid)
                        if utype == "comment":
                            comment_hits[cat].add(tid)
                        if len(examples[cat]) < a.examples:
                            rx = next(rx for rx in regs if rx.search(low))
                            examples[cat].append({"unit": utype, "sent": first_sentence_with(text, rx)})
                for cat, regs in informal.items():
                    if any(rx.search(low) for rx in regs):
                        informal_threads[cat].add(tid)

    T = n_threads
    print(f"MO/SE baseline — {T} threads "
          f"(q={unit_count['question']}, answers={unit_count['answer']}, comments={unit_count['comment']}) "
          f"from {len(glob.glob(a.glob))} files\n")
    print(f"{'family':6} {'category':11} {'arXiv%':>7} {'MOSE%thr':>9} {'%via-cmt':>9}  gap")
    print("-" * 64)
    for fam, cat, pred, _ in CUES:
        mose = 100.0 * len(thread_hits[cat]) / T if T else 0.0
        cmt = 100.0 * len(comment_hits[cat]) / T if T else 0.0
        ax = ARXIV_REF.get(cat, float("nan"))
        gap = ""
        if cat in ("Agree", "Challenge", "Retract", "implements", "beautiful"):
            # the "stripped" set: dialogue baseline should be much higher than arXiv
            if mose >= max(10.0, 3 * ax):
                gap = "  <-- STRIPPED-ON-PUBLICATION confirmed (common in dialogue, rare in arXiv)"
        print(f"{fam:6} {cat:11} {ax:>6.1f}% {mose:>8.1f}% {cmt:>8.1f}%{gap}")

    print("\nInformal-dialogical probe (MO/SE-native phrasings the arXiv cues miss; %threads):")
    for cat in ("Agree", "Challenge", "Retract"):
        strict = 100.0 * len(thread_hits[cat]) / T if T else 0.0
        inf = 100.0 * len(informal_threads[cat]) / T if T else 0.0
        print(f"  {cat:11} strict-cue {strict:5.1f}%   informal {inf:5.1f}%   (undercount factor ~{inf/strict:.0f}x)"
              if strict else f"  {cat:11} strict-cue {strict:5.1f}%   informal {inf:5.1f}%")

    print("\nHarvested example phrasings (Tier-2 exemplars + LLM alt-phrasing seed):")
    for fam, cat, pred, _ in CUES:
        for e in examples[cat][:1]:
            print(f"  [{cat:11}] ({e['unit']}) {e['sent']!r}")

    if a.dump:
        Path(a.dump).write_text(json.dumps({c: examples[c] for c in examples}, indent=2))
        print(f"\nharvested examples -> {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
