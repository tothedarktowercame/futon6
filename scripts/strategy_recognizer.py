#!/usr/bin/env python3
"""Strategy recognizer v0 — tag NL proof steps with Lean tactic-classes.

Reads holes/clean/tactic-gesture-vocab.edn and classifies each step's text by the
informal GESTURES it contains (E-strategy-recognizer.md). Buckets mirror rung-3 so
the comprehension floor can consume either:
  grounded   — a verifiable tactic gesture matched   (kernel-dischargeable strategy)
  thin       — an automation/heuristic gesture matched (recognized shape, unverified)
  conjecture — an author-declared-gap marker          (credited)
  ungrounded — a residue/hedge marker, or no gesture  (hand-wave / unrecognized)

This is the SEED recognizer: deterministic gesture matching, to be grown on CT.

Usage:
  futon6/.venv/bin/python scripts/strategy_recognizer.py \
      [--steps data/cas-select-steps/loop-run-70b] [--compare-rung3 data/rung3-technique/loop-run-70b]
"""
import argparse
import glob
import json
import os
import re
from collections import Counter
import edn_format as edn


def kwname(x):
    s = str(x)
    return s[1:] if s.startswith(":") else s


def load_vocab(path):
    v = {kwname(k): val for k, val in dict(edn.loads(open(path).read())).items()}
    tv = {}
    for tac, spec in dict(v["tactic-vocab"]).items():
        sd = {kwname(k): val for k, val in dict(spec).items()}
        tv[kwname(tac)] = {"gestures": [str(g) for g in sd["gestures"]],
                           "verifiable": bool(sd.get("verifiable?"))}
    heur = {}
    for tac, spec in dict(v.get("heuristic-tactics", {})).items():
        sd = {kwname(k): val for k, val in dict(spec).items()}
        heur[kwname(tac)] = [str(g) for g in sd.get("gestures", [])]
    return {"tactics": tv, "heuristic": heur,
            "residue": [str(x) for x in v["residue-markers"]],
            "conjecture": [str(x) for x in v["conjecture-markers"]]}


def matches(text, patterns):
    for p in patterns:
        try:
            if re.search(p, text):
                return p
        except re.error:
            if p in text:
                return p
    return None


def classify(text, vocab):
    t = text.lower()
    if matches(t, vocab["conjecture"]):
        return "conjecture", None
    # verifiable tactic?
    for tac, spec in vocab["tactics"].items():
        if spec["verifiable"] and matches(t, spec["gestures"]):
            return "grounded", tac
    # heuristic automation?
    for tac, gs in vocab["heuristic"].items():
        if matches(t, gs):
            return "thin", tac
    if matches(t, vocab["residue"]):
        return "ungrounded", "(hedge)"
    return "ungrounded", None


def strip_latex(text):
    text = re.sub(r"\$[^$]*\$", " ", text)        # inline math
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)    # commands
    text = re.sub(r"[{}\\]", " ", text)             # braces/backslashes
    return text


def recognize_text(text, vocab):
    """Sentence-level recognition over (LaTeX) prose -> bucket + tactic counters."""
    b, tac = Counter(), Counter()
    for sent in re.split(r"(?<=[.!?])\s+|\n", strip_latex(text)):
        sent = sent.strip()
        if len(sent) < 15:
            continue
        bucket, t = classify(sent, vocab)
        b[bucket] += 1
        if bucket == "grounded" and t:
            tac[t] += 1
    return b, tac


def strat_score(b, thin_credit=0.5):
    moves = sum(b.values())
    conj = b.get("conjecture", 0)
    ass = moves - conj
    return (b.get("grounded", 0) + thin_credit * b.get("thin", 0)) / ass if ass > 0 else None


def rung3_score(buckets, thin_credit=0.5):
    moves = sum(buckets.values())
    conj = buckets.get("conjecture", 0)
    grounded = buckets.get("grounded-by-pattern", 0) + buckets.get("grounded-by-citation", 0)
    thin = buckets.get("thin", 0)
    ass = moves - conj
    return (grounded + thin_credit * thin) / ass if ass > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="holes/clean/tactic-gesture-vocab.edn")
    ap.add_argument("--steps", default="data/cas-select-steps/loop-run-70b")
    ap.add_argument("--candidates", default=None,
                    help="dir of *.candidate.json — recognize on each source-window PROSE")
    ap.add_argument("--compare-rung3", default="data/rung3-technique/loop-run-70b")
    ap.add_argument("--thin-credit", type=float, default=0.5)
    args = ap.parse_args()

    vocab = load_vocab(args.vocab)

    if args.candidates:
        print("PROSE-sourced recognition (candidate source-windows) vs claim-level vs rung-3\n")
        print(f"{'paper':14s} {'prose-S':>7s} {'claim-S':>7s} {'rung3-S':>7s}  prose-buckets")
        print("-" * 80)
        tac = Counter()
        for cf in sorted(glob.glob(os.path.join(args.candidates, "*.candidate.json"))):
            pid = os.path.basename(cf).split(".candidate")[0]
            cand = json.load(open(cf))
            window = cand.get("source-window", "")
            b, tc = recognize_text(window, vocab)
            tac.update(tc)
            pS = strat_score(b, args.thin_credit)
            # claim-level (cas-select-steps) for the same paper
            cS = None
            sf = os.path.join(args.steps, f"{pid}.steps.json")
            if os.path.exists(sf):
                cb = Counter()
                for st in json.load(open(sf))["steps"]:
                    cb[classify(st.get("text", ""), vocab)[0]] += 1
                cS = strat_score(cb, args.thin_credit)
            # rung-3
            rS = None
            rp = os.path.join(args.compare_rung3, f"{pid}.technique.json")
            if os.path.exists(rp):
                rS = rung3_score(json.load(open(rp)).get("buckets", {}), args.thin_credit)
            f = lambda x: " -- " if x is None else f"{x:.2f}"
            print(f"{pid:14s} {f(pS):>7s} {f(cS):>7s} {f(rS):>7s}  "
                  f"g{b.get('grounded',0)} t{b.get('thin',0)} u{b.get('ungrounded',0)} c{b.get('conjecture',0)}")
        print("-" * 80)
        print(f"tactic-classes recognized in prose: {dict(tac.most_common())}")
        return
    tac_dist = Counter()
    rows = []
    for f in sorted(glob.glob(os.path.join(args.steps, "*.steps.json"))):
        doc = json.load(open(f))
        pid = doc.get("paper_id", os.path.basename(f).split(".")[0])
        b = Counter()
        for step in doc["steps"]:
            bucket, tac = classify(step.get("text", ""), vocab)
            b[bucket] += 1
            if bucket == "grounded" and tac:
                tac_dist[tac] += 1
        moves = sum(b.values())
        conj = b["conjecture"]
        ass = moves - conj
        S = (b["grounded"] + args.thin_credit * b["thin"]) / ass if ass > 0 else None
        # rung-3 comparison
        r3 = None
        rp = os.path.join(args.compare_rung3, f"{pid}.technique.json")
        if os.path.exists(rp):
            r3 = rung3_score(json.load(open(rp)).get("buckets", {}), args.thin_credit)
        rows.append({"pid": pid, "moves": moves, "b": dict(b), "S": S, "r3": r3})

    def fmt(x):
        return " -- " if x is None else f"{x:.2f}"
    print(f"{'paper':14s} {'moves':>5s} {'recog-S':>7s} {'rung3-S':>7s}  buckets")
    print("-" * 78)
    tg = tt = tu = tc = 0
    for r in rows:
        b = r["b"]
        tg += b.get("grounded", 0); tt += b.get("thin", 0)
        tu += b.get("ungrounded", 0); tc += b.get("conjecture", 0)
        print(f"{r['pid']:14s} {r['moves']:>5d} {fmt(r['S']):>7s} {fmt(r['r3']):>7s}  "
              f"g{b.get('grounded',0)} t{b.get('thin',0)} u{b.get('ungrounded',0)} c{b.get('conjecture',0)}")
    print("-" * 78)
    tot = tg + tt + tu + tc
    print(f"corpus moves={tot}  grounded={tg}  thin={tt}  ungrounded={tu}  conjecture={tc}")
    print(f"recognized (grounded+thin) = {tg+tt}/{tot} = {(tg+tt)/tot:.1%}" if tot else "")
    print(f"\ntactic-classes recognized (grounded): {dict(tac_dist.most_common())}")


if __name__ == "__main__":
    main()
