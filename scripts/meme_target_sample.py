#!/usr/bin/env python3
"""TARGETED-SAMPLE: sample human→agent asks ABOUT a given rollout mission (turn→mission by term-match).

Random sampling gives ~0 rollout-mission coverage (the asks don't name the missions). To validate
meme→move→ΔG end-to-end on CPU BEFORE the box, sample asks that mention a target mission's distinctive
terms — a cheap, model-free, high-precision proxy for turn→pattern→mission. The mined memes then resolve
to that mission → bridge to a move → ΔG with real provenance.

  --scan [--top N]      rank rollout missions by # matching asks (which missions ARE meme-groundable here)
  --mission <stem> [--k N]   dump the targeted ask sample for one mission
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meme_mine_runner import read_asks  # the auto-excluding, thread-windowed ask reader

ROOT = "/home/joe/code/futon6"
STOP = {"the", "and", "for", "with", "from", "into", "mission", "futon", "code", "scoping", "review", "via"}


def mission_stems():
    nodes = {}
    for s in json.load(open(f"{ROOT}/data/diffsub-scopes.json")):
        m = s.get("mission")
        if m and m not in nodes and s.get("mission_node"):
            nodes[m] = s["mission_node"]
    return nodes


def terms(stem):
    toks = [t for t in re.split(r"[-/]", stem.lower()) if len(t) >= 4 and t not in STOP]
    return stem.replace("-", " ").lower(), toks


def matches(ask_low, phrase, toks):
    if phrase in ask_low:
        return True
    return sum(1 for t in set(toks) if re.search(rf"\b{re.escape(t)}", ask_low)) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--mission")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    asks = read_asks(None)
    lows = [s["ask"].lower() for s in asks]
    stems = mission_stems()

    if a.scan:
        counts = []
        for stem in stems:
            phrase, toks = terms(stem)
            if not toks:
                continue
            c = sum(1 for lo in lows if matches(lo, phrase, toks))
            if c:
                counts.append((c, stem))
        counts.sort(reverse=True)
        print(f"asks: {len(asks)} · rollout missions: {len(stems)} · missions with >=1 matching ask: {len(counts)}")
        print(f"top {a.top} meme-groundable missions (by # matching asks):")
        for c, stem in counts[:a.top]:
            print(f"  {c:4d}  M-{stem}")
        return

    if a.mission:
        stem = a.mission[2:] if a.mission.startswith("M-") else a.mission
        phrase, toks = terms(stem)
        hits = [s for s, lo in zip(asks, lows) if matches(lo, phrase, toks)][:a.k]
        print(f"M-{stem}  terms={toks}  → {len(hits)} targeted asks (showing {min(a.k,len(hits))}):\n")
        for i, s in enumerate(hits):
            print(f"[{i}] {s['id']} ({s['project'][:22]})\n    {s['ask'][:200]}\n")
        os.makedirs(f"{ROOT}/data/meme-mine", exist_ok=True)
        json.dump({"mission": stem, "asks": hits},
                  open(f"{ROOT}/data/meme-mine/target-{stem}.json", "w"), indent=2)
        print(f"wrote data/meme-mine/target-{stem}.json")
        return
    ap.error("use --scan | --mission <stem>")


if __name__ == "__main__":
    main()
