#!/usr/bin/env python3
"""Salvage the ~50% good C-entries from a pre-fix backward run, using the gate as a PER-ITEM filter.

The improved rerun will supersede this, but the good half is real signal we can wire the downstream off
NOW (the belly's reach/correction channels). Keep only gate-passing records:
  reach      — has a non-empty assistant_span (I1 evidence; drops the fabricated empty-span reaches),
  correction — a GENUINE pivot (PIVOT cue present, not agreement-open) AND a named redirect target.
Everything else (recap/agreement false-corrections, empty reaches) is dropped. Output is marked
provisional so the downstream knows it's replaceable by the clean rerun.

  futon6/.venv/bin/python scripts/salvage_c_entries.py data/c-vector/c-entries.openai.pre-instr-fix.json
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_goals_holes_gates import PIVOT, AGREE_OPEN, aspan, rspan, gref

def keep(e):
    fl = e.get("flavour")
    if fl == "reach":
        return bool(aspan(e).strip())
    if fl == "correction":
        rs = rspan(e)
        named = str((e.get("preferred") or {}).get("value") or "").strip()
        return bool(PIVOT.search(rs) and not AGREE_OPEN.match(rs) and len(named) >= 4)
    return False

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/c-vector/c-entries.openai.pre-instr-fix.json"
    d = json.load(open(src))
    kept = [e for e in d if keep(e)]
    for e in kept:
        e.setdefault("provenance", {})["salvage"] = {"from": os.path.basename(src), "provisional": True}
    out = "data/c-vector/c-entries.salvaged.json"
    json.dump(kept, open(out, "w"), indent=2)
    inn = Counter(e.get("flavour") for e in d); outc = Counter(e.get("flavour") for e in kept)
    print(f"salvage {src}")
    print(f"  in:  {len(d)}  {dict(inn)}")
    print(f"  out: {len(kept)}  {dict(outc)}  ({100*len(kept)//max(1,len(d))}% kept)")
    print(f"  dropped reach (empty span): {inn['reach']-outc['reach']} · dropped correction (false): {inn['correction']-outc['correction']}")
    print(f"  wrote {out} (marked provisional)")

if __name__ == "__main__":
    main()
