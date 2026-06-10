#!/usr/bin/env python3
"""B' selection harness — measure where the grounded posterior, switched ON, changes cascade SELECTION.

Two outputs:
  1. SELECTION-DELTA: for each detached scope, does the chosen pattern set differ between
     posterior_weight=0 (default off) and posterior_weight=W (grounded table on)?
  2. COMPETITIVE-β CANDIDATES: scopes whose chosen cascade contains a CONNECTED pattern (>=1
     semi-lattice edge to another chosen pattern). A *failed* fold on such a hole would be a
     competitive β (connected -> B' charges utility -> changes future selection). These are the
     holes to hand to claude-3's E-ground-G fold-judgment surface (DO NOT fabricate the outcome).

Read-only; writes nothing to any ledger. Uses futon3a/.venv (sentence_transformers).
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path("/home/joe/code")
FUTON6 = ROOT / "futon6"
CASCADE_PATH = ROOT / "futon3a/holes/labs/M-memes-arrows/cascade_construct.py"
DEFAULT_SCOPES = FUTON6 / "data/diffsub-scopes.json"
DEFAULT_POSTERIORS = FUTON6 / "data/pattern_posteriors.grounded.json"


def load_cascade_module():
    spec = importlib.util.spec_from_file_location("cascade_construct", CASCADE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scope_query(scope):
    parts = [scope.get("scope_id", ""), scope.get("mission", ""), scope.get("passage", ""),
             scope.get("capability") or "", " ".join(scope.get("concepts") or [])]
    return " ".join(str(p) for p in parts if p)


def chosen_ids(cascade):
    return [pid for _, pid, _ in cascade["trajectory"]]


def connected_evidence_members(cascade, posteriors, cc):
    """Chosen patterns that BOTH carry posterior evidence AND are connected in the semi-lattice."""
    sl = cascade["semi-lattice"]
    connected = set()
    for a, b in sl["descent"]:
        connected.add(a); connected.add(b)
    for edge in sl["co_app"]:
        connected.add(edge[0]); connected.add(edge[1])
    out = []
    for pid in chosen_ids(cascade):
        if pid in connected and cc.pattern_stem(pid) in posteriors:
            out.append(pid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", default=str(DEFAULT_SCOPES))
    ap.add_argument("--posteriors", default=str(DEFAULT_POSTERIORS))
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="cap scopes processed (0=all detached)")
    args = ap.parse_args()

    cc = load_cascade_module()
    table = cc.load_posteriors(args.posteriors) if Path(args.posteriors).exists() else cc.load_posteriors()
    posteriors = table.get("patterns", {})
    scopes = [s for s in json.loads(Path(args.scopes).read_text()) if s.get("state") == "detached"]
    if args.limit:
        scopes = scopes[: args.limit]

    changed, candidates = [], []
    for s in scopes:
        q = scope_query(s)
        off = cc.construct_cascade(q, posterior_weight=0.0)
        on = cc.construct_cascade(q, posterior_weight=args.weight, posterior_table=table)
        if chosen_ids(off) != chosen_ids(on):
            changed.append(s.get("scope_id"))
        members = connected_evidence_members(on, posteriors, cc)
        if members:
            candidates.append((s.get("scope_id"), members))

    print(f"scopes={len(scopes)} weight={args.weight} posterior-table={table.get('label')} n-evidence={len(posteriors)}")
    print(f"SELECTION-DELTA: {len(changed)}/{len(scopes)} scopes change chosen set when grounded posterior is switched ON")
    for sid in changed[:15]:
        print(f"  changed: {sid}")
    print(f"COMPETITIVE-β CANDIDATES (connected evidence-bearing members -> a FAILED fold here = competitive β): {len(candidates)}")
    for sid, members in candidates[:15]:
        print(f"  {sid}: {[cc.pattern_stem(m) for m in members]}")


if __name__ == "__main__":
    main()
