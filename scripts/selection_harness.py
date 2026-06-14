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


def load_learn_module():
    spec = importlib.util.spec_from_file_location("cascade_learn", FUTON6 / "scripts/cascade_learn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan_real_closures_for_competitive_beta(closures_path, computed):
    """The HONEST competitive-β detector (claude-3's surfacer correction): scan REAL recorded folds,
    not prospective/detached scopes. A competitive β = a FAILED fold whose used pattern is COMPUTED-
    connected within its actual :used set (B''s own gate). Cross-domain mis-retrievals are edge-isolated
    by construction (the phylogeny encodes co-application history), so they route to the coverage-gap;
    only a 'right-cluster-wrong-member' real failure yields a connected β. Reports none until one is
    recorded organically — same recording-discipline as the first β. Fabricates nothing."""
    cl = load_learn_module()
    records = cl.parse_closure_folds(str(closures_path))
    connected_failures, isolated_failures = [], []
    for rec in records:
        if bool(rec.get("success")):
            continue
        used = [cl.stem(p) for p in rec.get("used", [])]
        for p in used:
            (connected_failures if cl.connected_within(p, used, computed) else isolated_failures).append(
                (rec.get("scope"), p))
    return connected_failures, isolated_failures


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

    changed = []
    for s in scopes:
        q = scope_query(s)
        off = cc.construct_cascade(q, posterior_weight=0.0)
        on = cc.construct_cascade(q, posterior_weight=args.weight, posterior_table=table)
        if chosen_ids(off) != chosen_ids(on):
            changed.append(s.get("scope_id"))

    print(f"scopes={len(scopes)} weight={args.weight} posterior-table={table.get('label')} n-evidence={len(posteriors)}")
    print(f"SELECTION-DELTA: {len(changed)}/{len(scopes)} scopes change chosen set when grounded posterior is switched ON")
    for sid in changed[:15]:
        print(f"  changed: {sid}")

    # HONEST competitive-β detector: scan REAL closures (not prospective scopes) for connected failures.
    cl = load_learn_module()
    computed = cl.load_computed(str(FUTON6 / "data/pattern-phylogeny-edges.json"))
    connected_failures, isolated_failures = scan_real_closures_for_competitive_beta(
        FUTON6 / "holes/closure-folds.edn", computed)
    print(f"COMPETITIVE-β IN THE REAL LEDGER (connected failed-fold members): {len(connected_failures)}")
    for scope, p in connected_failures[:15]:
        print(f"  competitive-β: {p} @ {scope}")
    print(f"  (isolated failures routed to coverage-gap, NOT utility-β: {len(isolated_failures)} -> {[p for _,p in isolated_failures]})")
    if not connected_failures:
        print("  -> none yet. A competitive β awaits the next REAL connected-pattern failure (recording-discipline), not a fabricated one.")


if __name__ == "__main__":
    main()
