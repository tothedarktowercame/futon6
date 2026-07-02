#!/usr/bin/env python3
"""Gates-as-code for the PROOF-MINE runner (D2) — per futon6/holes/proof-mine-runner-spec.md.

Torch-free, GPU-free, stdlib-only. Runs on the LAPTOP before the box is rented (smoke-before-the-
paid-run) — the same move the fold-embed gate made when it caught a null ablation before spend.
Author != producer: this checker is independent of proof_mine.py.

Gates (HARD ones exit nonzero so linode-proof-mine.sh bails BEFORE any GPU call):
  G-doc-found      every target mission's doc is on disk (else the dossier is empty).
  G-code-trail     every dossier has ≥1 citing commit OR the explicit :no-code-trail flag (consistency).
  G-canonical      every mission id resolves canonically via the mission-index bridge (D6).
  G-gold-parses    every A-next gold EMPIRICAL few-shot file parses (bb if present, else loose loader).
  G-c-entries      (SOFT) each dossier has ≥1 c-entry attached — report, don't block.

  python scripts/check_proof_mine_gates.py                 # gate the 10 A-next gold missions
  python scripts/check_proof_mine_gates.py --missions M-x M-y
"""
import argparse, glob, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proof_mine_dossier import assemble                       # noqa: E402
from proof_mine import (mission_index, resolve_ref, GOLD_MISSIONS, GOLD_DIR,  # noqa: E402
                        _load_edn_loose)

OK = lambda b: "PASS" if b else "FAIL"                         # noqa: E731


def gold_files():
    files = []
    for stem in GOLD_MISSIONS:
        files += glob.glob("%s/A-next-%s/*-sorry-EMPIRICAL.edn" % (GOLD_DIR, stem))
    return files


def _bb_parses(path):
    """True if `bb` can read the EDN (authoritative), or the loose loader recovers fields (fallback)."""
    try:
        r = subprocess.run(["bb", "-e", '(clojure.edn/read-string (slurp "%s"))' % path],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    g = _load_edn_loose(path)                                  # fallback: did we recover any fields?
    return bool(g.get("endpoints") or g.get("grades"))


def run_gates(missions):
    idx = mission_index()
    fails = []
    print("== PROOF-MINE gates ==  missions=%d  mission-index=%d canonical ids" % (len(missions), len(idx)))

    dossiers = {m: assemble(m) for m in missions}

    # G-doc-found (HARD)
    missing = [m for m, d in dossiers.items() if not d.get("doc_found")]
    hard = not missing
    fails += [] if hard else ["G-doc-found"]
    print("  [%s] G-doc-found     missing-doc=%s" % (OK(hard), missing or "none"))

    # G-nonempty (HARD): a found dossier must carry SOME evidence — commits OR endpoints OR
    # c-entries. All-three-empty means the assembler (or the mission-stem match) silently returned
    # nothing — the empty-cascades / used-var-bundle failure class the fold gate was built to catch.
    empty = [m for m, d in dossiers.items()
             if d.get("doc_found") and not d["commits"] and not d["endpoints"] and not d["c_entries"]]
    hard = not empty
    fails += [] if hard else ["G-nonempty"]
    print("  [%s] G-nonempty      silently-empty-dossiers=%s" % (OK(hard), empty or "none"))

    # G-canonical (HARD): every found mission id resolves canonically
    unresolved = []
    for m, d in dossiers.items():
        if not d.get("doc_found"):
            continue
        ref, ok = resolve_ref(d["mission"], idx)
        if not ok:
            unresolved.append(m)
    hard = not unresolved
    fails += [] if hard else ["G-canonical"]
    print("  [%s] G-canonical     unresolvable-mission-ids=%s" % (OK(hard), unresolved or "none"))

    # G-gold-parses (HARD): the yardstick itself must parse
    gf = gold_files()
    unparsed = [os.path.basename(p) for p in gf if not _bb_parses(p)]
    hard = bool(gf) and not unparsed
    fails += [] if hard else ["G-gold-parses"]
    print("  [%s] G-gold-parses   gold-files=%d unparsed=%s" % (OK(hard), len(gf), unparsed or "none"))

    # G-c-entries (SOFT): report dossiers with no c-entries attached
    noce = [m for m, d in dossiers.items() if d.get("doc_found") and not d["c_entries"]]
    print("  [%s] G-c-entries     dossiers-with-no-c-entries=%s  (SOFT)"
          % (OK(not noce), noce or "none"))

    print("== PROOF-MINE gates: %s ==" % ("PASS" if not fails else "FAIL " + ",".join(fails)))
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="*", help="mission stems to gate (default = the 10 A-next gold)")
    a = ap.parse_args()
    missions = a.missions or GOLD_MISSIONS
    sys.exit(0 if run_gates(missions) else 1)


if __name__ == "__main__":
    main()
