#!/usr/bin/env python3
"""Full-pipeline witness harness — producer + checker spine, with per-SEAM conformance.

`mark4_pipeline_runner.py` validates only the producer DAG (anatomy -> IATC). This
extends the view to the *whole* pipeline — the mark4 producer AND the checking spine
(rung -1 SFC -> rung 0-2 -> R2d -> CAS-SEL -> CAS-CERT) — declares each stage's
data contract (the SEAM), and traces ONE witness paper through it on a MOCK basis
(on-disk artifacts only, no GPU), reporting at each seam:
    PASS  artifact present + conforms to the next stage's expected shape
    EMPTY artifact present but a downstream-required field is missing/empty
    MISS  artifact not materialized yet
    NA    stage not built yet (in-flight) — N/A, not a failure (gate-is-describer stance)
    GAP   a real seam gap: no producer wires stage N's output into stage N+1's input

    pipeline_witness.py --plan                 # full DAG + seam contracts + status
    pipeline_witness.py --witness 0706.1286    # trace one witness through every seam

Goal: surface the seam gaps BEFORE the next pre-superpod Linode run, so we validate the
wiring end-to-end on a witness first. Pure stdlib + edn_format; deterministic; no network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import edn_format
except Exception:
    edn_format = None

# device: cpu | GPU       status: built | in-flight | gap
# path:   on-disk artifact for the witness (── = no per-paper artifact / corpus-level)
# needs:  the downstream-required fields the seam must carry (conformance contract)
STAGES = [
    {"id": "1.anatomy", "device": "cpu", "status": "built",
     "consumes": ["paper.tex"], "produces": "marks",
     "path": "data/showcases/ct-anatomy/golden/fable-{pid}-dp-emacs.json",
     "needs": ["text", "marks"]},
    {"id": "2.candidates", "device": "cpu", "status": "built",
     "consumes": ["marks"], "produces": "candidate",
     "path": "data/iatc-candidates/{pid}.candidate.json",
     "needs": []},
    {"id": "3.iatc", "device": "GPU", "status": "built",
     "consumes": ["candidate"], "produces": "argument-graph",
     "path": "data/iatc-argument-graphs/loop-run-70b/{pid}.edn",
     "needs": [":nodes", ":edges"]},
    {"id": "4.repair+gate", "device": "cpu", "status": "built",
     "consumes": ["argument-graph"], "produces": "gated-graph",
     "path": "data/iatc-argument-graphs/loop-run-70b/{pid}.edn",
     "needs": [":nodes", ":edges"]},
    # --- checker spine ---
    {"id": "5.semcheck(rung0-2+R2d)", "device": "cpu", "status": "built",
     "consumes": ["gated-graph", "marks"], "produces": "semcheck-profile",
     "path": "data/iatc-argument-graphs/loop-run-70b/{pid}.rung2.edn",
     "needs": []},
    {"id": "5b.cas_segment", "device": "cpu", "status": "built",
     "consumes": ["argument-graph"], "produces": "proof-steps",
     "path": "data/cas-select-steps/loop-run-70b/{pid}.steps.json",
     "needs": ["steps"]},
    {"id": "6.cas_select", "device": "cpu", "status": "built",
     "consumes": ["proof-steps"], "produces": "topology+sorry",
     "path": "data/cas-select-steps/loop-run-70b/{pid}.steps.json",
     "needs": ["steps"]},
    {"id": "7.cas_checks", "device": "cpu", "status": "built",
     "consumes": ["topology+sorry"], "produces": "executed-checks",
     "path": "──", "needs": []},
    {"id": "8.cas_cert", "device": "cpu", "status": "in-flight",
     "consumes": ["semcheck-profile", "topology+sorry"], "produces": "port-ledger",
     "path": "data/cas-cert/{pid}.cert.edn", "needs": []},
    {"id": "9.rung-3", "device": "cpu", "status": "in-flight",
     "consumes": ["topology+sorry"], "produces": "technique-ports",
     "path": "──", "needs": []},
]


def _load(path: Path):
    txt = path.read_text()
    if path.suffix == ".edn":
        if edn_format is None:
            return ("edn", txt)            # can't parse; presence-only
        return ("edn", edn_format.loads(txt))
    return ("json", json.loads(txt))


def _has(obj, key):
    """True if key is present + non-empty. Unwraps the (kind, data) pair from _load;
    for unparsed edn text, falls back to a ':key' substring match."""
    kind = None
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] in ("edn", "json"):
        kind, obj = obj
    if kind == "edn" and isinstance(obj, str):
        return key in obj                      # presence-only on unparsed edn text
    if hasattr(obj, "get"):
        if key.startswith(":") and edn_format is not None:
            try:
                if obj.get(edn_format.Keyword(key[1:])) not in (None, [], {}, ""):
                    return True
            except Exception:
                pass
        return obj.get(key) not in (None, [], {}, "")
    return False


def witness(pid: str) -> int:
    print(f"=== witness trace: {pid} — full pipeline (producer + checker spine) ===\n")
    print(f"  {'stage':24} {'dev':3} {'status':9} seam")
    gaps, misses = [], []
    for s in STAGES:
        rel = s["path"].replace("{pid}", pid)
        mark = ""
        if s["status"] == "gap":
            state = "GAP"
            gaps.append(s)
        elif rel == "──":
            state = "NA " if s["status"] == "in-flight" else "—  "
        else:
            f = REPO / rel
            if not f.exists():
                state = "NA " if s["status"] == "in-flight" else "MISS"
                if s["status"] != "in-flight":
                    misses.append(s)
            else:
                try:
                    obj = _load(f)
                    missing = [k for k in s["needs"] if not _has(obj, k)]
                    if missing:
                        state, mark = "EMPTY", f" missing {missing}"
                    else:
                        state = "PASS"
                except Exception as e:
                    state, mark = "ERR ", f" {type(e).__name__}: {e}"
        produces = s["produces"]
        print(f"  {state:5} {s['id']:24} {s['device']:3} {s['status']:9} →{produces}{mark}")
    print()
    print(f"  artifacts MISSING (built stage, not materialized for this witness): "
          f"{[s['id'] for s in misses] or 'none'}")
    print(f"  SEAM GAPS (no producer wires the seam): {[s['id'] for s in gaps] or 'none'}")
    for g in gaps:
        print(f"    · {g['id']}: consumes {g['consumes']} but nothing produces it for arXiv papers")
    return 0


def plan() -> int:
    print("=== full pipeline DAG — producer + checker spine, with seam contracts ===\n")
    produced = {"paper.tex"}
    ok = True
    for s in STAGES:
        unmet = [c for c in s["consumes"] if c not in produced]
        flag = "✓" if not unmet else f"✗ unmet {unmet}"
        print(f"  [{s['id']:24}] {s['device']:3} {s['status']:9} consumes {s['consumes']} {flag}")
        produced.add(s["produces"])
        if unmet and s["status"] not in ("gap",):
            ok = False
    print(f"\n  DAG order valid (each stage's inputs produced upstream): {'✓' if ok else '✗'}")
    print("  NOTE: stage 5b.cas_segment is the seam-6 producer: it segments arXiv IATC")
    print("        graphs into proof-steps consumed by 6.cas_select. APM hand-authored")
    print("        fixtures remain the CAS-SEL oracle path; arXiv now has a deterministic")
    print("        CPU proof-step producer.")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--witness", metavar="PID")
    a = ap.parse_args(argv)
    if a.witness:
        return witness(a.witness)
    return plan()


if __name__ == "__main__":
    raise SystemExit(main())
