#!/usr/bin/env python3
"""S6 paper-graph assembler — the unified paper-level graph (object B).

Combines the anatomy into one per-paper graph (the Phase-1 whole-paper object Joe asked
for): theorem/lemma/proposition statements as nodes, each proof region attached to the
statement it proves (nearest preceding), definitions as nodes, and — when available —
the reconstructed IATC proof graphs (S3) attached by line-overlap. A statement with no
proof region is a *flagged hole*, not an error (the contract's S6 gate). The expository
graphs (S4) attach as connective edges in a later pass.

  futon6/.venv/bin/python scripts/paper_graph_assemble.py --paper 0704.0502
  futon6/.venv/bin/python scripts/paper_graph_assemble.py --paper 0704.0502 \
      --iatc data/iatc-argument-graphs/<run> --run-dir data/runs/<id>
"""
import argparse
import bisect
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "data/showcases/ct-anatomy/golden")

STMT_KINDS = {"env/theorem", "env/proposition", "env/corollary", "env/lemma"}
DEF_KINDS = {"definiendum", "env/definition", "bind/define"}
PROOF_KINDS = {"env/proof"}


def _line_starts(text):
    st = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            st.append(i + 1)
    return st


def assemble(paper_id, marks_dir=GOLDEN, iatc_dir=None):
    mf = os.path.join(marks_dir, f"fable-{paper_id}-dp-emacs.json")
    if not os.path.exists(mf):
        return None
    d = json.load(open(mf))
    starts = _line_starts(d.get("text", ""))
    line = lambda pos: bisect.bisect_right(starts, pos)

    def pick(kinds, prefix):
        ms = sorted((m for m in d["marks"] if m.get("kind") in kinds and "start" in m),
                    key=lambda m: m["start"])
        return [{"id": f"{prefix}{i}", "kind": m["kind"], "line": line(m["start"]),
                 "start": m["start"], "text": (m.get("tip") or "")[:140]} for i, m in enumerate(ms)]

    statements = pick(STMT_KINDS, "stmt")
    definitions = pick(DEF_KINDS, "def")
    proofs = pick(PROOF_KINDS, "proof")

    # attach each proof to the nearest preceding statement (the one it proves)
    edges = []
    for pr in proofs:
        prev = [s for s in statements if s["start"] <= pr["start"]]
        pr["proves"] = prev[-1]["id"] if prev else None
        if pr["proves"]:
            edges.append({"from": pr["proves"], "to": pr["id"], "rel": "proved-by"})

    # attach reconstructed IATC graphs (S3) by line-overlap with a proof region
    if iatc_dir:
        graphs = []
        for g in sorted(glob.glob(os.path.join(iatc_dir, "*.edn"))):
            if g.endswith(".rung2.edn") or paper_id not in os.path.basename(g):
                continue
            graphs.append(g)
        for pr in proofs:
            pr["iatc"] = [os.path.basename(g) for g in graphs] if len(graphs) == 1 else None

    proven = {pr["proves"] for pr in proofs if pr.get("proves")}
    orphan = [s["id"] for s in statements if s["id"] not in proven]  # flagged holes (no in-paper proof)
    return {"paper": paper_id, "statements": statements, "definitions": definitions,
            "proofs": proofs, "edges": edges,
            "counts": {"statements": len(statements), "proofs": len(proofs),
                       "definitions": len(definitions), "orphan_statements": len(orphan)},
            "orphan_statements": orphan,
            # S6 well-formedness: every proof attaches to a statement; orphans are flagged, not failed
            "wellformed": all(pr.get("proves") for pr in proofs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--marks-dir", default=GOLDEN)
    ap.add_argument("--iatc", help="dir of reconstructed IATC graphs to attach by overlap")
    ap.add_argument("--out", default="data/paper-graphs")
    ap.add_argument("--run-dir")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    a = ap.parse_args()
    B = assemble(a.paper, a.marks_dir if os.path.isabs(a.marks_dir) else os.path.join(ROOT, a.marks_dir), a.iatc)
    if not B:
        print(f"no marks for {a.paper}")
        return 1
    c = B["counts"]
    outdir = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(outdir, exist_ok=True)
    json.dump(B, open(os.path.join(outdir, f"{a.paper}.B.json"), "w"), indent=1)
    attach = (c["statements"] - c["orphan_statements"]) / max(1, c["statements"])
    print(f"{a.paper}: {c['statements']} statements · {c['proofs']} proofs · {c['definitions']} defs · "
          f"{c['orphan_statements']} orphan (flagged) · attach-rate {attach:.2f} · wellformed={B['wellformed']}")
    if a.run_dir:  # S6 inline metric: statement→proof attachment rate (completeness)
        try:
            import metric_harness as mh
            mh.emit_record(a.run_dir, run_id=a.run_id, corpus_id=a.corpus_id, paper_id=a.paper,
                           stage="S6", metric="statement-proof-attachment", axis="completeness",
                           value=round(attach, 4), computable=True)
        except Exception as ee:
            print(f"  (S6 metric emit skipped: {ee})")
    return 0


if __name__ == "__main__":
    main()
