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

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import futon6_config as config
import stage_accounting as accounting
from paper_ids import paper_id_from_name

import argparse
import bisect
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = str(config.marks())

STMT_KINDS = {"env/theorem", "env/proposition", "env/corollary", "env/lemma"}
DEF_KINDS = {"definiendum", "env/definition", "bind/define"}
PROOF_KINDS = {"env/proof"}


def _line_starts(text):
    st = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            st.append(i + 1)
    return st


def _source_lines(path, kind=None):
    text = open(path, errors="replace").read()
    tail = rf", :kind :{kind}" if kind else ""
    m = re.search(r":source \{:lines \[(\d+) (\d+)\]" + tail, text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def assemble(paper_id, marks_dir=GOLDEN, iatc_dir=None, expo_dir=None):
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
                 "end_line": line(m.get("end", m["start"])),
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

    # Attach this paper's accepted S3 graphs to the proof region their :kind :proof
    # source lines overlap most. Exact paper-id parsing, not substring matching.
    # Graphs that overlap no proof region are listed, not hidden: they mean S1 and
    # S3 disagree about where the proofs are.
    unattached_iatc, refused = [], []
    for pr in proofs:
        pr["iatc"] = []
    if iatc_dir:
        # Only finals with verified acceptance provenance are consumed.
        graphs, refused_all = accounting.accepted_finals(iatc_dir)
        refused += [r for r in refused_all if paper_id_from_name(r[0]) == paper_id]
        for _, g in graphs:
            g = str(g)
            if paper_id_from_name(os.path.basename(g)) != paper_id:
                continue
            span = _source_lines(g, "proof")
            best = max(proofs, key=lambda pr: _overlap(span, pr), default=None) if span else None
            if best is not None and _overlap(span, best) > 0:
                best["iatc"].append(os.path.basename(g))
            else:
                unattached_iatc.append({"graph": os.path.basename(g), "lines": list(span) if span else None})

    # S4 expository scopes are part of the whole-paper object (DAG contract S6→S4).
    expository = []
    if expo_dir:
        scopes, refused_expo = accounting.accepted_finals(expo_dir)
        refused += [r for r in refused_expo if paper_id_from_name(r[0]) == paper_id]
        for passage, g in scopes:
            if paper_id_from_name(g.name) == paper_id:
                expository.append({"scope": g.name, "passage": passage, "lines": _source_lines(g)})

    proven = {pr["proves"] for pr in proofs if pr.get("proves")}
    orphan = [s["id"] for s in statements if s["id"] not in proven]  # flagged holes (no in-paper proof)
    return {"paper": paper_id, "statements": statements, "definitions": definitions,
            "proofs": proofs, "edges": edges, "expository": expository,
            "unattached_iatc": unattached_iatc, "refused_inputs": [list(r) for r in refused],
            "counts": {"statements": len(statements), "proofs": len(proofs),
                       "definitions": len(definitions), "orphan_statements": len(orphan),
                       "iatc_attached": sum(len(pr["iatc"]) for pr in proofs),
                       "iatc_unattached": len(unattached_iatc), "expository": len(expository)},
            "orphan_statements": orphan,
            # S6 well-formedness: every proof attaches to a statement; orphans are flagged, not failed
            "wellformed": all(pr.get("proves") for pr in proofs)}


def _overlap(span, pr):
    lo, hi = span
    return max(0, min(hi, pr["end_line"]) - max(lo, pr["line"]) + 1)


def assemble_one(paper, a, marks_dir):
    """Write one paper's B object; return (status, reason, path) for accounting."""
    B = assemble(paper, marks_dir, a.iatc, a.expo)
    if not B:
        print(f"no marks for {paper}")
        return "errored", f"no S1 marks in {marks_dir}", None
    c = B["counts"]
    out = a.out or os.path.join("data", "iatc-paper-graphs", a.run_id or "adhoc")
    outdir = out if os.path.isabs(out) else os.path.join(ROOT, out)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{paper}.B.json")
    json.dump(B, open(path, "w"), indent=1)
    attach = (c["statements"] - c["orphan_statements"]) / max(1, c["statements"])
    print(f"{paper}: {c['statements']} statements · {c['proofs']} proofs · {c['definitions']} defs · "
          f"{c['orphan_statements']} orphan (flagged) · attach-rate {attach:.2f} · "
          f"iatc {c['iatc_attached']} attached/{c['iatc_unattached']} unattached · "
          f"{c['expository']} expository · wellformed={B['wellformed']}")
    if a.run_dir:  # S6 inline metric: statement→proof attachment rate (completeness)
        try:
            import metric_harness as mh
            mh.emit_record(a.run_dir, run_id=a.run_id, corpus_id=a.corpus_id, paper_id=paper,
                           stage="S6", metric="statement-proof-attachment", axis="completeness",
                           value=round(attach, 4), computable=True)
        except Exception as ee:
            print(f"  (S6 metric emit skipped: {ee})")
    if B["refused_inputs"]:
        # Stale or foreign upstream output for this paper: nothing downstream may
        # treat the paper object as complete.
        return "rejected", f"refused upstream outputs: {B['refused_inputs'][:3]}", path
    if not B["wellformed"]:
        # A proof that attaches to no statement is a defect in the whole-paper
        # object, not a note: returning 0 here let two malformed objects sit
        # behind a passing stage ledger entry.
        unattached = [pr.get("id") for pr in B.get("proofs", []) if not pr.get("proves")]
        print(f"✗ {paper}: NOT well-formed — {len(unattached)} proof(s) attach to no "
              f"statement, e.g. {unattached[:3]}")
        lines = [pr["line"] for pr in B["proofs"] if not pr.get("proves")]
        return "rejected", (f"not well-formed: {len(unattached)} proof region(s) precede every "
                            f"statement (lines {lines[:5]})"), path
    return "accepted", "", path


def main():
    ap = argparse.ArgumentParser()
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--paper")
    which.add_argument("--list", help="file of paper ids; every paper is assembled and accounted")
    ap.add_argument("--marks-dir", default=GOLDEN)
    ap.add_argument("--iatc", help="dir of accepted IATC graphs (S3) to attach by line overlap")
    ap.add_argument("--expo", help="dir of accepted expository scopes (S4) to include")
    # RETRIEVE collects data/iatc-paper-graphs/<run-id>; the old default wrote to
    # data/paper-graphs, so S6's product was outside the retrieval manifest and
    # would have been destroyed at cluster teardown.
    ap.add_argument("--out", default=None,
                    help="default: data/iatc-paper-graphs/<run-id> (the RETRIEVE path)")
    ap.add_argument("--run-dir")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    a = ap.parse_args()
    if a.run_dir and "adhoc" in (a.run_id, a.corpus_id):
        ap.error("--run-dir requires explicit --run-id and --corpus-id (records would be tagged adhoc)")
    marks_dir = a.marks_dir if os.path.isabs(a.marks_dir) else os.path.join(ROOT, a.marks_dir)
    papers = [a.paper] if a.paper else [l.strip() for l in open(a.list) if l.strip()]
    # Every paper is evaluated even after one fails: the loop used to stop at the
    # first malformed object, leaving later papers with neither artifact nor verdict.
    ledger = accounting.Accounting("S6", "assemble", papers)
    for paper in papers:
        status, reason, path = assemble_one(paper, a, marks_dir)
        ledger.record(paper, status, reason, paper=paper,
                      artifacts=[accounting.relative(path)] if path else [], outputs=[paper])
    counts = ledger.counts()
    if len(papers) > 1:
        print(f"\nS6: {counts['accepted']}/{len(papers)} paper objects accepted · "
              f"{counts['rejected']} rejected · {counts['errored']} errored")
    return 1 if ledger.failed() else 0


if __name__ == "__main__":
    # main() returns a status; discarding it made every failure exit 0, so the
    # S6 loop's `|| exit 1` could never fire and two malformed paper objects sat
    # behind a passing stage ledger entry.
    sys.exit(main() or 0)
