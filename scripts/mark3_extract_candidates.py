#!/usr/bin/env python3
"""mark3 IATC candidate extractor — the SCRIPTED half of the model loop.

codex-4's honest H2 finding: a deterministic mark parser CAN select candidate
argument passages + line anchors, but CANNOT faithfully reconstruct the warranted
DAG — that needs an LLM reading the passage. So this script does only the
script-doable half: pick the passage, anchor it, and pull the source window +
binder context the model needs to read. It emits NO graph (that was the rejected
generate_iatc_gh200.py shell mistake). The model loop (mark3_iatc_loop.py) turns
each candidate into a reconstructed graph and self-gates it.

Selection logic is salvaged verbatim from the (otherwise-rejected) gh200 generator.

Usage:
    python scripts/mark3_extract_candidates.py --out data/iatc-candidates [--papers a b c]
    # default: 10 gh200 papers that have marks and are NOT in the accepted pilot.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import futon6_config as config
import stage_accounting as accounting


import argparse
import bisect
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
GH200_DIR = REPO / "data" / "showcases" / "ct-anatomy" / "gh200"
MARKS_DIR = config.marks()
PILOT_DIR = REPO / "data" / "iatc-argument-graphs" / "gh200"
CONTEXT_LINES = 4  # window padding around the selected passage
SCHEMA = "iatc-candidate/v2-enriched"  # bumped when the candidate payload changes

# Deterministic-anatomy mark kinds worth inlining for the model (symbol typings +
# structural anchors). Excludes the dense low-level noise (classified/math/raw symbol).
ENRICH_KINDS = {
    "symbol-grounded", "bind/typed", "bind/define", "bind/let", "let-binder",
    "definiendum", "definiens", "assume/explicit", "quant/universal", "proof-move",
    "constrain/relation", "constrain/where", "constrain/such-that",
    "label", "cite", "env/proof", "env/lemma", "env/theorem",
    "env/proposition", "env/corollary",
}
ENRICH_CAP = 60  # bound prompt size; windows are small so this rarely bites

# --- salvaged selection helpers (verbatim from generate_iatc_gh200.py @ c20fdd3) ---


def line_starts(text: str) -> list[int]:
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


def line_for(starts: list[int], pos: int) -> int:
    return bisect.bisect_right(starts, pos)


def mark_line(mark: dict[str, Any], starts: list[int]) -> int:
    return line_for(starts, int(mark["start"]))


def marks_of(marks: list[dict[str, Any]], *kinds: str) -> list[dict[str, Any]]:
    wanted = set(kinds)
    return [m for m in marks if m.get("kind") in wanted]


def env_marks(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good = {"env/theorem", "env/lemma", "env/proposition", "env/corollary"}
    return [m for m in marks if m.get("kind") in good]


def choose_passage(marks: list[dict[str, Any]], starts: list[int]) -> dict[str, Any] | None:
    proof_moves = sorted(marks_of(marks, "proof-move"), key=lambda m: (mark_line(m, starts), m["start"]))
    if proof_moves:
        conclusion = proof_moves[0]
        c_line = mark_line(conclusion, starts)
        premises = [
            m for m in marks
            if m.get("kind") in {"assume/explicit", "quant/universal"}
            and 0 <= c_line - mark_line(m, starts) <= 80
        ]
        premise = sorted(premises, key=lambda m: (c_line - mark_line(m, starts), m["start"]))[0] if premises else conclusion
        return {"selection": ":proof-move", "premise": premise, "conclusion": conclusion, "edge": conclusion}

    assumptions = sorted(marks_of(marks, "assume/explicit"), key=lambda m: (mark_line(m, starts), m["start"]))
    consequents = sorted(marks_of(marks, "quant/universal") + env_marks(marks), key=lambda m: (mark_line(m, starts), m["start"]))
    for premise in assumptions:
        p_line = mark_line(premise, starts)
        after = [m for m in consequents if 0 <= mark_line(m, starts) - p_line <= 80]
        if after:
            return {"selection": ":conditional-passage", "premise": premise, "conclusion": after[0], "edge": after[0]}

    envs = sorted(env_marks(marks), key=lambda m: (mark_line(m, starts), m["start"]))
    if envs:
        return {"selection": ":statement-passage", "premise": envs[0], "conclusion": envs[0], "edge": envs[0]}
    return None


# --- candidate extraction (new: emit reading material, not a graph) ---


def window_text(text: str, starts: list[int], lo_line: int, hi_line: int) -> tuple[str, list[int]]:
    a = max(1, lo_line - CONTEXT_LINES)
    b = min(len(starts), hi_line + CONTEXT_LINES)
    start_char = starts[a - 1]
    end_char = starts[b] if b < len(starts) else len(text)
    return text[start_char:end_char], [a, b]


def binder_context(marks: list[dict[str, Any]], starts: list[int], before_line: int) -> list[str]:
    """let-binders/definienda before the passage — the variable typing the model needs."""
    out = []
    for m in marks:
        if m.get("kind") in {"let-binder", "bind/let", "definiendum", "definiens"} \
                and mark_line(m, starts) < before_line and m.get("tip"):
            out.append(f"({m['kind']}) {m['tip']}")
    return out[-12:]  # nearest dozen


def window_enrichment(marks: list[dict[str, Any]], starts: list[int],
                      lo_line: int, hi_line: int) -> list[dict[str, Any]]:
    """The deterministic anatomy the detector found INSIDE the candidate window —
    symbol->type groundings, definitions, quantifiers, proof-moves, citations. This
    is the enrichment that previously never reached the model (marks-path was a dead
    pointer); inlining it here is what makes the candidate self-contained."""
    out = []
    for m in marks:
        if m.get("kind") in ENRICH_KINDS and m.get("tip"):
            ln = mark_line(m, starts)
            if lo_line <= ln <= hi_line:
                out.append({"line": ln, "kind": m["kind"], "tip": m["tip"]})
    out.sort(key=lambda r: (r["line"], r["kind"]))
    return out[:ENRICH_CAP]


def extract(paper_id: str) -> dict[str, Any] | None:
    mf = MARKS_DIR / f"fable-{paper_id}-dp-emacs.json"
    if not mf.exists():
        return None
    data = json.loads(mf.read_text())
    text = data["text"]
    starts = line_starts(text)
    marks = [m for m in data["marks"] if "start" in m and "end" in m]
    chosen = choose_passage(marks, starts)
    if not chosen:
        return None
    p, c = chosen["premise"], chosen["conclusion"]
    lo = min(mark_line(p, starts), mark_line(c, starts))
    hi = max(mark_line(p, starts), mark_line(c, starts))
    win, win_lines = window_text(text, starts, lo, hi)
    return {
        "paper-id": paper_id,
        "passage-id": f"{paper_id}:{chosen['selection'][1:]}:L{win_lines[0]}-{win_lines[1]}",
        "selection": chosen["selection"],
        "anchor-lines": {"premise": mark_line(p, starts), "conclusion": mark_line(c, starts)},
        "window-lines": win_lines,
        "binder-context": binder_context(marks, starts, hi + 1),
        "enrichment": window_enrichment(marks, starts, win_lines[0], win_lines[1]),
        "source-window": win,
        "marks-path": _display(mf),
        "schema": SCHEMA,
    }


def _display(path: Path) -> str:
    # Run-owned marks may live outside the checkout (--run-dir /scratch/...).
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


STATEMENT_KINDS = {"env/theorem", "env/lemma", "env/proposition", "env/corollary"}
STATEMENT_GAP = 20  # a statement ending further than this above its proof is not shown with it
SCHEMA_PROOF = "iatc-candidate/v3-proof"  # one candidate per S1-identified proof


def proof_regions(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Outermost S1 proof regions in source order.

    A proof nested inside another proof is part of that proof's argument, so it is
    reconstructed with it rather than as a second, overlapping candidate.
    """
    proofs = sorted((m for m in marks if m.get("kind") == "env/proof"),
                    key=lambda m: (m["start"], -m["end"]))
    outer: list[dict[str, Any]] = []
    for m in proofs:
        if outer and m["start"] < outer[-1]["end"]:
            continue
        outer.append(m)
    return outer


def extract_all(paper_id: str) -> list[dict[str, Any]]:
    """One candidate per proof that S1 identified, with the statement it proves.

    This replaced grouping `proof-move` marks ("it is easy to see", "clearly") within
    40 lines: those groups were not proofs. On the historical 98-graph run only
    42 of their windows overlapped any proof region, so most model work went into
    arbitrary stretches of prose. Text outside proofs is exposition (S4).
    """
    mf = MARKS_DIR / f"fable-{paper_id}-dp-emacs.json"
    if not mf.exists():
        return []
    data = json.loads(mf.read_text())
    text = data["text"]
    starts = line_starts(text)
    marks = [m for m in data["marks"] if "start" in m and "end" in m]
    statements = sorted((m for m in marks if m.get("kind") in STATEMENT_KINDS), key=lambda m: m["start"])
    cands = []
    for i, proof in enumerate(proof_regions(marks)):
        p_lo = line_for(starts, proof["start"])
        p_hi = line_for(starts, max(proof["start"], proof["end"] - 1))
        before = [m for m in statements if m["start"] <= proof["start"]]
        statement = before[-1] if before else None
        lo = p_lo
        proved = None
        if statement is not None:
            s_lo = line_for(starts, statement["start"])
            s_hi = line_for(starts, max(statement["start"], statement["end"] - 1))
            proved = {"kind": statement["kind"].split("/", 1)[1], "lines": [s_lo, s_hi],
                      "text": text[statement["start"]:statement["end"]][:3000]}
            if p_lo - s_hi <= STATEMENT_GAP:
                lo = s_lo
        start_char = starts[lo - 1]
        end_char = starts[p_hi] if p_hi < len(starts) else len(text)
        cands.append({
            "paper-id": paper_id,
            "proof-id": f"{paper_id}__p{i}",
            "passage-id": f"{paper_id}:proof{i}:L{lo}-{p_hi}",
            "selection": ":proof",
            "proof-lines": [p_lo, p_hi],
            "proved": proved,
            "window-lines": [lo, p_hi],
            "binder-context": binder_context(marks, starts, lo),
            "enrichment": window_enrichment(marks, starts, lo, p_hi),
            "source-window": text[start_char:end_char].rstrip("\n"),
            "marks-path": _display(mf),
            "schema": SCHEMA_PROOF,
        })
    return cands


def default_papers() -> list[str]:
    gh = sorted(p.stem for p in GH200_DIR.glob("*.html"))
    pilot = {p.stem for p in PILOT_DIR.glob("*.edn")}
    out = []
    for pid in gh:
        if pid in pilot:
            continue
        if (MARKS_DIR / f"fable-{pid}-dp-emacs.json").exists():
            out.append(pid)
        if len(out) >= 10:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "iatc-candidates"))
    ap.add_argument("--papers", nargs="*", help="paper ids; default = 10 non-pilot gh200 with marks")
    ap.add_argument("--list", help="file of paper ids, one per line (same as emit_marks --list)")
    ap.add_argument("--all-proofs", action="store_true",
                    help="one candidate per proof identified by S1 (the Mark7 path), not one legacy passage")
    a = ap.parse_args()
    papers = a.papers or (a.list and [l.strip() for l in open(a.list) if l.strip()]) or default_papers()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    # Every requested paper is accounted for. A paper that cannot be extracted is
    # an errored item, not a silent omission from the frozen corpus.
    ledger = accounting.Accounting("S3", "extract", papers)
    manifest = []
    for pid in papers:
        if not (MARKS_DIR / f"fable-{pid}-dp-emacs.json").exists():
            ledger.record(pid, "errored", f"no S1 marks at {MARKS_DIR}", paper=pid)
            print(f"  ERROR {pid}: no marks")
            continue
        try:
            cands = extract_all(pid) if a.all_proofs else ([c] if (c := extract(pid)) else [])
        except Exception as exc:
            ledger.record(pid, "errored", f"extraction raised {type(exc).__name__}: {exc}", paper=pid)
            print(f"  ERROR {pid}: {exc}")
            continue
        written = []
        for cand in cands:
            fid = cand.get("proof-id", pid)
            path = outdir / f"{fid}.candidate.json"
            path.write_text(json.dumps(cand, indent=2))
            written.append((fid, path))
            manifest.append({"paper-id": pid, "proof-id": cand.get("proof-id", pid),
                             "passage-id": cand["passage-id"], "selection": cand["selection"],
                             "window-lines": cand["window-lines"]})
        # A paper whose anatomy has no proof region is a legitimate, explicit zero.
        ledger.record(pid, "accepted", "" if cands else "no proof identified by S1",
                      paper=pid, artifacts=[accounting.relative(p) for _, p in written],
                      outputs=[f for f, _ in written])
        if not cands:
            print(f"  {pid}: 0 proofs identified by S1")
            continue
        print(f"  {pid}: {len(cands)} proof(s)" if a.all_proofs else
              f"  {pid}: {cands[0]['selection']} lines {cands[0]['window-lines']} "
              f"({len(cands[0]['source-window'])} chars, {len(cands[0]['binder-context'])} binders, "
              f"{len(cands[0]['enrichment'])} anatomy marks)")
    (outdir / "manifest.json").write_text(json.dumps({"papers": manifest}, indent=2))
    n_papers = len({m["paper-id"] for m in manifest})
    print(f"\n{len(manifest)} candidate(s) from {n_papers}/{len(papers)} papers -> {outdir}")
    return 1 if ledger.failed() else 0


if __name__ == "__main__":
    raise SystemExit(main())
