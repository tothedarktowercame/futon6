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

import argparse
import bisect
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
GH200_DIR = REPO / "data" / "showcases" / "ct-anatomy" / "gh200"
MARKS_DIR = REPO / "data" / "showcases" / "ct-anatomy" / "golden"
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
        "marks-path": str(mf.relative_to(REPO)),
        "schema": SCHEMA,
    }


PROOF_GAP = 40  # proof-moves within this many lines group into one proof region


def all_passages(marks: list[dict[str, Any]], starts: list[int]) -> list[dict[str, Any]]:
    """ALL proof regions in a paper (whole-paper extraction), not the single best passage.
    Proof-moves within PROOF_GAP lines group into one proof region; each region -> one
    passage (premise = nearest assumption before the region, conclusion = last move).
    Falls back to choose_passage's single conditional/statement passage if no proof-moves."""
    pms = sorted(marks_of(marks, "proof-move"), key=lambda m: (mark_line(m, starts), m["start"]))
    if not pms:
        one = choose_passage(marks, starts)
        return [one] if one else []
    groups = [[pms[0]]]
    for m in pms[1:]:
        if mark_line(m, starts) - mark_line(groups[-1][-1], starts) <= PROOF_GAP:
            groups[-1].append(m)
        else:
            groups.append([m])
    out = []
    for g in groups:
        first_line = mark_line(g[0], starts)
        conclusion = g[-1]
        premises = [m for m in marks if m.get("kind") in {"assume/explicit", "quant/universal"}
                    and 0 <= first_line - mark_line(m, starts) <= 80]
        premise = (sorted(premises, key=lambda m: (first_line - mark_line(m, starts), m["start"]))[0]
                   if premises else g[0])
        out.append({"selection": ":proof-move", "premise": premise, "conclusion": conclusion, "edge": conclusion})
    return out


def extract_all(paper_id: str) -> list[dict[str, Any]]:
    mf = MARKS_DIR / f"fable-{paper_id}-dp-emacs.json"
    if not mf.exists():
        return []
    data = json.loads(mf.read_text())
    text = data["text"]
    starts = line_starts(text)
    marks = [m for m in data["marks"] if "start" in m and "end" in m]
    cands = []
    for i, ch in enumerate(all_passages(marks, starts)):
        p, c = ch["premise"], ch["conclusion"]
        lo = min(mark_line(p, starts), mark_line(c, starts))
        hi = max(mark_line(p, starts), mark_line(c, starts))
        win, win_lines = window_text(text, starts, lo, hi)
        cands.append({
            "paper-id": paper_id,
            "proof-id": f"{paper_id}__p{i}",
            "passage-id": f"{paper_id}:p{i}:{ch['selection'][1:]}:L{win_lines[0]}-{win_lines[1]}",
            "selection": ch["selection"],
            "anchor-lines": {"premise": mark_line(p, starts), "conclusion": mark_line(c, starts)},
            "window-lines": win_lines,
            "binder-context": binder_context(marks, starts, hi + 1),
            "enrichment": window_enrichment(marks, starts, win_lines[0], win_lines[1]),
            "source-window": win,
            "marks-path": str(mf.relative_to(REPO)),
            "schema": SCHEMA,
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
    ap.add_argument("--all-proofs", action="store_true",
                    help="extract EVERY proof region per paper (whole-paper), not one passage")
    a = ap.parse_args()
    papers = a.papers or default_papers()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pid in papers:
        cands = extract_all(pid) if a.all_proofs else ([c] if (c := extract(pid)) else [])
        if not cands:
            print(f"  skip {pid}: no marks or no selectable passage")
            continue
        for cand in cands:
            fid = cand.get("proof-id", pid)
            (outdir / f"{fid}.candidate.json").write_text(json.dumps(cand, indent=2))
            manifest.append({"paper-id": pid, "proof-id": cand.get("proof-id", pid),
                             "passage-id": cand["passage-id"], "selection": cand["selection"],
                             "window-lines": cand["window-lines"]})
        print(f"  {pid}: {len(cands)} proof(s)" if a.all_proofs else
              f"  {pid}: {cands[0]['selection']} lines {cands[0]['window-lines']} "
              f"({len(cands[0]['source-window'])} chars, {len(cands[0]['binder-context'])} binders, "
              f"{len(cands[0]['enrichment'])} anatomy marks)")
    (outdir / "manifest.json").write_text(json.dumps({"papers": manifest}, indent=2))
    n_papers = len({m["paper-id"] for m in manifest})
    print(f"\n{len(manifest)} candidate(s) from {n_papers}/{len(papers)} papers -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
