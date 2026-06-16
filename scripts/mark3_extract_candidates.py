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
        "source-window": win,
        "marks-path": str(mf.relative_to(REPO)),
    }


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
    a = ap.parse_args()
    papers = a.papers or default_papers()
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pid in papers:
        cand = extract(pid)
        if not cand:
            print(f"  skip {pid}: no marks or no selectable passage")
            continue
        (outdir / f"{pid}.candidate.json").write_text(json.dumps(cand, indent=2))
        manifest.append({"paper-id": pid, "passage-id": cand["passage-id"],
                         "selection": cand["selection"], "window-lines": cand["window-lines"]})
        print(f"  {pid}: {cand['selection']} lines {cand['window-lines']} "
              f"({len(cand['source-window'])} chars, {len(cand['binder-context'])} binders)")
    (outdir / "manifest.json").write_text(json.dumps({"papers": manifest}, indent=2))
    print(f"\n{len(manifest)}/{len(papers)} candidates -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
