#!/usr/bin/env python3
"""rung-3-3 — bounded LLM pass over rung-3-2's thin/ungrounded residue.

For each residue gap (bucket in {thin, ungrounded}) from a rung-3-2 gap-map — and
ONLY those, never a grounded move — classify *novel-technique vs real-gap* and emit
a phrased **ArSE question** via the RM question-pattern menu
(holes/excursions/rung-3-spec.md, "Gap to ArSE question mapping"). The output is a
QUESTION, never a truth/correctness verdict.

Backends, env-split like scripts/sfc_symbol_grounding.py:
  - `stub`   : deterministic, no network — fills the menu template (for tests).
  - `openai` : real LLaMA-70B via OPENAI_BASE_URL (the mark3_iatc_loop client shape).

Bounded: iterates only the residue; one model call per emitted question (asserted);
--max-questions caps the budget and logs the drop.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "rung3-questions" / "loop-run-70b"
RESIDUE_BUCKETS = ("thin", "ungrounded")

# RM question-pattern menu — source of truth: holes/excursions/rung-3-spec.md
# "Gap to ArSE question mapping". Keyed by gap bucket.
QUESTION_MENU: dict[str, dict[str, str]] = {
    "thin": {
        "rm_pattern": "STRUCTURAL PROBE",
        "template": "What verifiable inference discharges the heuristic step {pattern} here?",
    },
    # The spec's template is "…from <premise> to <conclusion>", but an IATC move
    # carries undivided prose — filling both slots with it duplicated the text
    # ("from X to X"), so the single-descriptor variant of the same menu entry
    # is used instead.
    "ungrounded": {
        "rm_pattern": "THEOREM APPLICABILITY / TECHNIQUE LANDSCAPE",
        "template": (
            "Which known theorem or proof technique, if any, licenses this move: "
            "“{move}”?"
        ),
    },
}
MENU_SOURCE = "holes/excursions/rung-3-spec.md#gap-to-arse-question-mapping"
CLASSIFICATIONS = ("novel-technique", "real-gap")


def load_gapmap(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def residue_gaps(gapmap: dict[str, Any]) -> list[dict[str, Any]]:
    return [g for g in (gapmap.get("gaps") or []) if g.get("bucket") in RESIDUE_BUCKETS]


def moves_by_step(gapmap: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m.get("step"): m for m in (gapmap.get("moves") or [])}


def phrase(gap: dict[str, Any], move: dict[str, Any] | None) -> tuple[str, str]:
    """Phrase the gap as a question from the RM menu (deterministic given the gap)."""
    menu = QUESTION_MENU.get(gap.get("bucket"), QUESTION_MENU["ungrounded"])
    pattern = gap.get("pattern") or "the matched pattern"
    text = ((move or {}).get("text") or "").strip()
    descriptor = text if text else "this move"
    question = menu["template"].format(pattern=pattern, move=descriptor)
    return menu["rm_pattern"], question


def call_stub(gap: dict[str, Any], move: dict[str, Any] | None, model: str) -> dict[str, Any]:
    """Deterministic: the stub does NOT judge novelty (that is the model's job) — it
    conservatively marks residue as a real gap and fills the menu template."""
    rm_pattern, question = phrase(gap, move)
    return {"classification": "real-gap", "rm_pattern": rm_pattern, "question": question}


def call_openai(gap: dict[str, Any], move: dict[str, Any] | None, model: str) -> dict[str, Any]:
    import urllib.request

    rm_pattern, template_q = phrase(gap, move)
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "x")
    prompt = (
        "A proof move was only weakly grounded.\n"
        f"bucket: {gap.get('bucket')}\n"
        f"matched pattern: {gap.get('pattern')}\n"
        f"move: {((move or {}).get('text') or '').strip()}\n\n"
        "Decide whether this is a NOVEL technique the author is genuinely using, or a "
        "REAL GAP that needs more work. Then phrase ONE open question for the author/"
        f'reader, guided by this template: "{template_q}". Do NOT judge whether the '
        "mathematics is true. Reply with JSON only: "
        '{"classification":"novel-technique"|"real-gap","question":"..."}'
    )
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        # A malformed model reply must not kill a corpus pass — degrade this one
        # gap to the deterministic template (same shape as the stub) and say so.
        print(f"[rung3_residue_llm] unparseable model JSON for step {gap.get('step')}; "
              "falling back to the menu template")
        parsed = {}
    classification = parsed.get("classification")
    if classification not in CLASSIFICATIONS:
        classification = "real-gap"
    question = parsed.get("question") or template_q
    return {"classification": classification, "rm_pattern": rm_pattern, "question": question}


def questions_for_gapmap(
    gapmap: dict[str, Any],
    *,
    backend: str = "stub",
    model: str = "mark4-70b",
    max_questions: int | None = None,
) -> dict[str, Any]:
    paper_id = gapmap.get("paper_id") or gapmap.get("paper-id")
    moves = moves_by_step(gapmap)
    residue = residue_gaps(gapmap)
    backend_fn = call_openai if backend == "openai" else call_stub

    calls = 0
    questions: list[dict[str, Any]] = []
    for gap in residue:
        if max_questions is not None and len(questions) >= max_questions:
            break
        res = backend_fn(gap, moves.get(gap.get("step")), model)
        calls += 1
        questions.append(
            {
                "step": gap.get("step"),
                "bucket": gap.get("bucket"),
                "pattern": gap.get("pattern"),
                "classification": res["classification"],
                "rm_pattern": res["rm_pattern"],
                "question": res["question"],
                # ArSE-ready shape (a typed-bell :query/:ref) — NOT opened here.
                "ref": f"arse:{paper_id}:{gap.get('step')}",
            }
        )
    # bounded invariant: exactly one model call per emitted question, and the loop
    # only ever visits residue gaps — so the model never touches a grounded move.
    assert calls == len(questions)
    dropped = max(0, len(residue) - len(questions))
    if dropped:
        print(f"[rung3_residue_llm] --max-questions dropped {dropped} residue gap(s)")
    return {
        "schema": "rung3-questions/v0",
        "paper_id": paper_id,
        "menu_source": MENU_SOURCE,
        "backend": backend,
        "questions": questions,
        "summary": {
            "residue": len(residue),
            "asked": len(questions),
            "novel": sum(1 for q in questions if q["classification"] == "novel-technique"),
            "gap": sum(1 for q in questions if q["classification"] == "real-gap"),
            "dropped_by_budget": dropped,
        },
    }


def write_doc(doc: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gapmap", type=Path, help="rung-3-2 gap-map JSON")
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    gapmap = load_gapmap(args.gapmap)
    doc = questions_for_gapmap(
        gapmap, backend=args.backend, model=args.model, max_questions=args.max_questions
    )
    out_path = args.out or (args.out_dir / f"{doc['paper_id']}.questions.json")
    write_doc(doc, out_path)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
