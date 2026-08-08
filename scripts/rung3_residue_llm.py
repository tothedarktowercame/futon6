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

import llm_json
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
        {"model": model, "messages": [{"role": "user", "content": prompt}],
         "temperature": 0,
         # H26: uncapped requests let the model generate to the context limit;
         # in cas_select that cost ~15 min/call and read as a slow endpoint.
         "max_tokens": int(os.environ.get("FUTON6_LLM_MAX_TOKENS", "512"))}
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=int(os.environ.get("FUTON6_LLM_TIMEOUT", "300"))) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    # Shared with cas_select via llm_json: the greedy {.*} here spanned the first
    # `{` to the LAST `}`, and bare property names (the observed GLM failure) were
    # not repaired, so a recoverable reply degraded to the deterministic template.
    # That is worse than a loud failure -- the run still emits a question, but a
    # template one, which inflates the apparent yield of the LLM pass.
    parsed = llm_json.parse_object(txt, {})
    if not parsed:
        print(f"[rung3_residue_llm] unparseable model JSON for step {gap.get('step')}; "
              "falling back to the menu template")
    classification, question, source = _fields(parsed)
    if question is None:
        question, source = template_q, "template"
    return {"classification": classification, "rm_pattern": rm_pattern,
            "question": question, "source": source}


# The prompt asks for {"classification", "question"}; GLM-4.5-Air answers with
# {"decision": "REAL GAP", "open_question": "..."}. The JSON parsed fine, both
# lookups missed, and the code substituted the default classification and the
# TEMPLATE question -- silently, with no warning, because nothing had failed.
# That is the "reports success without doing the work" hazard class: the run
# emitted a full set of questions, none of which the model had written.
# Aliases are cheap; assuming a model honours a requested schema is not.
_CLASS_KEYS = ("classification", "decision", "verdict", "label")
_QUESTION_KEYS = ("question", "open_question", "open-question", "q")


def _norm_class(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower().replace(" ", "-").replace("_", "-")
    if v in CLASSIFICATIONS:
        return v
    if "novel" in v:
        return "novel-technique"
    if "gap" in v:
        return "real-gap"
    return None


def _fields(parsed: dict[str, Any]) -> tuple[str, str | None, str]:
    """(classification, question, provenance). Provenance distinguishes a model
    answer from a template fallback, so a run can report how much of its output
    the model actually wrote."""
    cls = next((c for k in _CLASS_KEYS if (c := _norm_class(parsed.get(k)))), None)
    q = next((str(parsed[k]).strip() for k in _QUESTION_KEYS
              if isinstance(parsed.get(k), str) and parsed[k].strip()), None)
    return (cls or "real-gap", q, "model" if q else "template")


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
                # Whether the model wrote this question or the template did. A
                # residue pass that silently emits templates looks identical to
                # one that worked; this is the field that tells them apart.
                "source": res.get("source", "template"),
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
            # How much of this document the model actually wrote. Without it, a
            # pass that emitted only templates reports the same shape as one the
            # model answered in full.
            "model_written": sum(1 for q in questions if q.get("source") == "model"),
            "template_fallback": sum(1 for q in questions if q.get("source") != "model"),
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
