#!/usr/bin/env python3
"""Generate checker-gated IATC argument graphs for the gh200 CT showcase.

The generator uses the deterministic layer-(a) marks as scaffold. It emits one
standoff graph per paper id found in data/showcases/ct-anatomy/gh200 when the
matching golden/fable-<id>-dp-emacs.json exists.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
GH200_DIR = REPO / "data/showcases/ct-anatomy/gh200"
GOLDEN_DIR = REPO / "data/showcases/ct-anatomy/golden"
DEFAULT_OUT = REPO / "data/iatc-argument-graphs/gh200"


def edn_key(s: str) -> str:
    return ":" + s


def edn_string(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def edn(obj: Any, indent: int = 0) -> str:
    sp = " " * indent
    if obj is None:
        return "nil"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, str):
        if obj.startswith(":"):
            return obj
        return edn_string(obj)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        return "[" + " ".join(edn(x, indent) for x in obj) + "]"
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = list(obj.items())
        lines = ["{"]
        for i, (k, v) in enumerate(items):
            prefix = " " * (indent + 1)
            key = edn_key(k) if not str(k).startswith(":") else str(k)
            suffix = "," if i < len(items) - 1 else ""
            lines.append(f"{prefix}{key} {edn(v, indent + 2)}{suffix}")
        lines.append(sp + "}")
        return "\n".join(lines)
    raise TypeError(f"cannot render {type(obj)!r}")


def line_starts(text: str) -> list[int]:
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


def line_for(starts: list[int], pos: int) -> int:
    return bisect.bisect_right(starts, pos)


def source_for(mark: dict[str, Any], starts: list[int]) -> dict[str, Any]:
    start = int(mark["start"])
    end = int(mark["end"])
    return {
        "lines": [line_for(starts, start), line_for(starts, max(start, end - 1))],
        "chars": [start, end],
        "mark/kind": ":" + mark.get("kind", "unknown").replace("/", "-"),
    }


def mark_line(mark: dict[str, Any], starts: list[int]) -> int:
    return line_for(starts, int(mark["start"]))


def wanted_for(mark: dict[str, Any], text: str) -> str:
    tip = (mark.get("tip") or "").lower()
    span = text[int(mark["start"]):int(mark["end"])].lower()
    probe = tip + " " + span
    span_kind = mark.get("kind") or ""
    if "suffices" in probe or "sufficient" in probe or "reduction" in probe:
        return ":reduction-justification-elided"
    if "wlog" in probe or "without loss" in probe or "symmetry" in probe:
        return ":symmetry-or-wlog-justification"
    if "similar" in probe:
        return ":analogy-to-previous-proof-elided"
    if any(w in probe for w in ["easy", "clear", "obvious", "straightforward", "immediate", "trivial"]):
        return ":local-verification-elided"
    if span_kind == "assume/explicit":
        return ":conditional-to-conclusion-link"
    if span_kind.startswith("env/"):
        return ":statement-warrant-elided"
    return ":prose-illative-warrant-elided"


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


def graph_for(paper_id: str, json_path: Path) -> dict[str, Any] | None:
    data = json.loads(json_path.read_text())
    text = data["text"]
    starts = line_starts(text)
    marks = [m for m in data["marks"] if "start" in m and "end" in m]
    chosen = choose_passage(marks, starts)
    if not chosen:
        return None

    premise = chosen["premise"]
    conclusion = chosen["conclusion"]
    edge_mark = chosen["edge"]
    src = source_for(edge_mark, starts)
    line_a, line_b = src["lines"]
    wanted = wanted_for(edge_mark, text)
    passage_id = f"{paper_id}:{chosen['selection'][1:]}:L{line_a}-{line_b}:{src['chars'][0]}-{src['chars'][1]}"

    return {
        "paper/id": paper_id,
        "passage/id": passage_id,
        "source": {
            "lines": src["lines"],
            "chars": src["chars"],
            "kind": ":argument-passage",
            "selection": chosen["selection"],
        },
        "provenance": {
            "generator": "scripts/generate_iatc_gh200.py",
            "scaffold": str(json_path.relative_to(REPO)),
            "layer-a": ":dp-emacs",
            "standoff-only": True,
        },
        "nodes": [
            {
                "id": ":passage-premise",
                "kind": ":claim",
                "source": source_for(premise, starts),
            },
            {
                "id": ":passage-conclusion",
                "kind": ":claim",
                "source": source_for(conclusion, starts),
            },
        ],
        "edges": [
            {
                "id": ":e-main",
                "kind": ":infer",
                "relation": ":because",
                "premise": [":passage-premise"],
                "warrant": {
                    "kind": ":missing-warrant",
                    "wanted": wanted,
                },
                "conclusion": ":passage-conclusion",
                "source": src,
            }
        ],
        "holes": [
            {
                "kind": ":missing-warrant",
                "edge": ":e-main",
                "wanted": wanted,
                "source": src,
            }
        ],
    }


def paper_ids(limit: int | None = None) -> list[str]:
    ids = sorted(p.stem for p in GH200_DIR.glob("*.html"))
    return ids[:limit] if limit else ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    emitted = 0
    skipped: list[dict[str, str]] = []
    for paper_id in paper_ids(args.limit):
        total += 1
        json_path = GOLDEN_DIR / f"fable-{paper_id}-dp-emacs.json"
        if not json_path.exists():
            skipped.append({"paper": paper_id, "reason": "missing-golden-json"})
            continue
        graph = graph_for(paper_id, json_path)
        if graph is None:
            skipped.append({"paper": paper_id, "reason": "no-argument-scaffold"})
            continue
        (out / f"{paper_id}.edn").write_text(edn(graph) + "\n")
        emitted += 1

    (out / "MANIFEST.txt").write_text(
        "\n".join(
            [
                f"total-gh200-html={total}",
                f"emitted-edn={emitted}",
                f"skipped={len(skipped)}",
                *[f"skip {s['paper']} {s['reason']}" for s in skipped],
                "",
            ]
        )
    )
    print(f"total={total} emitted={emitted} skipped={len(skipped)} out={out}")
    for s in skipped:
        print(f"skip {s['paper']} {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
