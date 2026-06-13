#!/usr/bin/env python3
"""Render a paper's DP classification as a paper-anatomy overlay.

Reuses anatomy_v0_sweep's own classifier so the overlay shows the EXACT
current Distributed-Proofreaders run: every control sequence inside a math
span coloured by class — recognised (classified), role-gap (recognised but
un-typed), or genuine unknown. Emits fable-<paper>-dp-emacs.json in the
golden dir, so `M-x paper-anatomy-open` on "<paper>-dp" overlays it.

    dp_paper_view.py 0809.2517
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anatomy_v0_sweep as sweep

EPRINTS = sweep.DEFAULT_EPRINTS
GOLDEN_DIR = Path("/home/joe/code/futon6/data/showcases/ct-anatomy/golden")
CSEQ_RE = re.compile(r"\\([A-Za-z@]+)|\\([^A-Za-z\s])")


def build(paper: str) -> dict:
    eprint = None
    for suffix in (".tar.gz", ".tex.gz", ".gz", ".tar", ".tex"):
        cand = EPRINTS / f"{paper}{suffix}"
        if cand.exists():
            eprint = cand
            break
    if eprint is None:
        raise SystemExit(f"no eprint for {paper} under {EPRINTS}")

    files, _meta = sweep.read_eprint_files(eprint)
    roles = sweep.load_latexml_roles(sweep.ROLE_TSV)
    plain = sweep.load_plain_cseq(sweep.PLAIN_CSEQ)
    macros = sweep.collect_macros(files, roles)

    # Concatenate .tex files into one display text; track each file's base.
    tex_files = [f for f in files if f["file"].endswith(".tex")] or files
    parts, bases, cursor = [], {}, 0
    for f in tex_files:
        header = f"% ===== {f['file']} =====\n"
        parts.append(header)
        cursor += len(header)
        bases[f["file"]] = cursor
        parts.append(f["text"])
        cursor += len(f["text"]) + 1
        parts.append("\n")
    text = "".join(parts)

    marks, counts = [], {"classified": 0, "role-gap": 0, "unknown": 0}
    for f in tex_files:
        base = bases[f["file"]]
        body_text = sweep.strip_comments(f["text"])
        # NB strip_comments may shift offsets; classify on the same text we map.
        ftext = f["text"]
        for start, end, delim, body in sweep.math_spans(ftext):
            body_off = start + len(delim)
            for m in CSEQ_RE.finditer(body):
                cs = m.group(1) or m.group(2)
                cls = sweep.classify_cseq(cs, macros, roles, plain)
                if cls["class"] == "UNKNOWN":
                    kind = "unknown"
                elif cls["role"] == "UNKNOWN":
                    kind = "role-gap"
                else:
                    kind = "classified"
                counts[kind] += 1
                g = base + body_off + m.start()
                marks.append({
                    "start": g,
                    "end": g + (m.end() - m.start()),
                    "layer": "dp",
                    "kind": kind,
                    "tip": f"\\{cs} · {cls['class']} · {cls['role']}"
                           + (f" · {cls.get('source','')}" if cls.get("source") else ""),
                })
    return {"paper": f"{paper}-dp", "text": text, "marks": marks, "_counts": counts}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: dp_paper_view.py <paper-id>")
        return 2
    paper = argv[0]
    data = build(paper)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / f"fable-{paper}-dp-emacs.json"
    out.write_text(json.dumps({k: v for k, v in data.items() if k != "_counts"}))
    c = data["_counts"]
    tot = sum(c.values()) or 1
    print(f"{paper}: {len(data['marks'])} marks — "
          f"classified {c['classified']} ({100*c['classified']//tot}%), "
          f"role-gap {c['role-gap']} ({100*c['role-gap']//tot}%), "
          f"unknown {c['unknown']} ({100*c['unknown']//tot}%)")
    print(f"wrote {out}  →  M-x paper-anatomy-open  RET  {paper}-dp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
