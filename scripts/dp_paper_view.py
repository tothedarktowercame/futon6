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
import importlib.util as _ilu


def _load_nlab_wiring():
    """Import the superpod scope detector (filename has a hyphen)."""
    p = Path(__file__).resolve().parent / "nlab-wiring.py"
    spec = _ilu.spec_from_file_location("nlab_wiring", p)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

EPRINTS = sweep.DEFAULT_EPRINTS
GOLDEN_DIR = Path("/home/joe/code/futon6/data/showcases/ct-anatomy/golden")
CSEQ_RE = re.compile(r"\\([A-Za-z@]+)|\\([^A-Za-z\s])")
# "Let $X$ be a <concept> ..." — the most regular binder in mathematics
# (W2: was dark). Subject = the $-symbol; concept = the noun phrase to the
# first clause boundary. Also catches "and $Y$ a <concept>" conjuncts.
BINDER_RE = re.compile(
    r"\b(?:Let|let)\s+(\$[^$]+\$)\s+(?:be|denote)\s+(?:an?\s+|the\s+)?"
    r"([^.,;:]+?)(?=[.,;:]|\s+such that|\s+and\s+\$|\s+in\s+\$|$)")
CONJUNCT_RE = re.compile(
    r"\band\s+(\$[^$]+\$)\s+(?:be\s+)?(?:an?\s+|the\s+)?"
    r"([^.,;:]+?)(?=[.,;:]|\s+such that|\s+and\s+\$|$)")


def _concept_head(phrase: str) -> str:
    """Last 1-3 words of a concept phrase, math/markup stripped, for lookup."""
    words = re.findall(r"[A-Za-z][A-Za-z-]+", re.sub(r"\$[^$]*\$|[\\{}]", " ", phrase))
    return " ".join(words[-3:]) if words else ""


def detect_binders(ftext, base, ca):
    """Emit let-binder scope marks for one file's text (global offsets)."""
    out = []
    for m in BINDER_RE.finditer(ftext):
        subj, phrase = m.group(1), m.group(2).strip()
        # scope extent = the whole "Let $X$ be <concept>" / "and $Y$ <concept>"
        binders = [(subj, phrase, m.start(), m.end())]
        for cm in CONJUNCT_RE.finditer(ftext, m.end(), m.end() + 160):
            binders.append((cm.group(1), cm.group(2).strip(),
                            cm.start(), cm.end()))
        for subj, phrase, ss, se in binders:
            concept = None
            if ca is not None:
                head = _concept_head(phrase)
                hit = ca.resolve(head) if head else None
                if hit:
                    concept = f"{hit.get('term')} [{hit.get('target')}]"
            out.append({
                "start": base + ss, "end": base + se,
                "layer": "dp", "kind": "let-binder",
                "tip": f"binds {subj} : {phrase[:60]}"
                       + (f" · concept: {concept}" if concept else ""),
                # structured fields → Scratch-style nested render in the panel
                "fields": [["binds", subj],
                           ["as", phrase[:70]],
                           ["canon", concept or "— (unresolved)"]],
            })
    return out


def detect_scope_manifest(ftext, base, entity_id, nw, ca):
    """Port the superpod scope detector (nlab-wiring.detect_scopes) onto one
    file's text, mapping to global offsets and paper-anatomy mark shape.
    The full ~40-type manifest, not a hand-rolled subset."""
    out = []
    for s in nw.detect_scopes(entity_id, ftext):
        content = s.get("hx/content", {})
        pos, end = content.get("position"), content.get("end")
        if pos is None or end is None or end <= pos:
            continue
        stype = s.get("hx/type", "scope")
        ends = s.get("hx/ends", [])
        fields = []
        for e in ends:
            role = e.get("role")
            val = e.get("latex") or e.get("text")
            if role and role != "entity" and val:
                fields.append([role, str(val)[:70]])
        # concept-type the bound symbol's type phrase if the authority knows it
        if ca is not None:
            for e in ends:
                if e.get("role") == "type" and e.get("text"):
                    hit = ca.resolve(_concept_head(e["text"]))
                    if hit:
                        fields.append(["canon", f"{hit.get('term')} [{hit.get('target')}]"])
                        break
        out.append({
            "start": base + pos, "end": base + min(end, pos + 200),
            "layer": "scope", "kind": stype,
            "tip": f"{stype} · " + " | ".join(f"{r}:{v}" for r, v in fields[:3]),
            "fields": fields or None,
        })
    return out


def build(paper: str, with_ca: bool = False, with_binders: bool = False,
          with_scopes: bool = False) -> dict:
    ca = None
    if with_ca or with_binders or with_scopes:
        from concept_authority import ConceptAuthority
        ca = ConceptAuthority()
    nw = _load_nlab_wiring() if with_scopes else None
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

    marks, counts = [], {"classified": 0, "role-gap": 0, "unknown": 0,
                         "concept-typed": 0, "let-binder": 0}
    for f in tex_files:
        base = bases[f["file"]]
        ftext = f["text"]
        if with_binders:
            bm = detect_binders(ftext, base, ca)
            counts["let-binder"] += len(bm)
            marks.extend(bm)
        if with_scopes:
            sm = detect_scope_manifest(ftext, base, f"arxiv-{paper}", nw, ca)
            counts["scope"] = counts.get("scope", 0) + len(sm)
            marks.extend(sm)
        for start, end, delim, body in sweep.math_spans(ftext):
            body_off = start + len(delim)
            for m in CSEQ_RE.finditer(body):
                cs = m.group(1) or m.group(2)
                cls = sweep.classify_cseq(cs, macros, roles, plain)
                concept = None
                if cls["class"] == "UNKNOWN":
                    kind = "unknown"
                elif cls["role"] == "UNKNOWN":
                    kind = "role-gap"
                    # concept-typing fold: resolve the role-gap against the
                    # authority. Guard single-char surfaces (\C->"c" junk).
                    if ca is not None and len(cs.lstrip("\\")) >= 2:
                        hit = ca.resolve(cs)
                        if hit:
                            kind = "concept-typed"
                            concept = f"{hit.get('term')} [{hit.get('target')}]"
                else:
                    kind = "classified"
                counts[kind] += 1
                g = base + body_off + m.start()
                tip = (f"\\{cs} · {cls['class']} · {cls['role']}"
                       + (f" · {cls.get('source','')}" if cls.get("source") else ""))
                if concept:
                    tip += f" · concept: {concept}"
                marks.append({
                    "start": g,
                    "end": g + (m.end() - m.start()),
                    "layer": "dp",
                    "kind": kind,
                    "tip": tip,
                })
    return {"paper": f"{paper}-dp", "text": text, "marks": marks, "_counts": counts}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: dp_paper_view.py <paper-id>")
        return 2
    paper = argv[0]
    with_ca = "--with-concept-authority" in argv[1:]
    with_binders = "--with-binders" in argv[1:]
    with_scopes = "--with-scopes" in argv[1:]
    data = build(paper, with_ca=with_ca, with_binders=with_binders,
                 with_scopes=with_scopes)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / f"fable-{paper}-dp-emacs.json"
    out.write_text(json.dumps({k: v for k, v in data.items() if k != "_counts"}))
    c = data["_counts"]
    tot = sum(c.values()) or 1
    print(f"{paper}: {len(data['marks'])} marks — "
          f"classified {c['classified']} ({100*c['classified']//tot}%), "
          f"role-gap {c['role-gap']} ({100*c['role-gap']//tot}%), "
          f"concept-typed {c['concept-typed']} ({100*c['concept-typed']//tot}%), "
          f"unknown {c['unknown']} ({100*c['unknown']//tot}%), "
          f"let-binders {c['let-binder']}, scopes {c.get('scope', 0)}")
    print(f"wrote {out}  →  M-x paper-anatomy-open  RET  {paper}-dp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
