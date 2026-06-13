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


def _load_xref():
    """Shuttle cross-ref components: mathlib names, PlanetMath finder."""
    import json as _j
    mathlib_names = []
    mj = Path("/home/joe/code/futon6/data/mathlib-defs-monoidal.json")
    if mj.exists():
        mathlib_names = [d["name"] for d in _j.loads(mj.read_text())]
    pd = _ilu.spec_from_file_location("mpd", Path(__file__).resolve().parent / "mine_prose_def.py")
    mpd = _ilu.module_from_spec(pd); pd.loader.exec_module(mpd)
    return mathlib_names, mpd


def _xref_fields(phrase, head, mathlib_names, mpd, ca):
    """Lean / PlanetMath / nLab cross-references + coverage verdict for a
    definiens — the three-Norn shuttle, surfaced as annotation fields."""
    import re as _re
    cam = "".join(w.capitalize() for w in _re.split(r"[\s-]+", (head or phrase).strip()))
    lean = next((n for n in mathlib_names
                 if cam.lower() in (n.lower(), n.lower().rstrip("_"), n.lower().rstrip("obj"))), None)
    pm = mpd.find_planetmath(head or phrase) if (head or phrase) else None
    pm = pm.name if pm else None
    hit = ca.resolve(head) if (ca and head) else None
    nlab = hit.get("target") if hit else None
    cov = ("fully covered" if (lean and pm) else
           "formalise: prose only" if (pm and not lean) else
           "Lean" if lean else "DEBT: undefined" if not nlab else "pointer only")
    return [["Lean", lean or "–"], ["PlanetMath", pm or "–"],
            ["nLab/NNexus", nlab or "–"], ["coverage", cov]]


def _concept_head(phrase: str) -> str:
    """Last 1-3 words of a concept phrase, math/markup stripped, for lookup."""
    words = re.findall(r"[A-Za-z][A-Za-z-]+", re.sub(r"\$[^$]*\$|[\\{}]", " ", phrase))
    return " ".join(words[-3:]) if words else ""


def detect_binders(ftext, base, ca, xref=None):
    """Emit Let-binder marks with explicit definiendum/definiens structure.
    Per binder, three marks within the blue Let scope:
      - the scope (kind let-binder),
      - the definiendum (the $symbol$; bold, term-indexed),
      - the definiens (the type phrase; underlined, same term-index).
    term-index distinguishes multiple defined terms in one Let sentence
    (Joe: graded gray, <=10/sentence)."""
    out = []
    for m in BINDER_RE.finditer(ftext):
        # primary + conjoined binders share the SENTENCE; index within it.
        binders = [(m.start(1), m.end(1), m.start(2), m.end(2), m.start(), m.end())]
        for cm in CONJUNCT_RE.finditer(ftext, m.end(), m.end() + 160):
            binders.append((cm.start(1), cm.end(1), cm.start(2), cm.end(2),
                            cm.start(), cm.end()))
        for term_i, (ds, de, ps, pe, ss, se) in enumerate(binders):
            subj = ftext[ds:de]
            phrase = ftext[ps:pe].strip()
            concept = None
            if ca is not None:
                hit = ca.resolve(_concept_head(phrase)) if phrase else None
                if hit:
                    concept = f"{hit.get('term')} [{hit.get('target')}]"
            # the blue Let scope
            out.append({
                "start": base + ss, "end": base + se,
                "layer": "dp", "kind": "let-binder",
                "tip": f"binds {subj} : {phrase[:60]}"
                       + (f" · concept: {concept}" if concept else ""),
                "fields": [["binds", subj], ["as", phrase[:70]],
                           ["canon", concept or "— (unresolved)"]],
            })
            # definiendum: the $symbol$ (bold, term-indexed)
            out.append({
                "start": base + ds, "end": base + de,
                "layer": "dp", "kind": "definiendum", "term-index": term_i,
                "tip": f"definiendum #{term_i}: {subj}",
            })
            # definiens: the type phrase (underlined, same term-index),
            # carrying the three-Norn cross-references as annotation fields
            if pe > ps:
                fields = None
                if xref is not None:
                    mathlib_names, mpd = xref
                    fields = _xref_fields(phrase, _concept_head(phrase),
                                          mathlib_names, mpd, ca)
                out.append({
                    "start": base + ps, "end": base + pe,
                    "layer": "dp", "kind": "definiens", "term-index": term_i,
                    "tip": f"definiens #{term_i}: {phrase[:60]}"
                           + (f" · {concept}" if concept else ""),
                    "fields": fields,
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
        # Clamp the overlay extent. Environment scopes (theorem/proof/defn)
        # legitimately span multiple sentences; binder/constraint/quantifier
        # scopes must NOT cross a sentence boundary — the period is English,
        # not mathematics (Joe). Stop before the first ". " after pos.
        if not stype.startswith("env/"):
            sent = ftext.find(". ", pos)
            if sent != -1 and sent < end:
                end = sent
        else:
            end = min(end, pos + 400)  # bounded, but room for a real env
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
        if end <= pos:
            continue
        out.append({
            "start": base + pos, "end": base + end,
            "layer": "scope", "kind": stype,
            "tip": f"{stype} · " + " | ".join(f"{r}:{v}" for r, v in fields[:3]),
            "fields": fields or None,
        })
    return out


def build(paper: str, with_ca: bool = False, with_binders: bool = False,
          with_scopes: bool = False, with_xref: bool = False) -> dict:
    ca = None
    if with_ca or with_binders or with_scopes or with_xref:
        from concept_authority import ConceptAuthority
        ca = ConceptAuthority()
    nw = _load_nlab_wiring() if with_scopes else None
    xref = _load_xref() if with_xref else None
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
            bm = detect_binders(ftext, base, ca, xref=xref)
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
    with_xref = "--with-xref" in argv[1:]
    data = build(paper, with_ca=with_ca, with_binders=with_binders,
                 with_scopes=with_scopes, with_xref=with_xref)
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
