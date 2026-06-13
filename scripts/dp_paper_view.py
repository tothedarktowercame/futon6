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
DISPLAY_RE = re.compile(
    r"\\begin\{(equation|eqnarray|align|displaymath|gather|multline)\*?\}"
    r"(.*?)\\end\{\1\*?\}|\\\[(.*?)\\\]", re.S)
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
            # a binder/constraint/quantifier scope must not (a) cross a
            # sentence boundary, (b) run into a display equation, or (c)
            # exceed a sane length — else it becomes the huge nonsemantic
            # blob Joe flagged (a 414-char constrain/relation across a
            # GrCalc display). Clamp to the earliest of all three.
            limits = [end, pos + 140]
            sent = ftext.find(". ", pos)
            if sent != -1:
                limits.append(sent)
            for delim in (r"\begin{", r"\[", "$$"):
                d = ftext.find(delim, pos + 1)
                if d != -1:
                    limits.append(d)
            end = min(limits)
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
        # SUPPRESS the compound-noun false relation (Joe): "$A$-module" is a
        # typed noun, not a relation between $A$ and "module". The detector
        # latches the "-module" suffix as a relation head (text begins with a
        # bare hyphen) and runs through the following prose — that's the purple
        # blob meeting the blue assume inside the compound. A real relation
        # symbol (=, ⊆, →, "is a") never begins with "-", so drop these.
        if stype == "constrain/relation" and any(
                r == "relation" and str(v).lstrip().startswith("-") for r, v in fields):
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
    # PRE-PASS (Joe's grounding point): the "Let $H$ be a Hopf algebra"
    # stanza grounds the symbol H. Collect symbol → (global-pos, concept)
    # bindings so each later symbol occurrence resolves to its binder (R4,
    # the use→binder edge) instead of leaving an ungrounded "symbol" debt.
    bindings = {}  # bare symbol -> sorted [(global_pos, label)]

    def _add_binding(sym_text, label, gpos):
        s = re.search(r"[A-Za-z]+", sym_text or "")
        if not s or not label:
            return
        bindings.setdefault(s.group(0), []).append((gpos, label.strip()[:50]))

    if with_binders or with_scopes:
        for f in tex_files:
            base = bases[f["file"]]
            # (1) Let-binders (cleanest symbol+phrase)
            for m in BINDER_RE.finditer(f["text"]):
                pairs = [(m.group(1), m.group(2), m.start())]
                for cm in CONJUNCT_RE.finditer(f["text"], m.end(), m.end() + 160):
                    pairs.append((cm.group(1), cm.group(2), cm.start()))
                for subj, phrase, pos in pairs:
                    label = phrase
                    if ca is not None:
                        hit = ca.resolve(_concept_head(phrase))
                        if hit:
                            label = hit.get("term")
                    _add_binding(subj, label, base + pos)
            # (2) the FULL scope manifest: every binding-like scope grounds its
            # symbol — bind/typed, assume ("If $M$ is a right $A$-module"),
            # quant, where-binding — not just Let (Joe: the 4029 ungrounded are
            # mostly bound by non-Let forms the grounding never consulted).
            if nw is not None:
                for s in nw.detect_scopes(f"arxiv-{paper}", f["text"]):
                    pos = base + (s.get("hx/content", {}).get("position") or 0)
                    ends = s.get("hx/ends", [])
                    sym = next((e.get("latex") or e.get("text") for e in ends
                                if e.get("role") in ("symbol", "condition")), None)
                    typ = next((e.get("text") or e.get("latex") for e in ends
                                if e.get("role") in ("type", "relation", "value")), None)
                    if sym and not typ:
                        # assume/if: pull the type from the match phrase
                        mm = re.search(r"\bis\s+(?:a|an|the)\s+([^.,;]+)",
                                       s.get("hx/content", {}).get("match", ""))
                        typ = mm.group(1) if mm else None
                    if sym and typ:
                        _add_binding(sym, typ, pos)
        for k in bindings:
            bindings[k].sort()

    def ground(sym, g):
        """Latest binding of SYM at-or-before global offset G (its scope)."""
        cand = [(p, lab) for p, lab in bindings.get(sym, []) if p <= g]
        return cand[-1][1] if cand else None

    for f in tex_files:
        base = bases[f["file"]]
        ftext = f["text"]
        if with_binders:
            bm = detect_binders(ftext, base, ca, xref=xref)
            counts["let-binder"] += len(bm)
            marks.extend(bm)
        if with_scopes:
            sm = detect_scope_manifest(ftext, base, f"arxiv-{paper}", nw, ca)
            # SNAP (Joe): a math span $...$ is atomic — NO scope boundary may
            # fall strictly inside one. A start inside a span snaps out to the
            # span's end; an end inside a span snaps back to its start. (The
            # codiagonal mess was a constrain/relation that began on the closing
            # $ of $A$, so blue+purple met inside $...$.) Display envs too.
            spans = [(base + s, base + e) for s, e, _d, _b in sweep.math_spans(ftext)]
            spans += [(base + dm.start(), base + dm.end()) for dm in DISPLAY_RE.finditer(ftext)]
            for m in sm:
                for s, e in spans:
                    if s < m["start"] < e:
                        m["start"] = e
                    if s < m["end"] < e:
                        m["end"] = s
            sm = [m for m in sm if m["end"] > m["start"]]
            counts["scope"] = counts.get("scope", 0) + len(sm)
            marks.extend(sm)
        # DISPLAY EQUATIONS (Joe): \begin{eqnarray}/\[...\] are math scopes
        # too — math_spans only sees $-delims. Don't parse the GrCalc layout
        # this pass; DO mark the whole thing a display-math scope and surface
        # the variables it relates (+ := => a definition). "Going inside" the
        # mess: at minimum, this is a display equation relating these symbols.
        for dm in DISPLAY_RE.finditer(ftext):
            envname = dm.group(1) or "\\[ \\]"
            dbody = dm.group(2) or dm.group(3) or ""
            counts["display"] = counts.get("display", 0) + 1
            # PRINCIPLE (Joe): a display environment IS a math scope — by virtue
            # of being a math environment, independently of whether we can read
            # its display semantics. Same status as $...$ (R1), so it's emitted
            # UNCONDITIONALLY. The definition/relating-variables below is
            # ENRICHMENT that rides on the math scope; it does not constitute it.
            dvars = sorted({s for s in re.findall(r"(?<!\\)(?<![A-Za-z])[A-Za-z]", dbody)
                            if s not in "et"})[:8]  # the cell variables
            is_def = ":=" in dbody or "\\stackrel{def}" in dbody
            tip = f"math scope · display ({envname})"
            if is_def:
                tip += " · DEFINITION (:=)"
            if dvars:
                tip += f" · relates {', '.join(dvars)}"
            marks.append({
                "start": base + dm.start(), "end": base + dm.end(),
                "layer": "dp", "kind": "math",
                "tip": tip,
                "fields": [["scope", f"display math ({envname})"],
                           ["semantics", ("definition (:=)" if is_def else "equation")
                                         if (dvars or is_def) else "unread (math scope only)"],
                           ["relates", ", ".join(dvars) or "—"]],
            })
            # mark + ground the variable symbols inside the display
            for sm in re.finditer(r"(?<!\\)(?<![A-Za-z])[A-Za-z][A-Za-z0-9]*", dbody):
                sym = sm.group(0)
                if sym in ("got", "gcl", "gbeg", "gend", "gnl", "gvac", "gob",
                           "grm", "gmu", "gbr", "gcn", "grcm", "gcmu", "hspace",
                           "scalebox", "ot", "label"):
                    continue  # GrCalc / layout macro names, not variables
                g = base + dm.start() + (dm.start(2) - dm.start() if dm.group(2) else
                                         dm.start(3) - dm.start()) + sm.start()
                bound = ground(sym, g)
                k = "symbol-grounded" if bound else "symbol"
                counts[k] = counts.get(k, 0) + 1
                mk = {"start": g, "end": g + len(sym), "layer": "dp", "kind": k,
                      "tip": (f"{sym} : {bound}" if bound else f"symbol {sym} (in display)")}
                if bound:
                    mk["fields"] = [["symbol", sym], ["bound", bound]]
                marks.append(mk)
        for start, end, delim, body in sweep.math_spans(ftext):
            body_off = start + len(delim)
            # RULE (Joe): anything between dollar signs IS a math scope —
            # never null. Even a bare $H$ (no control sequence) gets an
            # envelope; LaTeX itself is telling us it's mathematics.
            if body.strip():
                counts["math"] = counts.get("math", 0) + 1
                marks.append({
                    "start": base + start, "end": base + end,
                    "layer": "dp", "kind": "math",
                    "tip": f"math: {body.strip()[:60]}",
                    "fields": [["math", body.strip()[:70]],
                               ["delim", "display ($$)" if delim == "$$" else "inline ($)"]],
                })
            # SATIETY (Joe): the $...$ scope is hungry for content — annotate
            # the symbols inside, not just control sequences. A bare $H$ whose
            # H is unmarked is a hungry envelope (a violation). Mark each
            # symbol (letter/identifier run not part of a \cseq name).
            for sm in re.finditer(r"[A-Za-z][A-Za-z0-9]*", body):
                if sm.start() > 0 and body[sm.start() - 1] == "\\":
                    continue  # it's a control-sequence name, handled below
                sym = sm.group(0)
                g = base + body_off + sm.start()
                bound = ground(sym, g)
                kind = "symbol-grounded" if bound else "symbol"
                counts[kind] = counts.get(kind, 0) + 1
                mark = {
                    "start": g, "end": g + len(sym),
                    "layer": "dp", "kind": kind,
                    "tip": (f"{sym} : {bound}" if bound else f"symbol {sym} (ungrounded)"),
                }
                if bound:
                    mark["fields"] = [["symbol", sym], ["bound", bound]]
                marks.append(mark)
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
