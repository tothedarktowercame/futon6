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
from dp_capabilities.binders import (
    BINDER_RE,
    CONJUNCT_RE,
    DEFINE_RES,
    IS_RE,
    QUANT_RES,
    WHERE_RES,
    _concept_head,
    _load_xref,
    detect_binders,
)
from dp_capabilities.math_envelope import (
    CSEQ_RE,
    DISPLAY_RE,
    _nonsym_kind,
    _textmode_regions,
    display_assign_grounding,
    harvest_display_assigns,
    script_base_grounding,
)
from dp_capabilities.proof_moves import detect_proof_moves
from dp_capabilities.references import _harvest_labels, detect_references
from dp_capabilities.scopes import detect_scope_manifest
from dp_capabilities.wellformed import (
    BINDER_KINDS,
    _clamp_structural_sentence_markers,
    _reconcile_structural_crossings,
    _snap_marks_to_math_atoms,
)

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
    # R6 (claude-3): bare symbols display-defined by "X := ..." ground to that
    # definition. Harvested once over the whole text (global offsets), consumed
    # as a fallback at the ground() seam below.
    assign_defs = harvest_display_assigns(text)

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
            # (1) Let-binders + "$M$ is a right $A$-module" (symbol+phrase)
            pairs = []
            for m in BINDER_RE.finditer(f["text"]):
                pairs.append((m.group(1), m.group(2), m.start()))
                for cm in CONJUNCT_RE.finditer(f["text"], m.end(), m.end() + 160):
                    pairs.append((cm.group(1), cm.group(2), cm.start()))
            for m in IS_RE.finditer(f["text"]):
                pairs.append((m.group(1), m.group(2), m.start()))
            # defined-in-paper: definition-prose definienda (define/denote/write)
            for rx in DEFINE_RES:
                for m in rx.finditer(f["text"]):
                    pairs.append((m.group(1), m.group(2), m.start()))
            # quantifier + where/with binding (Joe's QUANTIFIER/POSITIONAL tail)
            for rx, qlabel in QUANT_RES:
                for m in rx.finditer(f["text"]):
                    pairs.append((m.group(1), qlabel, m.start()))
            for rx in WHERE_RES:
                for m in rx.finditer(f["text"]):
                    pairs.append((m.group(1), m.group(2), m.start()))
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
                        # assume/if: the scope match is often just "If $M$" —
                        # pull the type from the FILE TEXT at the scope position,
                        # not the truncated match.
                        raw = s.get("hx/content", {}).get("position") or 0
                        mm = re.search(r"\bis\s+(?:a|an|the)\s+([^.,;]+)",
                                       f["text"][raw:raw + 120])
                        typ = mm.group(1) if mm else None
                    if sym and typ:
                        _add_binding(sym, typ, pos)
        for k in bindings:
            bindings[k].sort()

    def ground(sym, g):
        """Latest binding of SYM at-or-before global offset G (its scope)."""
        cand = [(p, lab) for p, lab in bindings.get(sym, []) if p <= g]
        return cand[-1][1] if cand else None

    # REFERENCE GRAPH pre-pass: collect every \label key across the WHOLE paper
    # first, so a \ref to a forward label (later section/file) still resolves.
    label_keys = set()
    for f in tex_files:
        for _s, _e, key in _harvest_labels(f["text"]):
            label_keys.add(key)

    for f in tex_files:
        base = bases[f["file"]]
        ftext = f["text"]
        bm = []
        sm = []
        if with_binders:
            bm = detect_binders(ftext, base, ca, xref=xref)
        if with_scopes:
            sm = detect_scope_manifest(ftext, base, f"arxiv-{paper}", nw, ca)
        if bm or sm:
            # SNAP (Joe): math spans are atomic. Generalize the old
            # scope-manifest-only pass to binder marks too, then reconcile the
            # two structural layers so manifest-vs-binder scopes never cross.
            spans = [(base + s, base + e) for s, e, _d, _b in sweep.math_spans(ftext)]
            spans += [(base + dm.start(), base + dm.end()) for dm in DISPLAY_RE.finditer(ftext)]
            structural = _snap_marks_to_math_atoms(bm + sm, spans)
            structural = _clamp_structural_sentence_markers(structural, text)
            structural = _snap_marks_to_math_atoms(structural, spans)
            structural = _reconcile_structural_crossings(structural)
            counts["let-binder"] += sum(1 for m in structural
                                         if m.get("layer") == "dp"
                                         and m.get("kind") in BINDER_KINDS)
            counts["scope"] = counts.get("scope", 0) + sum(
                1 for m in structural if m.get("layer") == "scope")
            marks.extend(structural)
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
            tm_regions = _textmode_regions(dbody)
            for sm in re.finditer(r"(?<!\\)(?<![A-Za-z])[A-Za-z][A-Za-z0-9]*", dbody):
                sym = sm.group(0)
                if sym in ("got", "gcl", "gbeg", "gend", "gnl", "gvac", "gob",
                           "grm", "gmu", "gbr", "gcn", "grcm", "gcmu", "hspace",
                           "scalebox", "ot", "label"):
                    continue  # GrCalc / layout macro names, not variables
                g = base + dm.start() + (dm.start(2) - dm.start() if dm.group(2) else
                                         dm.start(3) - dm.start()) + sm.start()
                nonsym = _nonsym_kind(dbody, sm.start(), sym, tm_regions)
                if nonsym is not None:
                    counts[nonsym] = counts.get(nonsym, 0) + 1
                    marks.append({
                        "start": g, "end": g + len(sym), "layer": "dp",
                        "kind": nonsym,
                        "tip": f"{nonsym}: {sym} (non-math token, excluded)"})
                    continue
                bound = ground(sym, g)
                k = "symbol-grounded" if bound else "symbol"
                counts[k] = counts.get(k, 0) + 1
                mk = {"start": g, "end": g + len(sym), "layer": "dp", "kind": k,
                      "tip": (f"{sym} : {bound}" if bound else f"symbol {sym} (in display)")}
                if bound:
                    mk["fields"] = [["symbol", sym], ["bound", bound]]
                marks.append(mk)
        # INFORMAL PROOF MOVES (Joe): the proof's discourse gestures — "it is
        # not difficult to check" and kin — asserting a step while declining to
        # carry it out. A layer distinct from the structural scopes.
        for pmark in detect_proof_moves(ftext, base):
            counts["proof-move"] = counts.get("proof-move", 0) + 1
            marks.append(pmark)
        # REFERENCE GRAPH (Joe): \label/\ref/\cite harvest — the in-paper
        # reference graph + outbound citations. Unconditional, like proof-moves;
        # lives in prose, so the coverage/well-formedness invariants never see it.
        for rmark in detect_references(ftext, base, label_keys):
            counts[rmark["kind"]] = counts.get(rmark["kind"], 0) + 1
            if rmark["kind"] == "ref" and any(
                    k == "target" and v == "dangling" for k, v in rmark["fields"]):
                counts["ref-dangling"] = counts.get("ref-dangling", 0) + 1
            marks.append(rmark)
        for start, end, delim, body in sweep.math_spans(ftext):
            body_off = start + len(delim)
            # RULE (Joe): anything between dollar signs IS a math scope —
            # never null. Even a bare $H$ (no control sequence) gets an
            # envelope; LaTeX itself is telling us it's mathematics.
            counts["math"] = counts.get("math", 0) + 1
            marks.append({
                "start": base + start, "end": base + end,
                "layer": "dp", "kind": "math",
                "tip": f"math: {(body.strip() or 'empty math span')[:60]}",
                "fields": [["math", (body.strip() or "— empty span —")[:70]],
                           ["delim", "display ($$)" if delim == "$$" else "inline ($)"]],
            })
            # SATIETY (Joe): the $...$ scope is hungry for content — annotate
            # the symbols inside, not just control sequences. A bare $H$ whose
            # H is unmarked is a hungry envelope (a violation). Mark each
            # symbol (letter/identifier run not part of a \cseq name).
            tm_regions = _textmode_regions(body)
            for sm in re.finditer(r"[A-Za-z][A-Za-z0-9]*", body):
                if sm.start() > 0 and body[sm.start() - 1] == "\\":
                    continue  # it's a control-sequence name, handled below
                sym = sm.group(0)
                g = base + body_off + sm.start()
                # NON-MATH token (length unit / env-name / text-mode prose)? Tag
                # it so the checker excludes it from the symbol denominator —
                # never could be grounded, so it is not a symbol (claude-3).
                nonsym = _nonsym_kind(body, sm.start(), sym, tm_regions)
                if nonsym is not None:
                    counts[nonsym] = counts.get(nonsym, 0) + 1
                    marks.append({
                        "start": g, "end": g + len(sym), "layer": "dp",
                        "kind": nonsym,
                        "tip": f"{nonsym}: {sym} (non-math token, excluded)"})
                    continue
                bound = ground(sym, g)
                if not bound:
                    # sub/superscript of a grounded base grounds to it (claude-4)
                    bound = script_base_grounding(body, sm.start(), ground,
                                                  base + body_off)
                if not bound:
                    # symbol display-defined by "X := ..." grounds to it (R6, claude-3)
                    bound = display_assign_grounding(sym, g, assign_defs)
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
    print(f"  reference graph: {c.get('label', 0)} labels, "
          f"{c.get('ref', 0)} refs ({c.get('ref-dangling', 0)} dangling), "
          f"{c.get('cite', 0)} cites")
    print(f"wrote {out}  →  M-x paper-anatomy-open  RET  {paper}-dp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
