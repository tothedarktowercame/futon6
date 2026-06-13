"""Scope-manifest capability for DP paper views."""

from __future__ import annotations

from .binders import _concept_head

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
