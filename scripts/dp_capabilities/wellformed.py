"""Well-formedness reconciliation helpers for DP paper marks."""

BINDER_KINDS = {"let-binder", "definiendum", "definiens"}


def _snap_marks_to_math_atoms(marks, spans):
    """Move detector boundaries out of atomic math spans.

    This is the SNAP pass generalized beyond scope-manifest marks: binder
    detectors also produce structural extents, and those extents must not split
    a $...$ or display-math atom. A start strictly inside a math span moves to
    that span's end; an end strictly inside moves to that span's start.
    """
    for mark in marks:
        for s, e in spans:
            if s < mark["start"] < e:
                mark["start"] = e
            if s < mark["end"] < e:
                mark["end"] = s
    return [m for m in marks if m["end"] > m["start"]]


def _is_manifest_structural(mark):
    return mark.get("layer") == "scope" and not mark.get("kind", "").startswith("env/")


def _is_binder_scope(mark):
    return mark.get("layer") == "dp" and mark.get("kind") == "let-binder"


def _is_structural_scope(mark):
    return _is_binder_scope(mark) or _is_manifest_structural(mark)


def _crosses(a, b):
    return a["start"] < b["start"] < a["end"] < b["end"]


def _reconcile_structural_crossings(marks):
    """Make structural scopes nest or become disjoint.

    The two detectors often describe the same "Let ... be ..." prose at
    slightly different grains. Preserve the binder as the canonical local
    definition scope and clamp only the crossing manifest boundary to the
    binder edge. This avoids gaming by deletion while making the two structural
    layers comparable. For manifest-vs-manifest crossings, preserve the earlier
    structural read and start the later one at the shared edge. Containing/nested
    scopes remain intact; crossings become adjacent regions.
    """
    structural = [m for m in marks if _is_structural_scope(m)]
    for _ in range(4):
        changed = False
        structural.sort(key=lambda m: (m["start"], -m["end"]))
        for i, a in enumerate(structural):
            if a["end"] <= a["start"]:
                continue
            for b in structural[i + 1:]:
                if b["start"] >= a["end"]:
                    break
                if b["end"] <= b["start"]:
                    continue
                if _crosses(a, b):
                    if _is_binder_scope(b) and not _is_binder_scope(a):
                        a["end"] = b["start"]
                    else:
                        b["start"] = a["end"]
                    changed = True
        if not changed:
            break
    return [m for m in marks if m["end"] > m["start"]]


def _clamp_structural_sentence_markers(marks, text):
    """Keep structural scopes on one checker sentence.

    The checker's sentence invariant is deliberately text-only. That means a
    TeX ellipsis written as `... ` inside a binder's math subject is also seen
    as a sentence boundary. Clamp such scopes before the first `. ` and let the
    math-atom snap/drop pass remove any degenerate residue.
    """
    for mark in marks:
        if not _is_structural_scope(mark):
            continue
        i = text.find(". ", mark["start"], mark["end"])
        if i != -1:
            mark["end"] = i
    return [m for m in marks
            if m["end"] > m["start"]
            and text[m["start"]:m["end"]].strip() not in {"Let", "For", "If", "Assume"}]


# NON-SYMBOL TOKENS (claude-3): LETTER_RUN catches letter-runs inside $$ displays
# that are NOT math symbols and can never be grounded — TeX length units, env
# names, and text-mode prose. The checker (ff9099e) excludes any run COVERED by a
# kind 'layout'/'text-mode' mark from the symbol denominator. We classify them
