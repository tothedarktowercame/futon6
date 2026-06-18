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


# ALL extent-bearing scopes (not just the structural ones) must nest cleanly.
# Authority rank: a genuine \begin..\end environment is authoritative and never
# moves; manifest/binder scopes next; the Let–Then implication; then the
# heuristic IATC claim, which yields first. On a crossing, the lower-rank scope's
# boundary is clamped to the higher one's edge (nest -> stays; cross -> becomes
# adjacent). Equal rank: the later scope yields (mirrors the structural pass).
def _scope_rank(m):
    k = m.get("kind", "")
    if k.startswith("env/"):
        return 4
    if m.get("layer") == "scope" or k == "let-binder" or k.startswith("bind/"):
        return 3
    if k == "implies":
        return 2
    if k == "claim":
        return 1
    return None  # not a scope (math/symbol/cite/inference edge/...)


def scope_crossings(marks):
    """Pairs (a, b) of extent scopes that partially overlap (cross, not nest)."""
    sc = sorted((m for m in marks if _scope_rank(m) is not None),
                key=lambda m: (m["start"], -m["end"]))
    out = []
    for i, a in enumerate(sc):
        for b in sc[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if _crosses(a, b):
                out.append((a, b))
    return out


def reconcile_all_scopes(marks):
    """Clamp crossings among ALL extent scopes so they nest or are disjoint."""
    sc = [m for m in marks if _scope_rank(m) is not None]
    for _ in range(6):
        changed = False
        sc.sort(key=lambda m: (m["start"], -m["end"]))
        for i, a in enumerate(sc):
            if a["end"] <= a["start"]:
                continue
            for b in sc[i + 1:]:
                if b["start"] >= a["end"]:
                    break
                if b["end"] <= b["start"] or not _crosses(a, b):
                    continue
                ra, rb = _scope_rank(a), _scope_rank(b)
                if ra > rb:        # b lower authority -> start it at a's edge
                    b["start"] = a["end"]
                else:              # a lower or equal -> end it at b's edge
                    a["end"] = b["start"]
                changed = True
        if not changed:
            break
    return [m for m in marks
            if _scope_rank(m) is None or m["end"] > m["start"]]


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
