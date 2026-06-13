"""Reference graph harvesting for DP paper views."""

from __future__ import annotations

import re

# invariants never see it (zero regression by construction). Pairs with the
# informal-proof-move layer: a hedge + the \eqref it defers its verification to.
# REF alternation lists "ref" LAST so \eqref/\autoref/\cref win their prefix.
# We match only the command + opening brace, then brace-BALANCE the argument:
# real keys in this corpus carry nested braces (e.g. \label{t_{M,V}}), which a
# naive \{([^}]*)\} truncates at the first '}' — falsely splitting the key and
# reporting in-paper refs as dangling (verified on 0809.2517).
LABEL_CMD = re.compile(r"\\label\s*\{")
REF_CMD = re.compile(r"\\(eqref|autoref|cref|Cref|vref|pageref|ref)\s*\{")
CITE_CMD = re.compile(r"\\(cite[a-zA-Z]*)\s*(?:\[[^\]]*\])?\{")
_ANCHOR_BEGIN_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
_ANCHOR_SECTION_RE = re.compile(r"\\((?:sub)*section|paragraph)\*?\{")


def _braced_arg(text, i):
    """text[i] must be '{'. Return (inner, end) where INNER is the
    brace-balanced content and END is the index just past the closing '}'.
    Returns (None, i) if the braces never balance (truncated source)."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    return None, i


def _split_top_commas(s):
    """Split S on commas at brace-depth 0, so a key like t_{M,V} stays whole."""
    out, depth, cur = [], 0, []
    for c in s:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def _harvest_labels(ftext):
    """[(start, end, key)] for each \\label, keys brace-balanced."""
    out = []
    for m in LABEL_CMD.finditer(ftext):
        inner, end = _braced_arg(ftext, m.end() - 1)
        if inner is not None:
            out.append((m.start(), end, inner.strip()))
    return out


def _anchor_index(ftext):
    """Sorted [(start, typename)] of constructs a \\label can name — every
    \\begin{env} and every sectioning command — so each label is typed by the
    nearest one preceding it (a label names the thing it sits inside)."""
    idx = [(m.start(), m.group(1)) for m in _ANCHOR_BEGIN_RE.finditer(ftext)]
    idx += [(m.start(), m.group(1)) for m in _ANCHOR_SECTION_RE.finditer(ftext)]
    idx.sort()
    return idx


def _anchor_for(idx, pos):
    """typename of the nearest anchor strictly before POS (None if none)."""
    name = None
    for start, typename in idx:
        if start < pos:
            name = typename
        else:
            break
    return name


def detect_references(ftext, base, label_keys):
    """Harvest \\label / \\ref|\\eqref|... / \\cite into reference-graph marks.

    label_keys is the set of EVERY \\label key in the whole paper, so a \\ref to
    a forward label (declared in a later section/file) still resolves in-paper.
    """
    out = []
    anchors = _anchor_index(ftext)
    for s, e, key in _harvest_labels(ftext):
        names = _anchor_for(anchors, s)
        out.append({
            "start": base + s, "end": base + e,
            "layer": "dp", "kind": "label",
            "tip": f"label {key}" + (f" · names {names}" if names else ""),
            "fields": [["label", key], ["names", names or "—"]],
        })
    for m in REF_CMD.finditer(ftext):
        inner, end = _braced_arg(ftext, m.end() - 1)
        if inner is None:
            continue
        cmd, raw = m.group(1), inner.strip()
        keys = _split_top_commas(raw)
        resolved = bool(keys) and all(k in label_keys for k in keys)
        out.append({
            "start": base + m.start(), "end": base + end,
            "layer": "dp", "kind": "ref",
            "tip": f"\\{cmd} → {raw}"
                   + ("" if resolved else " · DANGLING (no matching \\label)"),
            "fields": [["ref", raw], ["via", f"\\{cmd}"],
                       ["target", "in-paper" if resolved else "dangling"]],
        })
    for m in CITE_CMD.finditer(ftext):
        inner, end = _braced_arg(ftext, m.end() - 1)
        if inner is None:
            continue
        cmd, keys = m.group(1), inner.strip()
        out.append({
            "start": base + m.start(), "end": base + end,
            "layer": "dp", "kind": "cite",
            "tip": f"\\{cmd} → {keys} (bibliography)",
            "fields": [["cite", keys], ["via", f"\\{cmd}"],
                       ["target", "bibliography"]],
        })
    return out
