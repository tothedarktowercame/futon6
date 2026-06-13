"""Math-envelope token helpers for DP paper views."""

from __future__ import annotations

import re

CSEQ_RE = re.compile(r"\\([A-Za-z@]+)|\\([^A-Za-z\s])")
# "Let $X$ be a <concept> ..." — the most regular binder in mathematics
# (W2: was dark). Subject = the $-symbol; concept = the noun phrase to the
# first clause boundary. Also catches "and $Y$ a <concept>" conjuncts.
DISPLAY_RE = re.compile(
    r"\\begin\{(equation|eqnarray|align|displaymath|gather|multline)\*?\}"
    r"(.*?)\\end\{\1\*?\}|\\\[(.*?)\\\]", re.S)

LENGTH_UNITS = {"cm", "mm", "pt", "pc", "in", "ex", "em", "bp", "dd", "sp", "mu"}
# text-mode commands whose brace argument is genuine TEXT MODE — prose, not math
# symbols. \stackrel is DELIBERATELY EXCLUDED: TeX sets its above-argument in
# MATH mode, and it is routinely a real morphism/arrow label (\stackrel{S}{\to},
# \stackrel{f}{\to}) — blanket-classifying it leaks real symbols, failing the
# precision gate. Its textual labels (\stackrel{nat.}{=}) are left as honest
# residue for a future, more-surgical morphism-label capability.
TEXTMODE_CMD = re.compile(
    r"\\(?:mbox|hbox|text|textrm|textbf|textit|textsf|texttt)\s*\{")


def _textmode_regions(body):
    """Inner char-spans of text-mode command arguments in a math body, brace
    matched. A letter-run inside one is prose/label text, not a math symbol."""
    regions = []
    for m in TEXTMODE_CMD.finditer(body):
        i, depth = m.end(), 1
        while i < len(body) and depth:
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
            i += 1
        regions.append((m.end(), i - 1))  # inner span, exclusive of close brace
    return regions


_SCRIPT_BASE_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*$")


def script_base_grounding(body, pos, ground_fn, base_off):
    """Ground a sub/superscript letter-run via its base symbol (claude-4).

    A script is PART of its base's compound symbol — the ``coH`` in ``M^{coH}``,
    the ``i`` in ``X_i``, the ``op`` in ``A^{op}`` are modifiers of M / X / A,
    not independent variables. So when the base bare-symbol is ALREADY grounded,
    the script grounds to it too; this only flips ungrounded->grounded (it adds
    no marks, shrinks no denominator) and attacks the C-SYM-GROUND tail.

    BODY[pos:] is the candidate script run; GROUND_FN(sym, global_pos)->label|None
    is build()'s grounder; BASE_OFF is the global offset of BODY[0]. Returns the
    base's binding label (so the script inherits it) or None. Conservative by
    design: the script must sit immediately after ``^``/``_`` (optionally just
    inside one ``{``), and the base must be a bare letter-run (a control-sequence
    base like ``\\alpha`` is not in the binder table, so it is left as residue).
    """
    i = pos - 1
    if i >= 0 and body[i] == "{":
        i -= 1
    if i < 0 or body[i] not in "^_":
        return None
    op = body[i]
    base_end = i  # the base symbol ends immediately before the ^ / _
    m = _SCRIPT_BASE_RE.search(body[:base_end])
    if not m:
        return None
    base = m.group(1)
    base_label = ground_fn(base, base_off + m.start(1))
    if not base_label:
        return None
    return f"{base}{op}-script : {base_label}"


def _nonsym_kind(body, pos, tok, tm_regions):
    """Classify a letter-run at BODY[pos:pos+len(tok)] as a NON-math token, else
    None. A CONTEXT test, not a symbol denylist:
      - 'layout': an env-name right after \\begin{ / \\end{ , or a TeX length
        unit immediately preceded by a digit (\\hspace{-0,4cm} -> 'cm').
      - 'text-mode': a run inside a \\mbox/\\text/\\textrm text argument.
    """
    pre = body[:pos]
    if pre.endswith("\\begin{") or pre.endswith("\\end{"):
        return "layout"
    if tok in LENGTH_UNITS and pos > 0 and body[pos - 1].isdigit():
        return "layout"
    end = pos + len(tok)
    for s, e in tm_regions:
        if s <= pos and end <= e:
            return "text-mode"
    return None
