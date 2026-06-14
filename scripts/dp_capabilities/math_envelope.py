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
    r"\\(?:mbox|hbox|text|textrm|textbf|textit|textsf|texttt|textnormal|textsc"
    r"|textup|textmd|textsl|intertext|shortintertext|emph|caption|footnote)\s*\{")

# \mathrm/\mathnormal/\mathit set their argument upright but in MATH mode — so
# \mathrm{Hom}, \mathrm{dim}, \mathrm{op} are REAL operator/decoration symbols and
# must NOT be excluded. The only safe context discriminator is INTERNAL SPACING:
# an operator name is a single contiguous token, while an inter-formula prose
# spacer carries an explicit spacing macro (\quad, \, …) or run of spaces
# (\mathrm{\quad and \quad}). We classify a \mathrm-arg as text-mode ONLY when it
# contains such a spacer — a context test on construct shape, not word spelling.
MATHRM_CMD = re.compile(r"\\(?:mathrm|mathnormal|mathit)\s*\{")
_SPACER_RE = re.compile(r"\\(?:quad|qquad|,|;|:|!|\s)| {2,}")

# Reference-graph commands whose brace argument is a CITATION/LABEL KEY — never a
# math symbol. When a \label/\ref/\cite sits inside a display-math environment
# (\begin{eqnarray} \label{coc-coass} …), LETTER_RUN otherwise tags the key
# fragments (coc, coass, surjec, …) as symbols. The key is layout, by context.
REF_KEY_CMD = re.compile(
    r"\\(?:label|ref|eqref|cref|Cref|vref|pageref|nameref|autoref|hyperref"
    r"|cite[a-zA-Z]*)\s*(?:\[[^\]]*\])?\{")
# Column-alignment SPEC of an array/tabular — \begin{array}{ll}, {rcl}, {p{..}}:
# pure layout letters (l/c/r/p), not math. The env NAME is already caught by the
# \begin{ guard in _nonsym_kind; this catches the spec ARGUMENT after it.
ARRAY_SPEC_CMD = re.compile(
    r"\\begin\s*\{(?:array|tabular\*?|tabularx|longtable)\}\s*(?:\[[^\]]*\])?\s*\{")


def _brace_inner(body, open_idx):
    """OPEN_IDX indexes a '{'. Return (inner_start, inner_end) of the brace-
    matched argument (inner_end exclusive of the closing '}')."""
    i, depth = open_idx + 1, 1
    while i < len(body) and depth:
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
        i += 1
    return open_idx + 1, i - 1


def _textmode_regions(body):
    """Inner char-spans of text-mode command arguments in a math body, brace
    matched. A letter-run inside one is prose/label text, not a math symbol.
    Includes \\mathrm-as-spacer (\\mathrm{\\quad and \\quad}) — gated on an
    internal spacing macro so real operator names (\\mathrm{dim}) are untouched."""
    regions = [_brace_inner(body, m.end() - 1) for m in TEXTMODE_CMD.finditer(body)]
    for m in MATHRM_CMD.finditer(body):
        s, e = _brace_inner(body, m.end() - 1)
        if _SPACER_RE.search(body[s:e]):
            regions.append((s, e))
    return regions


def _layout_regions(body):
    """Inner char-spans of LAYOUT command arguments in a math body: reference
    keys (\\label/\\ref/\\cite …) and array/tabular column specs. A letter-run
    inside one is a key or alignment letter, not a math symbol."""
    regions = [_brace_inner(body, m.end() - 1) for m in REF_KEY_CMD.finditer(body)]
    regions += [_brace_inner(body, m.end() - 1) for m in ARRAY_SPEC_CMD.finditer(body)]
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


# R6 — display-defined (:=) grounding (claude-3). A bare symbol introduced as the
# WHOLE left-hand side of a ":=" definition ("$X := ...$", "$$L := ...$$") IS that
# definition; its later uses inherit it. KEY precision (claude-1's gate): X must be
# a STANDALONE definiendum — preceded by a math delimiter / hard separator, never a
# binary operator or accent — so "a \cdot l :=" (l is a factor), "X_i :=" (i is a
# subscript), and GrCalc diagram LHS are NOT harvested. Like script_base_grounding,
# this only flips ungrounded->grounded at the ground() seam: no marks, no denom.
_ASSIGN_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*:=")
_ASSIGN_SEP = re.compile(r"(?:\$|&|\\\\|\\quad|\\qquad|\\,|\\;|\\:|\[|^)\s*$")


def harvest_display_assigns(text):
    """Map each standalone ':='-definiendum bare symbol to its sorted global
    definition offsets (X-:=-grounds-X). The _ASSIGN_SEP guard requires X to be
    the whole LHS, so a factor/subscript/diagram-internal letter is never taken."""
    defs = {}
    for m in _ASSIGN_RE.finditer(text):
        if _ASSIGN_SEP.search(text[:m.start(1)]):
            defs.setdefault(m.group(1), []).append(m.start(1))
    for k in defs:
        defs[k].sort()
    return defs


def display_assign_grounding(sym, gpos, assign_defs):
    """Fallback grounder (R6): if SYM is display-defined by 'SYM := ...' at or
    before GPOS, ground SYM to that definition. Forward-only (a use before the
    definition is not yet grounded), mirroring ground()'s at-or-before scope."""
    ps = assign_defs.get(sym)
    if ps and ps[0] <= gpos:
        return f"display-defined (:=) : {sym}"
    return None


def _nonsym_kind(body, pos, tok, tm_regions, layout_regions=()):
    """Classify a letter-run at BODY[pos:pos+len(tok)] as a NON-math token, else
    None. A CONTEXT test (where it sits), not a symbol denylist:
      - 'layout': an env-name right after \\begin{ / \\end{ , a TeX length unit
        immediately preceded by a digit (\\hspace{-0,4cm} -> 'cm'), a reference
        key inside \\label/\\ref/\\cite, or an array/tabular column-spec letter.
      - 'text-mode': a run inside a \\mbox/\\text/\\textrm/\\intertext text
        argument, or a \\mathrm-as-spacer (\\mathrm{\\quad and \\quad}).
    """
    pre = body[:pos]
    if pre.endswith("\\begin{") or pre.endswith("\\end{"):
        return "layout"
    if tok in LENGTH_UNITS and pos > 0 and body[pos - 1].isdigit():
        return "layout"
    end = pos + len(tok)
    for s, e in layout_regions:
        if s <= pos and end <= e:
            return "layout"
    for s, e in tm_regions:
        if s <= pos and end <= e:
            return "text-mode"
    return None
