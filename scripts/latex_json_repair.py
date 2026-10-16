#!/usr/bin/env python3
"""Restore LaTeX commands that model JSON delivered as control characters.

Two different faults reach the parser, and both damage a backslash command in
a JSON string:

RAW: the serving stack emits a control byte in place of a backslash and the
first letters of a command, which leaves the JSON invalid. In the APM run
(apm-501-xi-r1, 2026-10-16) the saved S4 answers read, with TAB written as ⇥:
``⇥sto`` (\\mapsto), ``⇥athbb`` (\\mathbb), ``⇥ightarrow`` (\\rightarrow),
``⇥ph{dense}`` (\\emph), ``⇥chi`` (\\chi), ``⇥imes`` (\\times), ``⇥o`` (\\to),
and BACKSPACE + ``bZ`` (\\bbZ, the paper's own macro). From zero to three
letters go with the backslash, so the command has to be recovered from the
letters that are left.

DECODED: the model writes a JSON escape where a LaTeX command belongs (``"\\theta"``
in the JSON text), which parses, but as TAB + ``heta``. In this case the
control character stands for exactly backslash plus one letter (t, b, f, n, r).

A repair is made only when the restored name is a known command: a standard
one (COMMANDS below) or one that the source passage itself uses, which brings
in a paper's own macros. Where several commands fit, the choice prefers, in
order: fewer dropped letters, a command the source uses, a dropped part that
starts with the escape's own letter, and a lower-case name. The other
candidates go into the repair record. Anything else is left alone for the
parser or the stage's checks to refuse.

A decoded newline or carriage return is treated as a command only inside
mathematics ($...$, \\(...\\)), raw or decoded, because in prose a line
break before a word is ordinary text ("\\nin the proof" is not \\in). Other
raw control bytes are never valid inside a JSON string, so they are repaired
anywhere in it.
"""
from __future__ import annotations

import json
import re

ESCAPE_LETTER = {"\t": "t", "\b": "b", "\f": "f", "\n": "n", "\r": "r"}
RAW_CONTROLS = "\t\b\f\n\r\v"
MAX_DROPPED = 3

COMMANDS = frozenset("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu
xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
times cdot circ bullet star ast oplus ominus otimes oslash odot wedge vee cap cup sqcup sqcap
setminus pm mp div amalg uplus wr diamond
leq geq neq le ge ne ll gg prec succ preceq succeq sim simeq approx cong equiv propto asymp
subset supset subseteq supseteq subsetneq supsetneq nsubseteq sqsubseteq sqsupseteq
in notin ni owns perp parallel mid nmid models vdash dashv
to gets mapsto longmapsto rightarrow leftarrow leftrightarrow Rightarrow Leftarrow
Leftrightarrow longrightarrow longleftarrow Longrightarrow Longleftarrow iff implies impliedby
uparrow downarrow updownarrow Uparrow Downarrow hookrightarrow hookleftarrow
twoheadrightarrow rightharpoonup leftharpoonup rightleftharpoons nearrow searrow nwarrow swarrow
infty nabla partial forall exists nexists emptyset varnothing neg lnot top bot angle
ell hbar Re Im aleph wp prime backslash triangle Box Diamond flat natural sharp
sum prod coprod int iint iiint oint bigcup bigcap bigoplus bigotimes bigvee bigwedge bigsqcup
lim liminf limsup sup inf max min arg det dim ker deg gcd hom exp log ln lg sin cos tan cot
sec csc arcsin arccos arctan sinh cosh tanh coth Pr
frac dfrac tfrac binom dbinom tbinom sqrt overline underline widehat widetilde hat tilde bar
vec dot ddot check breve acute grave mathring overbrace underbrace overrightarrow overleftarrow
mathbb mathbf mathcal mathfrak mathrm mathit mathsf mathtt mathscr boldsymbol bm operatorname
text textbf textit textrm texttt textsf emph mbox hbox
left right big Big bigg Bigg bigl bigr Bigl Bigr middle
langle rangle lfloor rfloor lceil rceil lvert rvert lVert rVert vert Vert
ldots cdots vdots ddots dots dotsc dotsb quad qquad
not stackrel overset underset substack limits nolimits displaystyle textstyle
label ref eqref cite
""".split())


def _in_math(text: str, i: int) -> bool:
    """Whether position i of a string falls inside $...$ or \\(...\\)."""
    before = text[:i]
    dollars = len(re.findall(r"(?<!\\)\$", before))
    opened = before.count("\\(") - before.count("\\)")
    return dollars % 2 == 1 or opened > 0


def source_commands(source: str) -> set[str]:
    return set(re.findall(r"\\([A-Za-z]+)", source or ""))


def _choose(control: str, word: str, local: set[str], decoded: bool):
    """(command, alternatives) for a control character followed by word, or None."""
    letter = ESCAPE_LETTER.get(control)
    known = COMMANDS | local
    if decoded:
        name = (letter or "") + word
        return (name, []) if letter and name in known else None
    found = []
    for name in known:
        dropped = len(name) - len(word)
        if 0 <= dropped <= MAX_DROPPED and name.endswith(word):
            head = name[:dropped]
            found.append(((dropped, name not in local,
                           not (letter and head.startswith(letter)),
                           not name[:1].islower(), name), name))
    if not found:
        return None
    found.sort()
    best = found[0][1]
    alternatives = [n for key, n in found[1:] if key[0] == found[0][0][0]]
    return best, alternatives


def restore_raw(raw: str, source: str = "") -> tuple[str, list[dict]]:
    """Repair raw control bytes inside the JSON strings of raw (RAW, above)."""
    local = source_commands(source)
    out: list[str] = []
    repairs: list[dict] = []
    in_string = escaped = False
    start = i = 0
    while i < len(raw):
        char = raw[i]
        if in_string and char in RAW_CONTROLS:
            word = re.match(r"[A-Za-z]*", raw[i + 1:]).group(0)
            prose_break = char in "\n\r" and not _in_math(raw[start:i], i - start)
            choice = (_choose(char, word, local, decoded=False)
                      if word and not prose_break else None)
            if choice:
                name, alternatives = choice
                out.append("\\\\" + name)          # two JSON backslashes decode to one
                entry = {"offset": i, "codepoint": ord(char), "restored-command": "\\" + name}
                if alternatives:
                    entry["alternatives"] = ["\\" + a for a in alternatives]
                repairs.append(entry)
                i += 1 + len(word)
                escaped = False
                continue
        out.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string, start = True, i + 1
        i += 1
    return "".join(out), repairs


def restore_text(text: str, source: str = "", path: str = "") -> tuple[str, list[dict]]:
    """Repair control characters that a parsed JSON string decoded to (DECODED)."""
    local = source_commands(source)
    repairs: list[dict] = []

    def fix(match: re.Match) -> str:
        control, word = match.group(1), match.group(2)
        if control in "\n\r" and not _in_math(text, match.start()):
            return match.group(0)
        choice = _choose(control, word, local, decoded=True)
        if not choice:
            return match.group(0)
        repairs.append({"path": path, "codepoint": ord(control), "restored-command": "\\" + choice[0]})
        return "\\" + choice[0]
    return re.sub(r"([\t\b\f\n\r])([A-Za-z]+)", fix, text), repairs


def restore_doc(value, source: str = "", path: str = ""):
    """(value, repairs) with restore_text applied to every string in a parsed document."""
    if isinstance(value, str):
        return restore_text(value, source, path)
    if isinstance(value, dict):
        repairs, out = [], {}
        for key, item in value.items():
            out[key], found = restore_doc(item, source, f"{path}.{key}" if path else str(key))
            repairs += found
        return out, repairs
    if isinstance(value, list):
        repairs, out = [], []
        for n, item in enumerate(value):
            fixed, found = restore_doc(item, source, f"{path}[{n}]")
            out.append(fixed)
            repairs += found
        return out, repairs
    return value, []


def parse(raw: str, source: str = "") -> tuple[object, str | None, list[dict]]:
    """(document, repaired raw text or None, repairs).

    Parses as-is; when that fails, repairs raw control bytes and parses again
    (raising the original error if that fails too). Decoded control characters
    are then restored in the document's strings. The raw text is never
    changed in place: the caller keeps it beside the repaired one.
    """
    sanitized = None
    raw_repairs: list[dict] = []
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as original:
        sanitized, raw_repairs = restore_raw(raw, source)
        if not raw_repairs:
            raise original
        try:
            doc = json.loads(sanitized)
        except json.JSONDecodeError:
            raise original
    doc, decoded = restore_doc(doc, source)
    for entry in decoded:
        entry["decoded"] = True
    return doc, sanitized, raw_repairs + decoded
