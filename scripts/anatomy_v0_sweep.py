#!/usr/bin/env python3
"""Anatomy v0 corpus sweep.

Stdlib-only deterministic substrate pass for the math.CT eprint corpus.
It emits one JSON object per paper under storage/futon6/data/ct-anatomy-v0.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import multiprocessing as mp
import os
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPRINTS = Path("/home/joe/code/storage/futon6/data/arxiv-math-ct-eprints")
DEFAULT_OUT = Path("/home/joe/code/storage/futon6/data/ct-anatomy-v0")
ROLE_TSV = ROOT / "holes" / "golden-graphs" / "latexml-math-roles.tsv"
PLAIN_CSEQ = ROOT / "holes" / "golden-graphs" / "tex-plain-cseq.txt"
TEXT_EXTS = {
    ".tex", ".sty", ".cls", ".bbl", ".bib", ".ltx", ".cfg", ".def", ".clo",
    ".fd", ".aux",
}
MATH_STRUCTURAL_PLAIN = {"atop", "over", "above", "choose", "mathchoice"}

# Alphabet/font wrappers: a macro defined as \mathcal{C}, \mathbf{Set}, etc.
# denotes an ATOM, not whatever role the wrapper itself carries. Treated as
# transparent in role resolution (the wrapped letters are the atom).
ALPHABET_WRAPPERS = {
    "mathcal", "mathbb", "mathbf", "mathrm", "mathsf", "mathfrak", "mathtt",
    "mathit", "mathscr", "boldsymbol", "operatorname", "operatornamewithlimits",
    "text", "textrm", "textbf", "textit", "textsf", "texttt", "bm", "mbox", "hbox",
}

# Standard math vocabulary the LaTeXML mining under-covers (loss-backlog C2):
# greek, alphabet wrappers, delimiters, spacing, common operators/relations.
# Seeded into the role table so direct uses don't read as UNKNOWN. Roles use
# the LaTeXML vocabulary (ID=atom; RELOP/ADDOP/MULOP/ARROW/SUMOP/INTOP/LIMITOP
# =operators; OPEN/CLOSE=delimiters; SPACE=layout).
def _build_standard_math_roles() -> dict[str, str]:
    out: dict[str, str] = {}
    greek = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta "
             "iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau "
             "upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi "
             "Sigma Upsilon Phi Psi Omega ell hbar imath jmath aleph nabla "
             "partial infty").split()
    for g in greek:
        out[g] = "ID"
    for w in ALPHABET_WRAPPERS:
        out[w] = "ID"  # atom-former; recognised, not unknown
    ops = {
        "frac": "MULOP", "cdot": "MULOP", "times": "MULOP", "otimes": "MULOP",
        "circ": "MULOP", "cap": "MULOP", "wedge": "MULOP", "prod": "MULOP",
        "oplus": "ADDOP", "vee": "ADDOP", "cup": "ADDOP", "pm": "ADDOP", "mp": "ADDOP",
        "le": "RELOP", "ge": "RELOP", "leq": "RELOP", "geq": "RELOP", "neq": "RELOP",
        "sim": "RELOP", "cong": "RELOP", "equiv": "RELOP", "approx": "RELOP",
        "subset": "RELOP", "subseteq": "RELOP", "supset": "RELOP", "supseteq": "RELOP",
        "in": "RELOP", "notin": "RELOP", "ni": "RELOP", "mid": "RELOP", "propto": "RELOP",
        "to": "ARROW", "rightarrow": "ARROW", "longrightarrow": "ARROW", "mapsto": "ARROW",
        "xrightarrow": "ARROW", "hookrightarrow": "ARROW", "twoheadrightarrow": "ARROW",
        "Rightarrow": "ARROW", "leftarrow": "ARROW", "leftrightarrow": "ARROW",
        "overset": "OVERACCENT", "underset": "OVERACCENT", "stackrel": "OVERACCENT",
        "sum": "SUMOP", "coprod": "SUMOP", "int": "INTOP", "oint": "INTOP",
        "lim": "LIMITOP", "colim": "LIMITOP", "varinjlim": "LIMITOP", "varprojlim": "LIMITOP",
    }
    out.update(ops)
    delims = {"langle": "OPEN", "rangle": "CLOSE", "lfloor": "OPEN", "rfloor": "CLOSE",
              "lceil": "OPEN", "rceil": "CLOSE", "{": "OPEN", "}": "CLOSE",
              "left": "OPEN", "right": "CLOSE", "big": "OPEN", "Big": "OPEN",
              "bigg": "OPEN", "Bigg": "OPEN"}
    out.update(delims)
    # C2 standard-vocab extension (claude-2, 2026-06-14): measured from the
    # CURRENT classifier-unknown tail (Greek/alphabets already cleared above; the
    # backlog's "Greek 18.2%" was stale). Adds the genuinely-missing standard
    # delimiters/relations/arrows/operators/atoms + layout/structural macros.
    # EXCLUDES xy-pic (\ar \ar@ \xymatrix = C3 package profile) and the bare
    # macro-parameter char. Additive: classify_cseq checks macros first, so these
    # only reclassify currently-UNKNOWN control sequences, never override.
    more_delims = ("lbrack rbrack lbrace rbrace vert Vert lvert rvert lVert rVert "
                   "lgroup rgroup ulcorner urcorner llcorner lrcorner backslash")
    for s in more_delims.split():
        out[s] = "OPEN" if s[0] in "lLuU" or s in ("vert", "Vert", "backslash") else "CLOSE"
    for s in ("ne doteq simeq asymp prec succ preceq succeq models vdash dashv "
              "perp parallel ll gg lll ggg subsetneq supsetneq sqsubseteq "
              "sqsupseteq sqsubset sqsupset triangleleft triangleright "
              "trianglelefteq trianglerighteq bowtie smile frown lesssim gtrsim "
              "approxeq backsim eqsim gtrless lessgtr ngeq nleq nsubseteq").split():
        out[s] = "RELOP"
    for s in ("xleftarrow xrightarrow Leftarrow Leftrightarrow Longrightarrow "
              "Longleftarrow longleftarrow longmapsto uparrow downarrow updownarrow "
              "Uparrow Downarrow nearrow searrow swarrow nwarrow hookleftarrow "
              "rightrightarrows leftleftarrows rightleftarrows leftrightarrows "
              "twoheadleftarrow rightarrowtail leftarrowtail dashrightarrow "
              "dashleftarrow rightsquigarrow leadsto rightharpoonup rightharpoondown "
              "leftharpoonup leftharpoondown rightleftharpoons").split():
        out[s] = "ARROW"
    for s in "bigcap bigcup bigsqcup bigvee bigwedge bigotimes bigoplus bigodot biguplus".split():
        out[s] = "SUMOP"
    for s in ("setminus smallsetminus ast star bullet diamond sqcap boxtimes "
              "boxdot ltimes rtimes wr odot ominus oslash dagger ddagger bigtriangleup "
              "bigtriangledown triangleq curlywedge divideontimes").split():
        out[s] = "MULOP"
    for s in "sqcup boxplus boxminus uplus amalg dotplus".split():
        out[s] = "ADDOP"
    for s in ("Box Diamond square blacksquare triangle blacktriangle emptyset "
              "varnothing top bot forall exists nexists neg lnot flat sharp natural "
              "angle measuredangle Re Im wp Bbbk prime backprime surd checkmark "
              "spadesuit heartsuit diamondsuit clubsuit star bigstar mho complement "
              "circledast circledcirc circleddash").split():
        out[s] = "ID"
    # layout / structural / font / reference macros: not math symbols -> layout
    # class (R15). Clears the GrCalc/text-mode tail (scalebox phantom fbox put ...).
    for s in ("begin end text scalebox phantom hphantom vphantom fbox framebox "
              "makebox raisebox put sf bf rm it tt sl scriptstyle displaystyle "
              "textstyle scriptscriptstyle hline noalign multicolumn label ref "
              "eqref cite notag tag intertext allowbreak smash limits nolimits "
              "newline linebreak textnormal textsc resizebox rule vspace* "
              "centering noindent").split():
        out[s] = "SPACE"
    for s in ("quad qquad ldots cdots vdots ddots dots hspace vspace smallskip "
              "medskip bigskip nonumber").split():
        out[s] = "SPACE"
    # single-char control symbols captured by CSEQ_RE's non-alpha branch:
    # `\#` is a literal hash atom (overloaded as smash-product in some papers);
    # `\\` (captured as cseq "\\") is the row break -> layout.
    out["#"] = "ID"
    out["\\"] = "SPACE"
    for s in "bigl Bigl biggl Biggl bigm Bigm biggm Biggm".split():
        out[s] = "OPEN"
    for s in "bigr Bigr biggr Biggr".split():
        out[s] = "CLOSE"
    for s in "dag ddag".split():
        out[s] = "ID"
    # non-alpha control symbols: \! \, \; \: \> thin/med spaces (layout);
    # \sb \sp are plain-TeX subscript/superscript aliases (engine-layer).
    for s in ("!", ",", ";", ":", ">", "sb", "sp"):
        out[s] = "SPACE"
    return out

STANDARD_MATH_ROLES = _build_standard_math_roles()


def strip_archive_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tex.gz", ".gz", ".tar", ".bin", ".tex"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def safe_decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore")


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        cut = None
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
        out.append(line if cut is None else line[:cut] + ("\n" if line.endswith("\n") else ""))
    return "".join(out)


def read_eprint_files(path: Path) -> tuple[list[dict], dict]:
    """Return text-ish files from an eprint archive, including .sty/.cls."""
    files = []
    meta = {"path": str(path), "status": "unknown"}
    lower = path.name.lower()
    try:
        if lower.endswith((".tar.gz", ".tar")):
            try:
                with tarfile.open(path, "r:*") as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        mpath = Path(member.name)
                        if mpath.suffix.lower() not in TEXT_EXTS:
                            continue
                        fh = tf.extractfile(member)
                        if fh is None:
                            continue
                        files.append({"file": member.name, "text": safe_decode(fh.read())})
                if files:
                    meta["status"] = "tar"
                    return files, meta
            except tarfile.TarError as exc:
                meta.setdefault("attempts", []).append({"tar-error": str(exc)})
                if not lower.endswith(".gz"):
                    return files, {**meta, "status": "tar-error"}

        if lower.endswith(".gz"):
            raw = gzip.decompress(path.read_bytes())
            files.append({"file": strip_archive_suffix(path) + ".tex", "text": safe_decode(raw)})
            meta["status"] = "plain-gzip"
            return files, meta

        if lower.endswith(".tex"):
            files.append({"file": path.name, "text": path.read_text(encoding="utf-8", errors="ignore")})
            meta["status"] = "plain-tex"
            return files, meta

        if lower.endswith(".bin"):
            raw = path.read_bytes()
            try:
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
                    for member in tf.getmembers():
                        if member.isfile() and Path(member.name).suffix.lower() in TEXT_EXTS:
                            fh = tf.extractfile(member)
                            if fh is not None:
                                files.append({"file": member.name, "text": safe_decode(fh.read())})
                if files:
                    meta["status"] = "bin-tar"
                    return files, meta
            except tarfile.TarError:
                pass
            files.append({"file": path.name, "text": safe_decode(raw)})
            meta["status"] = "bin-text"
            return files, meta
    except Exception as exc:
        meta["status"] = "error"
        meta["error"] = repr(exc)
    return files, meta


def load_latexml_roles(path: Path) -> dict[str, dict]:
    roles = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        roles[parts[0]] = {
            "role": parts[1],
            "source": parts[2] if len(parts) > 2 else "",
            "line": line_no,
        }
    # Seed/override with curated standard vocab (C2). Standard wins: the
    # mining left e.g. \mathbb tagged OVERACCENT, which mis-typed every
    # \mathbb{...}-defined atom; the curated entries are authoritative here.
    for k, v in STANDARD_MATH_ROLES.items():
        roles[k] = {"role": v, "source": "standard-vocab", "line": 0}
    return roles


def load_plain_cseq(path: Path) -> set[str]:
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line.lstrip("\\"))
    return out


def line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def parse_balanced_brace(text: str, open_pos: int) -> tuple[str | None, int]:
    if open_pos >= len(text) or text[open_pos] != "{":
        return None, open_pos
    depth = 0
    i = open_pos
    out_start = open_pos + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[out_start:i], i + 1
        i += 1
    return None, open_pos


def skip_space_and_options(text: str, pos: int) -> int:
    pos = skip_ws(text, pos)
    while pos < len(text) and text[pos] == "[":
        end = text.find("]", pos + 1)
        if end < 0:
            return pos
        pos = skip_ws(text, end + 1)
    return pos


def skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def control_sequences(text: str) -> list[str]:
    return [m.group(1) or m.group(2) for m in re.finditer(r"\\([A-Za-z@]+)|\\([^A-Za-z\s])", text)]


def collect_macros(files: list[dict], roles: dict[str, dict]) -> dict[str, dict]:
    macros: dict[str, dict] = {}
    cmd_re = re.compile(
        r"\\(?P<kind>newcommand|renewcommand)\*?\s*(?:\{\\(?P<braced>[A-Za-z@]+)\}|\\(?P<bare>[A-Za-z@]+))",
        re.S,
    )
    op_re = re.compile(
        r"\\(?P<kind>DeclareMathOperator)\*?\s*\{\\(?P<cs>[A-Za-z@]+)\}",
        re.S,
    )
    def_re = re.compile(r"\\def\s*\\(?P<cs>[A-Za-z@]+)", re.S)
    for f in files:
        text = strip_comments(f["text"])
        for m in cmd_re.finditer(text):
            pos = skip_space_and_options(text, m.end())
            rhs, _end = parse_balanced_brace(text, pos)
            if rhs is None:
                continue
            cs = m.group("braced") or m.group("bare")
            macros[cs] = {
                "cs": "\\" + cs,
                "rhs": rhs.strip(),
                "kind": "\\" + m.group("kind"),
                "file": f["file"],
                "line": line_for_offset(text, m.start()),
                "offset": m.start(),
            }
        for m in op_re.finditer(text):
            pos = skip_ws(text, m.end())
            rhs, _end = parse_balanced_brace(text, pos)
            if rhs is None:
                continue
            cs = m.group("cs")
            macros[cs] = {
                "cs": "\\" + cs,
                "rhs": rhs.strip(),
                "kind": "\\DeclareMathOperator",
                "file": f["file"],
                "line": line_for_offset(text, m.start()),
                "offset": m.start(),
            }
        for m in def_re.finditer(text):
            pos = m.end()
            while pos < len(text) and text[pos] != "{":
                pos += 1
            rhs, _end = parse_balanced_brace(text, pos)
            if rhs is None:
                continue
            cs = m.group("cs")
            macros[cs] = {
                "cs": "\\" + cs,
                "rhs": rhs.strip(),
                "kind": "\\def",
                "file": f["file"],
                "line": line_for_offset(text, m.start()),
                "offset": m.start(),
            }
    for cs in list(macros):
        role = resolve_macro_role(cs, macros, roles, seen=set())
        macros[cs]["role"] = role["role"]
        macros[cs]["role_source"] = role.get("source")
        macros[cs]["role_via"] = role.get("via", [])
    return macros


def resolve_macro_role(cs: str, macros: dict[str, dict], roles: dict[str, dict], seen: set[str]) -> dict:
    if cs in seen:
        return {"role": "UNKNOWN", "via": ["cycle"]}
    seen.add(cs)
    macro = macros.get(cs)
    if not macro:
        if cs in roles:
            r = roles[cs]
            return {"role": r["role"], "source": f"latexml-math-roles.tsv:{r['line']}", "via": [f"\\{cs}"]}
        return {"role": "UNKNOWN", "via": [f"\\{cs}"]}
    rhs = macro.get("rhs", "")
    # Alphabet/font wrappers are transparent: the wrapped letters are the atom.
    for rhs_cs in control_sequences(rhs):
        if rhs_cs in ALPHABET_WRAPPERS:
            continue
        if rhs_cs in roles:
            r = roles[rhs_cs]
            return {
                "role": r["role"],
                "source": f"latexml-math-roles.tsv:{r['line']}",
                "via": [macro["cs"], "\\" + rhs_cs],
            }
        if rhs_cs in macros:
            resolved = resolve_macro_role(rhs_cs, macros, roles, seen)
            if resolved.get("role") != "UNKNOWN":
                resolved["via"] = [macro["cs"], *resolved.get("via", [])]
                return resolved
    # No operator cseq in the RHS. If what remains (after stripping wrappers,
    # braces, scripts) is letters/digits, the macro names an ATOM — e.g.
    # \C := {\mathcal C}, \Set := \mathbf{Set}, \Hom := \mathrm{Hom}. Role ID,
    # not UNKNOWN (this is the dominant false-unknown class — registry seed).
    stripped = re.sub(r"\\[A-Za-z@]+|[{}$\\\s^_]", "", rhs)
    if stripped and re.fullmatch(r"[A-Za-z0-9'’.,\-]+", stripped):
        return {"role": "ID", "source": "atom-from-rhs", "via": [macro["cs"], "(atom)"]}
    return {"role": "UNKNOWN", "via": [macro["cs"]]}


def immediate_word_before(text: str, pos: int) -> str | None:
    prefix = text[max(0, pos - 100):pos]
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", prefix)
    return words[-1] if words else None


def authored_layer(files: list[dict]) -> dict:
    labels, refs, cites, stackrels = [], [], [], []
    cite_re = re.compile(r"\\cite\w*?(?:\[(?P<locus>[^\]]+)\])?\{(?P<keys>[^}]+)\}")
    for f in files:
        text = strip_comments(f["text"])
        for m in re.finditer(r"\\label\{([^}]+)\}", text):
            labels.append({"label": m.group(1), "file": f["file"], "offset": m.start(), "line": line_for_offset(text, m.start())})
        for m in re.finditer(r"\\(?P<kind>eqref|ref)\{(?P<label>[^}]+)\}", text):
            refs.append({
                "kind": "\\" + m.group("kind"),
                "label": m.group("label"),
                "ref-type": immediate_word_before(text, m.start()),
                "file": f["file"],
                "offset": m.start(),
                "line": line_for_offset(text, m.start()),
            })
        for m in cite_re.finditer(text):
            cites.append({
                "keys": [k.strip() for k in m.group("keys").split(",")],
                "locus": m.group("locus"),
                "file": f["file"],
                "offset": m.start(),
                "line": line_for_offset(text, m.start()),
            })
        for m in re.finditer(r"\\stackrel\{(?P<why>[^{}]+)\}\{=\}", text):
            stackrels.append({
                "justification": m.group("why"),
                "file": f["file"],
                "offset": m.start(),
                "line": line_for_offset(text, m.start()),
            })
    return {"labels": labels, "refs": refs, "cites": cites, "stackrel-justifications": stackrels}


def classify_cseq(cs: str, macros: dict[str, dict], roles: dict[str, dict], plain: set[str]) -> dict:
    if cs in macros:
        m = macros[cs]
        return {
            "cs": "\\" + cs,
            "class": "author-defined",
            "role": m.get("role", "UNKNOWN"),
            "source": f"{m['file']}:{m['line']}",
            "rhs": m.get("rhs", ""),
        }
    if cs in roles:
        r = roles[cs]
        return {"cs": "\\" + cs, "class": "latexml-standard-math", "role": r["role"], "source": f"latexml-math-roles.tsv:{r['line']}"}
    if cs in plain:
        role = "MATH-STRUCTURAL" if cs in MATH_STRUCTURAL_PLAIN else "engine-layer"
        return {"cs": "\\" + cs, "class": "tex-plain", "role": role}
    return {"cs": "\\" + cs, "class": "UNKNOWN", "role": "UNKNOWN"}


def _unescaped(text: str, k: int) -> bool:
    r"""True if text[k] is NOT backslash-escaped — i.e. preceded by an EVEN
    number of consecutive backslashes (zero counts). Critical for `$`-parity:
    `\\$` (a LaTeX line-break `\\` then `$`) is a REAL math delimiter, while
    `\$` (a lone backslash) escapes the `$`. The old `text[k-1] != '\\'` test
    mis-read `\\$$`/`\\$` (line-break + display/inline open, common in GrCalc
    bodies like `}\\$$`) as an escaped `$`, dropped the delimiter, and let
    `$`-parity drift so inter-formula PROSE was swallowed into a giant spurious
    span. Counting backslash parity is strictly MORE correct (it never escapes
    a `$` the old test treated as real; it only RECOVERS real delimiters the
    old test wrongly skipped), so spans only tighten."""
    b = 0
    k -= 1
    while k >= 0 and text[k] == "\\":
        b += 1
        k -= 1
    return b % 2 == 0


def math_spans(text: str):
    # A LaTeX comment (unescaped `%` to end of line) is NOT math — a `$` inside
    # one is not a delimiter. Skipping comment regions (offset-preserving: we
    # advance past them, we do not delete) keeps `$`-parity correct; otherwise a
    # lone comment-`$` mis-pairs real delimiters and swallows prose into giant
    # spurious "math spans" (verified on 0708.3326: 877 comment-`$` -> 838
    # W-ATOMIC + 1758 null spans). Callers that pre-`strip_comments` see no `%`,
    # so this is a no-op for them.
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "%" and (i == 0 or text[i - 1] != "\\"):
            nl = text.find("\n", i)
            if nl == -1:
                return
            i = nl + 1
            continue
        if ch != "$" or not _unescaped(text, i):
            i += 1
            continue
        delim = "$$" if i + 1 < n and text[i + 1] == "$" else "$"
        start = i
        j = i + len(delim)
        found = False
        while j < n:
            if text[j] == "%" and text[j - 1] != "\\":
                nl = text.find("\n", j)
                if nl == -1:
                    break
                j = nl + 1
                continue
            if text.startswith(delim, j) and _unescaped(text, j):
                yield start, j + len(delim), delim, text[i + len(delim):j]
                i = j + len(delim)
                found = True
                break
            j += 1
        if not found:
            i = start + 1


def token_census(files: list[dict], macros: dict[str, dict], roles: dict[str, dict], plain: set[str]) -> dict:
    spans = []
    classified_total = unknown_total = fully = role_gap_total = 0
    unknowns = Counter()
    role_gaps = Counter()
    for f in files:
        text = strip_comments(f["text"])
        for start, end, delim, body in math_spans(text):
            controls = []
            span_unknown = []
            for cs in control_sequences(body):
                cls = classify_cseq(cs, macros, roles, plain)
                controls.append(cls)
                if cls["class"] == "UNKNOWN":
                    # genuinely unrecognised: not in the paper's macros, not in
                    # the role lexicon, not a known TeX/plain control sequence.
                    unknown_total += 1
                    unknowns["\\" + cs] += 1
                    span_unknown.append("\\" + cs)
                else:
                    # recognised token. A still-UNKNOWN role is a role-gap
                    # (typing refinement), NOT a false "unknown control seq".
                    classified_total += 1
                    if cls["role"] == "UNKNOWN":
                        role_gap_total += 1
                        role_gaps["\\" + cs] += 1
            if not span_unknown:
                fully += 1
            spans.append({
                "file": f["file"],
                "offset": start,
                "line": line_for_offset(text, start),
                "delimiter": delim,
                "control-count": len(controls),
                "classified": len(controls) - len(span_unknown),
                "unknown": len(span_unknown),
                "unknown-list": sorted(set(span_unknown)),
                "controls": controls,
            })
    return {
        "dollar-spans": len(spans),
        "classified": classified_total,
        "unknown": unknown_total,
        "unknown-list": sorted(unknowns),
        "role-gap": role_gap_total,
        "role-gap-list": sorted(role_gaps),
        "spans-fully-classified": fully,
        "spans": spans,
    }


def analyze_one(task: tuple[str, str, str, bool]) -> dict:
    eprint_path_s, out_dir_s, root_s, force = task
    eprint_path = Path(eprint_path_s)
    out_dir = Path(out_dir_s)
    entity = strip_archive_suffix(eprint_path)
    out_path = out_dir / f"{entity}.json"
    if out_path.exists() and not force:
        try:
            row = json.loads(out_path.read_text())
            return {"entity": entity, "skipped": True, "ok": True, "aggregates": row.get("satiety-aggregates", {})}
        except Exception:
            pass

    roles = load_latexml_roles(Path(root_s) / "holes" / "golden-graphs" / "latexml-math-roles.tsv")
    plain = load_plain_cseq(Path(root_s) / "holes" / "golden-graphs" / "tex-plain-cseq.txt")
    files, meta = read_eprint_files(eprint_path)
    if not files:
        row = {"entity": entity, "source": str(eprint_path), "status": "no-files", "loader": meta}
        out_path.write_text(json.dumps(row, sort_keys=True) + "\n")
        return {"entity": entity, "ok": False, "aggregates": {}}
    macros = collect_macros(files, roles)
    authored = authored_layer(files)
    census = token_census(files, macros, roles, plain)
    macro_rows = sorted(macros.values(), key=lambda r: (r["file"], r["line"], r["cs"]))
    aggregates = {
        "macro-defs": len(macro_rows),
        "labels": len(authored["labels"]),
        "refs": len(authored["refs"]),
        "cites-with-locus": sum(1 for c in authored["cites"] if c.get("locus")),
        "stackrel-justifications": len(authored["stackrel-justifications"]),
        "unknown-cseqs": len(census["unknown-list"]),
        "dollar-spans": census["dollar-spans"],
        "spans-fully-classified": census["spans-fully-classified"],
    }
    row = {
        "entity": entity,
        "source": str(eprint_path),
        "loader": meta,
        "files": [{"file": f["file"], "chars": len(f["text"])} for f in files],
        "symbol-table": macro_rows,
        "authored-layer": authored,
        "token-census": census,
        "satiety-aggregates": aggregates,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(out_path)
    return {"entity": entity, "ok": True, "skipped": False, "aggregates": aggregates}


def aggregate_results(results: list[dict]) -> dict:
    totals = Counter()
    ok = skipped = failed = 0
    for r in results:
        ok += 1 if r.get("ok") else 0
        skipped += 1 if r.get("skipped") else 0
        failed += 0 if r.get("ok") else 1
        totals.update(r.get("aggregates") or {})
    return {"papers": len(results), "ok": ok, "failed": failed, "skipped": skipped, "totals": dict(totals)}


def iter_eprints(eprint_dir: Path) -> list[Path]:
    paths = [p for p in eprint_dir.iterdir() if p.is_file() and p.name.endswith((".tar.gz", ".gz", ".tar", ".tex", ".bin"))]
    return sorted(paths, key=lambda p: p.name)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eprints", type=Path, default=DEFAULT_EPRINTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paper-id", action="append", default=[], help="Restrict to safe id(s), e.g. 0809.2517 or math__0607126")
    ap.add_argument("--force", action="store_true", help="Recompute even when output exists")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = iter_eprints(args.eprints)
    if args.paper_id:
        wanted = set(args.paper_id)
        paths = [p for p in paths if strip_archive_suffix(p) in wanted]
    if args.limit is not None:
        paths = paths[: args.limit]
    start = time.time()
    tasks = [(str(p), str(args.out_dir), str(ROOT), args.force) for p in paths]
    results = []
    if args.workers == 1:
        iterator = map(analyze_one, tasks)
    else:
        pool = mp.Pool(processes=args.workers)
        iterator = pool.imap_unordered(analyze_one, tasks, chunksize=8)
    try:
        for idx, result in enumerate(iterator, 1):
            results.append(result)
            if idx % 200 == 0 or idx == len(tasks):
                snap = aggregate_results(results)
                elapsed = time.time() - start
                print(
                    f"[anatomy-v0] {idx}/{len(tasks)} elapsed={elapsed:.1f}s ok={snap['ok']} skipped={snap['skipped']} failed={snap['failed']} totals={snap['totals']}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if args.workers != 1:
            pool.close()
            pool.join()
    summary = aggregate_results(results)
    summary["elapsed-sec"] = round(time.time() - start, 3)
    summary["workers"] = args.workers
    summary["out-dir"] = str(args.out_dir)
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
