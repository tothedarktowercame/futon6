#!/usr/bin/env python3
"""Markup strategies: what a paper's own text settles, with no model, as a hypergraph.

S1 marks each token where it finds it. Two facts it does not record follow from the
text alone:

  terms    A term the paper defines (typeset as the definiendum inside a definition)
           is that term wherever it is used. One hyperedge per term joins its
           definition site to every use; a use of a parametrised term ($\\B$-preenvelope
           for the defined $\\F$-preenvelope) records the parameter it instantiates.
  symbols  Every occurrence of a symbol in math is an occurrence of that symbol. One
           hyperedge per symbol name joins all its binding sites to all its
           occurrences. Whether an occurrence is *the same* S as a given binding is
           not settled by the name, so each occurrence is linked to one binding site
           by a stated rule, and the rule is recorded with the link:
             in-environment  a binding earlier in the same theorem/definition/proof
             proved-statement  in a proof, a binding in the statement it proves
             in-section      assumed: the nearest earlier binding in the section
             in-paper        assumed: the nearest earlier binding in the paper
             unbound         no binding of this name before it
           S1's own grounding (the `bound` text on a symbol-grounded mark) is kept
           beside the link and compared with the binding's type.

Binding sites are S1's (definiendum, bind/let, bind/define, bind/typed) plus one
pattern S1 lacks, the article-noun apposition of a definition: "An object $S$ of
$\\T$ will be called ...".

Usage: scripts/markup_strategies.py MARKS_JSON [OUT_JSON]
       scripts/markup_strategies.py --list IDS --marks MARKS_DIR --out OUT_DIR
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

import expository_region_extract as regions

SYMBOL_KINDS = ("symbol", "symbol-grounded")
# A macro the paper defines for one of its objects (\T for the triangulated category)
# is a symbol too, but S1 files it as `classified` with the role ID.
MACRO_SYMBOL = re.compile(r"·\s*author-defined\s*·\s*ID\b")
ENV_KINDS = {"theorem", "lemma", "proposition", "corollary", "definition", "remark", "example",
             "proof", "question", "conjecture", "claim", "notation"}
# Emphasis in either spelling: \emph{...} and the older {\em ...}.
EMPH = re.compile(r"\\(?:emph|textbf|textit)\{((?:[^{}]|\{[^{}]*\})*)\}"
                  r"|\{\\(?:em|it|bf|sl)\s+([^{}]*)\}")
# A paper need not have definition environments: many define in running prose, and
# 0806.1324 (Krause) has no definition environment at all while calling 25 terms by
# name ("A category $\C$ is called \emph{small} if ...").
CALLED = re.compile(r"\b(?:is|are|will\s+be|shall\s+be)\s+called\s+(?:an?\s+|the\s+)?"
                    r"(?:\\(?:emph|textbf|textit)\{([^{}]{2,80})\}|\{\\(?:em|it|bf|sl)\s+([^{}]{2,80})\})"
                    r"|\b[Ww]e\s+(?:shall\s+|will\s+|now\s+)?call\s+[^.]{0,60}?"
                    r"(?:\\(?:emph|textbf|textit)\{([^{}]{2,80})\}|\{\\(?:em|it|bf|sl)\s+([^{}]{2,80})\})")


def emphasised(match: re.Match) -> str:
    r"""The emphasised text, whichever spelling matched, as a term or "" if it is not one.

    Emphasis carries line breaks ("small\nsets"), italic corrections (map\/) and
    occasionally a fragment with an unclosed bracket; a term also needs a letter.
    """
    raw = next(g for g in match.groups() if g is not None)
    term = re.sub(r"\s+", " ", raw).strip()
    term = re.sub(r"\\/$", "", term).strip(" ,;:.-")
    if term.count("(") != term.count(")"):
        term = term.split("(")[0].strip()
    return term if len(term) >= 3 and re.search(r"[A-Za-z]{2}", term) else ""
APPOSITION = re.compile(r"\b(?:[Aa]n?|[Tt]he|[Aa]ny|[Ee]very|[Ss]ome)\s+((?:[a-z][a-z-]*\s+){0,4}?[a-z][a-z-]*)\s+"
                        r"\$([^$]{1,24})\$((?:\s+(?:of|in|on)\s+\$[^$]{1,30}\$)?)(?![-\w])")
QUANTIFIED = re.compile(r"\bfor\s+(?:all|each|every|any)\s+\$([^$]{1,12})\$(?![-\w])")
# An apposition binds when its head noun names a mathematical thing. S1's concept marks
# miss many of these ("The object $B$" is unmarked in 0705.0102), so the list stands on
# its own and S1's concepts only add to it.
HEAD_NOUNS = {"object", "morphism", "map", "mapping", "category", "subcategory", "functor", "element",
              "ring", "module", "set", "subset", "algebra", "space", "subspace", "complex", "triangle",
              "sequence", "number", "integer", "group", "subgroup", "field", "ideal", "sheaf", "scheme",
              "variety", "matrix", "vector", "point", "function", "homomorphism", "isomorphism",
              "embedding", "projection", "inclusion", "class", "pair", "tuple", "family", "collection",
              "diagram", "arrow", "cone", "cocone", "limit", "colimit", "coproduct", "product", "quotient",
              "extension", "resolution", "filtration", "spectrum", "operator", "form", "curve", "surface",
              "manifold", "graph", "tree", "path", "cycle", "measure", "series", "sum", "basis", "constant",
              "variable", "index", "parameter", "structure", "system", "sequence", "series"}
WORD = re.compile(r"[a-z]{3,}")
STOP = {"the", "and", "for", "with", "that", "this", "its", "are", "which", "such", "some", "any"}


def _line_starts(text: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer("\n", text)]


def _line(starts: list[int], offset: int) -> int:
    return bisect.bisect_right(starts, offset)


def _fields(mark: dict) -> dict:
    return dict(mark.get("fields") or [])


def _content(text: str) -> set[str]:
    return set(WORD.findall(re.sub(r"\$[^$]*\$|\\[A-Za-z]+", " ", (text or "").lower()))) - STOP


# ---- terms ---------------------------------------------------------------------------

def term_pattern(term: str) -> tuple[str, bool]:
    """A regex for uses of `term`, and whether it has a leading parameter.
    A leading $\\F$- or $n$- stands for any symbol; the rest of the term is literal."""
    parts = re.split(r"(\$[^$]*\$)", term)
    out, param = [], False
    for i, part in enumerate(parts):
        lead = part.startswith("$") and i == 1 and not parts[0] and (
            part[1:-1] == "n" or part[1:-1].startswith("\\F"))
        param = param or lead
        out.append(r"(\$[^$]{1,40}\$)" if lead else re.escape(part))
    return "".join(out).replace(r"\ ", r"\s+") + "s?", param


# Emphasis is also how a bibliography sets journal and publisher names, and this paper
# emphasises 21 of them ("J. Math. Phys.", "Springer-Verlag", "preprint math.QA/9802029").
# None of them is a term the paper defines.
BIBLIOGRAPHIC = re.compile(r"\d|\b[A-Z][a-z]{0,4}\.|\b(?:preprint|Press|Verlag|Notes|Ann(?:als)?|Bull|Soc|Publ|Adv|Inc)\b")


def bibliographic(term: str) -> bool:
    """A journal, publisher or preprint number set in italics, not a defined term."""
    return bool(BIBLIOGRAPHIC.search(term)) or len(term) > 80 or len(term.split()) > 8


def bibliography_at(text: str, marks: list[dict]) -> int:
    """Where the references begin; emphasis after this point is a citation."""
    for m in marks:
        if m["kind"] == "env/thebibliography":
            return m["start"]
    at = text.find("\\begin{thebibliography}")
    return at if at >= 0 else len(text)


def defined_terms(text: str, marks: list[dict]) -> list[dict]:
    """Every term the paper defines, with its definition site and every use."""
    starts = _line_starts(text)
    body = max(0, text.find("\\begin{document}"))
    # How each term was found is part of what is known about it: emphasis in running
    # prose is weaker evidence than a definition environment, and a reader can weigh
    # them only if the page says which.
    found: dict[str, tuple[int, dict | None, str]] = {}
    defs = [m for m in marks if m["kind"] == "env/definition"]
    for m in defs:
        for e in EMPH.finditer(text, m["start"], m["end"]):
            if emphasised(e):
                found.setdefault(emphasised(e), (e.start(), m, "in a definition environment"))
    # Terms defined in prose: "... is called \emph{X}", "we call ... {\em X}".
    for e in CALLED.finditer(text, body):
        if emphasised(e):
            found.setdefault(emphasised(e), (e.start(), None, "called by name"))
    # Emphasis in running prose IS the definition, in the commonest shape mathematics
    # uses: "The {\em nerve} $N\C$ of the category $\C$ is the value on ...". Reading
    # emphasis only inside a definition environment or after "is called" found 10 of
    # math/9906038's terms and missed 20, nerve among them, though the paper italicises
    # every one of them at its first use (Joe, 2026-09-22).
    for e in EMPH.finditer(text, body):
        term = emphasised(e)
        if term and not bibliographic(term) and e.start() < bibliography_at(text, marks):
            found.setdefault(term, (e.start(), None, "emphasised at its first use"))
    import build_golden_paper                         # S1's own miner: "the \textit{heart} of"
    for d in build_golden_paper.mine_definitions(text):
        for e in EMPH.finditer(d.term):
            env = next((m for m in defs if m["start"] <= d.position < m["end"]), None)
            if emphasised(e):
                found.setdefault(emphasised(e), (d.position, env, "named by a definition pattern"))
    concepts = [m for m in marks if m["kind"] == "concept"]
    taken: list[tuple[int, int]] = []
    out = []
    for term, (at, env, how_found) in sorted(found.items(), key=lambda kv: -len(kv[0])):  # longest first
        pat, has_param = term_pattern(term)
        pats = [(pat, has_param)]
        head = re.sub(r"^\$[^$]*\$-", "", term)
        if head != term and head not in found:
            pats.append((re.escape(head) + "s?", False))
        uses = []
        for p, param in pats:
            for o in re.finditer(r"(?<![A-Za-z-])" + p + r"(?![A-Za-z])", text[body:], re.I):
                a, b = o.start() + body, o.end() + body
                if any(a < y and x < b for x, y in taken):
                    continue
                taken.append((a, b))
                over = [c for c in concepts if c["start"] < b and a < c["end"]]
                src = [_fields(c) for c in over]
                if any(f.get("source") == "defined-in-paper" for f in src):
                    s1, how = "tagged as defined", ""
                elif over:
                    s1 = "tagged generically"
                    how = ", ".join(sorted({f.get("grounded") or f.get("source") or "concept" for f in src}))
                else:
                    s1, how = "untagged", ""
                use = {"start": a, "end": b, "line": _line(starts, a), "s1": s1, "s1-grounding": how,
                       "definition-site": a <= at < b}
                if param and o.lastindex:
                    use["parameter"] = o.group(1)
                uses.append(use)
        uses.sort(key=lambda u: u["start"])
        # Without an environment, the definition shown is the sentence that names the
        # term: from the end of the previous sentence to the end of this one. A
        # sentence may end at a newline as readily as at a space.
        if env:
            where = env
        else:
            before = [m.end() for m in re.finditer(r"[.!?]\s", text[body:at])]
            after = re.search(r"[.!?](\s|$)", text[at:])
            where = {"start": body + (before[-1] if before else 0),
                     "end": at + (after.end() if after else 300)}
        out.append({"term": term, "at": at, "line": _line(starts, at), "how-found": how_found,
                    "definition": re.sub(r"\s+", " ", text[where["start"]:where["end"]])[:600],
                    "definition-span": [where["start"], where["end"]], "uses": uses})
    out.sort(key=lambda t: t["at"])
    return out


# ---- symbols -------------------------------------------------------------------------

def binding_sites(text: str, marks: list[dict]) -> list[dict]:
    """Where a symbol is introduced, with what it is said to be."""
    starts = _line_starts(text)
    sites = []
    definiens = sorted((m for m in marks if m["kind"] == "definiens"), key=lambda m: m["start"])
    for m in marks:
        k, f = m["kind"], _fields(m)
        if k == "definiendum":
            name = text[m["start"]:m["end"]].strip("$ ")
            follow = next((d for d in definiens if d["start"] >= m["end"] and d["start"] - m["end"] < 60), None)
            kind_of = text[follow["start"]:follow["end"]] if follow else ""
            sites.append({"name": name, "start": m["start"], "end": m["end"], "type": kind_of, "source": "definiendum"})
        elif k in ("bind/let", "bind/define", "bind/typed") and f.get("symbol"):
            sym = f["symbol"]
            at = text.find(sym, m["start"], m["end"])
            at = m["start"] if at < 0 else at
            sites.append({"name": sym, "start": at, "end": at + len(sym),
                          "type": f.get("type") or f.get("description") or "", "source": k})
    body = max(0, text.find("\\begin{document}"))
    # "An object $S$ of $\T$ will be called ...": an apposition counts only when its head
    # noun is a term S1 recognised, which keeps "the authors look at $t$" out.
    concepts = sorted((m["start"], m["end"]) for m in marks if m["kind"] == "concept")
    heads = [c[0] for c in concepts]
    def is_concept(a: int, b: int) -> bool:
        i = bisect.bisect_right(heads, b)
        while i > 0 and concepts[i - 1][0] > a - 200:       # every concept that could reach a
            i -= 1
            if concepts[i][0] < b and a < concepts[i][1]:
                return True
        return False
    for o in APPOSITION.finditer(text, body):
        noun, name, of = o.group(1), o.group(2).strip(), o.group(3).strip()
        head = noun.split()[-1]
        head_start = o.start(1) + len(noun) - len(head)
        if head.rstrip("s") not in {h.rstrip("s") for h in HEAD_NOUNS} and not is_concept(head_start, o.end(1)):
            continue
        sites.append({"name": name, "start": o.start(2) - 1, "end": o.end(2) + 1,
                      "type": (noun + (" " + of if of else "")).strip(), "source": "apposition"})
    for o in QUANTIFIED.finditer(text, body):
        sites.append({"name": o.group(1).strip(), "start": o.start(1) - 1, "end": o.end(1) + 1,
                      "type": "a quantified variable", "source": "quantifier"})
    # One site per place: several detectors often find the same "Let $S$ be ...".
    sites.sort(key=lambda s: (s["start"], -len(s["type"])))
    merged: list[dict] = []
    for s in sites:
        if merged and merged[-1]["name"] == s["name"] and abs(merged[-1]["start"] - s["start"]) <= 2:
            merged[-1]["source"] += "+" + s["source"]
            if not merged[-1]["type"]:
                merged[-1]["type"] = s["type"]
            continue
        merged.append(dict(s))
    for s in merged:
        s["line"] = _line(starts, s["start"])
    return merged


def symbol_hypergraph(text: str, marks: list[dict]) -> list[dict]:
    starts = _line_starts(text)
    envs = sorted(((m["start"], m["end"], m["kind"][4:]) for m in marks
                   if m["kind"].startswith("env/") and m["kind"][4:] in ENV_KINDS), key=lambda e: (e[0], -e[1]))
    lines = text.splitlines()
    body_start, body_end = regions.find_body_range(lines)
    sections = regions.parse_sections(lines, body_start, body_end)
    def section_of(offset: int):
        line = _line(starts, offset)
        inside = [s for s in sections if s.line_start <= line <= s.line_end]
        return max(inside, key=lambda s: s.line_start) if inside else None
    def innermost(offset: int):
        inside = [e for e in envs if e[0] <= offset < e[1]]
        return min(inside, key=lambda e: e[1] - e[0]) if inside else None
    def proved_by(proof):
        before = [e for e in envs if e[2] != "proof" and e[1] <= proof[0] and proof[0] - e[1] < 400]
        return max(before, key=lambda e: e[1]) if before else None

    sites = binding_sites(text, marks)
    by_name: dict[str, list[dict]] = {}
    for s in sites:
        by_name.setdefault(s["name"], []).append(s)
    occ = sorted((m for m in marks if m["kind"] in SYMBOL_KINDS
                  or (m["kind"] == "classified" and MACRO_SYMBOL.search(m.get("tip") or ""))),
                 key=lambda m: m["start"])
    # A letter inside the name of a defined term (the t of co-$t$-structure) is part of
    # that name, not a variable standing on its own.
    in_term = sorted((u["start"], u["end"]) for t in defined_terms(text, marks) for u in t["uses"]
                     if "$" in t["term"])
    def named(a: int, b: int) -> bool:
        i = bisect.bisect_right(in_term, (a, float("inf")))
        return any(x <= a and b <= y for x, y in in_term[max(0, i - 4):i])
    occ = [m for m in occ if not named(m["start"], m["end"])]
    edges: dict[str, dict] = {}
    for m in occ:
        name = text[m["start"]:m["end"]]
        binders = by_name.get(name, [])
        o = m["start"]
        env = innermost(o)
        choice, rule = None, "unbound"
        if env:
            here = [b for b in binders if env[0] <= b["start"] <= o]
            if here:
                choice, rule = here[-1], "in-environment"
            elif env[2] == "proof" and (st := proved_by(env)):
                stated = [b for b in binders if st[0] <= b["start"] < st[1]]
                if stated:
                    choice, rule = stated[-1], "proved-statement"
        if choice is None:
            sec = section_of(o)
            earlier = [b for b in binders if b["start"] <= o]
            in_sec = [b for b in earlier if sec and section_of(b["start"]) == sec]
            if in_sec:
                choice, rule = in_sec[-1], "in-section"
            elif earlier:
                choice, rule = earlier[-1], "in-paper"
        s1 = _fields(m).get("bound", "") if m["kind"] == "symbol-grounded" else ""
        agree = None
        if s1 and choice is not None:
            a, b = _content(s1), _content(choice["type"])
            agree = bool(a and b and len(a & b) / len(a | b) >= 0.5) or (s1.strip() == choice["type"].strip())
        e = edges.setdefault(name, {"name": name, "binders": binders, "occurrences": []})
        e["occurrences"].append({"start": m["start"], "end": m["end"], "line": _line(starts, o),
                                 "binder": binders.index(choice) if choice is not None else None,
                                 "rule": rule, "s1-grounding": s1, "s1-agrees": agree})
    out = sorted(edges.values(), key=lambda e: -len(e["occurrences"]))
    return out


MACRO_DEF = re.compile(r"\\(?:re)?newcommand\s*\*?\s*(?:\{\\([A-Za-z@]+)\}|\\([A-Za-z@]+))"
                       r"\s*(?:\[(\d)\])?\s*(?:\[[^\]]*\])?\s*\{")
DEF_DEF = re.compile(r"\\def\s*\\([A-Za-z@]+)\s*(?:#\d)*\s*\{")


def _braced(text: str, open_at: int) -> str:
    """The balanced { ... } starting at open_at, without its braces."""
    depth, i = 0, open_at
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
        i += 1
    return ""


def macro_table(text: str) -> dict[str, dict]:
    r"""What each macro the paper defines says, chased through the macros it uses.

    A run records that \a is author-defined with role ID and where it was defined,
    which says nothing about what it means: \a is \alpha, \T is {\mathcal T}. The
    definitions are in the paper's own preamble, so this is a read, not a guess.
    """
    out: dict[str, dict] = {}
    for pattern in (MACRO_DEF, DEF_DEF):
        for m in pattern.finditer(text):
            name = next((g for g in m.groups()[:2] if g), None) if pattern is MACRO_DEF else m.group(1)
            if not name:
                continue
            body = _braced(text, m.end() - 1).strip()
            if body:
                out[name] = {"body": body, "line": _line(_line_starts(text), m.start()),
                             "takes-argument": "#" in body}
    for name, entry in out.items():
        body, seen = entry["body"], {name}
        for _ in range(4):
            nested = [x for x in re.findall(r"\\([A-Za-z@]+)", body)
                      if x in out and x not in seen and not out[x]["takes-argument"]]
            if not nested:
                break
            for x in nested:
                seen.add(x)
                rhs = out[x]["body"]
                body = re.sub(r"\\" + re.escape(x) + r"(?![A-Za-z])", lambda _m: rhs, body)
        entry["expands-to"] = body[:80]
    return out


def strategies(text: str, marks: list[dict]) -> dict:
    terms = defined_terms(text, marks)
    symbols = symbol_hypergraph(text, marks)
    occ = [o for e in symbols for o in e["occurrences"]]
    rules = {r: sum(o["rule"] == r for o in occ)
             for r in ("in-environment", "proved-statement", "in-section", "in-paper", "unbound")}
    judged = [o for o in occ if o["s1-agrees"] is not None]
    macros = macro_table(text)
    return {"schema": "markup-strategies/v1", "terms": terms, "symbols": symbols, "macros": macros,
            "summary": {"terms": len(terms), "term-uses": sum(len(t["uses"]) for t in terms),
                        "symbols": len(symbols), "occurrences": len(occ),
                        "binding-sites": sum(len(e["binders"]) for e in symbols), "rules": rules,
                        "macros": len(macros),
                        "s1-compared": len(judged), "s1-agrees": sum(o["s1-agrees"] for o in judged)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("marks", type=Path, nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--list", dest="ids", type=Path, help="file of paper ids, one per line")
    ap.add_argument("--marks", dest="marks_dir", type=Path, help="directory of S1 marks JSON")
    ap.add_argument("--out", dest="out_dir", type=Path, help="directory for per-paper strategies")
    a = ap.parse_args()
    if a.ids:
        if not (a.marks_dir and a.out_dir):
            ap.error("--list needs --marks and --out")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        ids = [x.strip() for x in a.ids.read_text().splitlines() if x.strip()]
        total = {"papers": 0, "missing": [], "occurrences": 0, "bound": 0, "s1-compared": 0, "s1-agrees": 0}
        for pid in ids:
            src = a.marks_dir / f"fable-{pid}-dp-emacs.json"
            if not src.is_file():
                total["missing"].append(pid)
                continue
            data = json.loads(src.read_text())
            result = strategies(data["text"], data["marks"])
            (a.out_dir / f"{pid}.strategies.json").write_text(json.dumps(result, ensure_ascii=False))
            su = result["summary"]
            total["papers"] += 1
            total["occurrences"] += su["occurrences"]
            total["bound"] += su["occurrences"] - su["rules"]["unbound"]
            total["s1-compared"] += su["s1-compared"]
            total["s1-agrees"] += su["s1-agrees"]
        print(json.dumps(total))
        return 1 if total["missing"] else 0
    if not a.marks:
        ap.error("give a marks file, or --list with --marks and --out")
    data = json.loads(a.marks.read_text())
    result = strategies(data["text"], data["marks"])
    if a.out:
        a.out.write_text(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
