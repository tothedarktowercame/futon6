#!/usr/bin/env python3
"""Corpus-scale definition debt report from the WARP concordance.

Find concept terms that are used across many papers and defined in no corpus
paper. Split them into true external debt (also absent from Lean/mathlib,
PlanetMath, and nLab) versus corpus-undefined terms that are already covered by
an external layer. Also report the inverse: terms that are both widely defined
and widely used inside the corpus.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONCORDANCE = ROOT / "data" / "warp" / "concordance.json"
DEFAULT_OUT = ROOT / "data" / "warp" / "corpus-debt.json"
DEFAULT_MATHLIB = ROOT / "data" / "mathlib-defs.json"
DEFAULT_NLAB = ROOT / "data" / "nlab-wiring" / "pages.json"
DEFAULT_PLANETMATH = Path("/home/joe/code/planetmath")
CHUNK = 1 << 20
MACRO_NOISE = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi",
    "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda",
    "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega", "ell", "hbar",
    "imath", "jmath", "aleph", "nabla", "partial", "infty",
    "frac", "cdot", "times", "otimes", "circ", "cap", "wedge", "prod",
    "oplus", "vee", "cup", "pm", "mp", "le", "ge", "leq", "geq", "neq",
    "sim", "cong", "equiv", "approx", "subset", "subseteq", "supset",
    "supseteq", "in", "notin", "ni", "mid", "propto", "to", "rightarrow",
    "longrightarrow", "mapsto", "xrightarrow", "hookrightarrow",
    "twoheadrightarrow", "Rightarrow", "leftarrow", "leftrightarrow",
    "overset", "underset", "stackrel", "sum", "coprod", "int", "oint",
    "lim", "colim", "varinjlim", "varprojlim", "langle", "rangle",
    "lfloor", "rfloor", "lceil", "rceil", "left", "right", "big", "Big",
    "bigg", "Bigg", "quad", "qquad", "ldots", "cdots", "vdots", "ddots",
    "dots", "hspace", "vspace", "smallskip", "medskip", "bigskip",
    "nonumber", "mathcal", "mathbb", "mathbf", "mathrm", "mathsf",
    "mathfrak", "mathtt", "mathit", "mathscr", "boldsymbol",
    "operatorname", "operatornamewithlimits", "text", "textrm", "textbf",
    "textit", "textsf", "texttt", "bm", "mbox", "hbox", "overline",
    "underline", "tilde", "widetilde", "hat", "widehat", "bar", "ast",
    "star", "bullet", "colon", "emptyset", "forall", "exists", "not",
    "displaystyle", "end", "begin", "phantom", "scalebox",
    "simeq", "bigoplus", "bigcup", "dashv", "sqcup", "scriptstyle",
    "boxtimes", "backslash", "bigcap", "varnothing", "bigvee", "vdash",
    "vcenter", "coloneqq", "scriptscriptstyle", "relax", "mkern",
    "longmapsto", "bigsqcup", "bigwedge", "perp", "Vert", "vert",
    "nrightarrow", "nleftarrow", "leftrightarrows", "rightleftarrows",
    "rightrightarrows", "leftleftarrows", "rightsquigarrow", "leadsto",
    "Longrightarrow", "Longleftarrow", "Leftrightarrow", "Longleftrightarrow",
    "uparrow", "downarrow", "updownarrow", "nearrow", "searrow", "swarrow",
    "nwarrow", "parallel", "nparallel", "preceq", "succeq", "prec", "succ",
    "ll", "gg", "llbracket", "rrbracket", "lbrace", "rbrace", "lvert",
    "rvert", "lVert", "rVert", "bmod", "pmod", "mod", "qquad",
    "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr", "Biggl", "Biggr",
    "leftroot", "uproot", "smash", "mathop", "limits", "nolimits",
    "vphantom", "hphantom", "kern", "hfill", "vfill", "cr", "qquad",
    "qquad", "thinspace", "negthinspace", "enspace", "enskip",
    "rtimes", "ltimes", "textstyle", "subsetneq", "bigotimes", "hskip",
    "leqslant", "geqslant", "noindent", "underbrace", "setbox", "vskip",
    "expandafter", "ifnum", "endcsname", "csname", "baselineskip",
    "penalty", "noexpand", "def", "let", "newcommand", "renewcommand",
    "newenvironment", "hbox", "vbox", "halign", "valign", "hrule",
    "vrule", "offinterlineskip", "openup", "advance", "count", "dimen",
    "skip", "box", "copy", "wd", "ht", "dp", "lower", "raise", "moveleft",
    "moveright", "leaders", "noalign", "omit", "span", "multispan",
    "futurelet", "aftergroup", "global", "outer", "long", "catcode",
    "char", "mathchar", "delimiter", "font", "fam", "ifx", "ifcat",
    "ifdim", "ifodd", "ifcase", "else", "fi", "or", "the", "number",
    "romannumeral", "string", "meaning", "jobname", "input", "include",
    "includegraphics", "hline", "cline", "multicolumn", "par", "parskip",
    "parindent", "medmuskip", "thinmuskip", "thickmuskip",
    "defeq", "sqcap", "cleaders", "immediate", "mathord", "thesection",
    "write", "unskip", "clubpenalty", "widowpenalty", "mathrel",
    "ifmmode", "sfcode", "uppercase", "title", "maketitle", "thanks",
    "address", "author", "date", "section", "subsection", "subsubsection",
    "paragraph", "item", "itemize", "enumerate", "description", "label",
    "ref", "eqref", "cite", "bibitem", "emph", "bf", "it", "sl", "rm",
    "cal", "mit", "tt", "sf", "footnote", "footnotemark", "footnotetext",
    "abstractname", "keywords", "footnoterule", "dotsc", "displaywidth",
    "tabskip", "rightharpoonup", "copyrightyear", "pitchfork", "blank",
    "chaptermark", "dotsb", "sectionmark", "sqsubseteq", "lowercase",
    "mathbin", "bibname", "refname", "contentsname", "listfigurename",
    "listtablename", "appendixname", "figurename", "tablename", "indexname",
    "today", "pagenumbering", "pagestyle", "thispagestyle", "markboth",
    "markright", "pageref", "bibliography", "bibliographystyle",
    "tableofcontents", "newpage", "clearpage", "pagebreak", "linebreak",
    "nobreak", "allowbreak", "protect", "ignorespaces", "leavevmode",
    "nsubseteq", "mathsurround", "boxplus", "subsectionmark", "thempfn",
    "multimap", "thefootnote", "endrefs", "NoCaseChange", "endabstract",
    "mldepf", "fancyplain", "headstyle", "thefnpage", "thmark",
    "cfoot", "chead", "lfoot", "lhead", "rfoot", "rhead",
    "runningauthor", "runningtitle", "endreferences", "relaxp",
    "cauthor", "dedication", "eaddress", "endaref", "mathbfdef",
    "mathcaldef", "mathfkdef", "mathopdef", "mathrmdef", "newfont",
    "thebibliography", "endthebibliography", "url", "href",
    "mathopsldef", "mathrsfsdef", "mathsfdef", "mathssbxdef",
    "mathzcdef", "tcite", "acute", "fontdimen", "everycr", "coloneq",
    "overbrace", "smallsetminus", "circledast", "leftskip", "bigstar",
    "unpenalty",
    "hsize", "ifvmode", "rightskip", "thick", "ddagger", "footnotesize",
    "ominus", "mapstochar", "bigcirc", "uplus", "everypar", "xleaders",
    "parfillskip", "postdisplaypenalty", "openout", "abovedisplayskip",
    "belowdisplayskip", "abovedisplayshortskip", "belowdisplayshortskip",
    "predisplaypenalty", "interlinepenalty", "clubpenalties",
    "hyphenpenalty", "leftharpoonup", "hangindent", "labelenumi",
    "scriptsize", "topskip", "thesubsection", "theenumiv", "theenumi",
    "email", "labelenumii", "spaceskip", "makelabel", "andify",
    "curraddr", "unvbox", "labelenumiii", "labelenumiv", "theenumii",
    "theenumiii", "leftmargin", "leftmargini", "leftmarginii",
    "leftmarginiii", "parsep", "itemsep", "topsep", "partopsep",
    "uppercasenonmath", "urladdr", "trivlist", "leqno",
    "fullwidthdisplay", "nxandlist", "thesubsubsection",
    "descriptionlabel", "sloppy", "keywordsname", "dedicatory",
    "translator", "labelitemi", "labelitemii", "labelitemiii",
    "labelitemiv",
    "listoffigures", "listoftables", "specialsection", "endtheindex",
    "indexspace", "subitem", "subsubitem", "xandlist", "centerdot",
    "ifhmode", "endtitlepage", "titlepage", "breve", "ifvoid",
    "unhbox", "smile",
    "printindex", "eqdef", "subsetneqq", "afterassignment",
    "lastpenalty", "supsetneq", "sslash", "emergencystretch",
    "bowtie", "relbar", "lineskip", "mathring", "preccurlyeq",
    "tikzfig",
    "boxslash", "eqqcolon", "bibliofont", "boxdot", "upharpoonright",
    "dotsm", "indent", "partname", "lastskip", "scriptfont",
    "shortmid", "widecheck", "mapsfrom",
    "sqsubset", "prevdepth", "leftmark", "rightmark", "lastbox",
    "mskip", "calclayout", "unlhd", "chardef", "displayindent",
    "endeqnarray", "frontmatter", "mainmatter", "backmatter", "vfuzz",
    "lsuper", "doteq", "unkern", "ifhbox", "hfuzz", "spacefactor",
    "cleardoublepage", "ifinner", "brokenpenalty", "floatingpenalty",
    "tocsection", "arraystretch", "limsup", "bibsetup", "partrunhead",
    "overfullrule", "catname", "relto", "cocolon", "everydisplay",
    "exhyphenpenalty", "botmark",
    "oslash", "baselinestretch", "rightleftharpoons",
    "rightharpoondown", "thicksim", "grave", "seename", "smaller",
    "floatpagefraction", "after", "abstractheadfont", "abstractfont",
    "smfandname",
    "textfont", "tocpart", "larger", "copyrightinfo",
    "copyrightholder", "nonbreakingspace", "altabstractname",
    "alttitle", "topmark", "translatedby", "bottomfraction",
    "dblfloatpagefraction", "dbltopfraction", "textfraction",
    "topfraction", "splitmaxdepth", "splittopskip", "suchthat",
    "biguplus", "dedicatoryfont", "impliedby", "altkeywords",
    "altkeywordsname", "isopil", "datename", "hfilneg", "cdotp",
    "labelstyle", "firstaddress", "otheraddress", "pagespan",
    "issueinfo", "bigodot", "smfbyname", "iffalse", "fromto",
    "widebar",
    "frown", "theaddress", "varsubsetneq", "newtheorem",
    "boxminus", "IEEEeqnarraymathstyle", "thmheadnl", "swappedhead",
    "theoremstyle", "endsplit", "newtheoremstyle", "nonslanted",
    "circledcirc", "discard", "xfrom", "goesto", "liminf",
    "qedhere", "roundNbox", "equivto", "mathqed",
    "qedsymbol", "objectstyle", "newswitch", "altucnm", "commby",
    "tocappendix", "popQED", "pushQED", "setTrue",
    "DDelta",
}

MACRO_CONCEPT_ALIASES = {
    "hocolim": "homotopy colimit",
    "holim": "homotopy limit",
    "FPdim": "Frobenius Perron dimension",
    "fpdim": "Frobenius Perron dimension",
    "gldim": "global dimension",
    "twocat": "2 category",
    "cocart": "cocartesian fibration",
    "oplax": "oplax functor",
    "dgcat": "dg category",
    "dgCat": "dg category",
    "liminj": "filtered colimit",
    "Deltaop": "simplex category",
    "ModR": "module category",
    "Cobar": "cobar construction",
    "pretr": "pretriangulated category",
    "rdual": "right dual",
    "inprod": "inner product",
    "smcat": "symmetric monoidal category",
    "coend": "coend",
    "Coend": "coend",
    "Operad": "operad",
    "Cat": "category",
    "Set": "set",
}
NON_CONCEPT_DEBT_WORDS = {
    "all", "every", "some", "forall", "exists", "quantified", "variable",
    "variables", "script", "follow", "following", "where", "when", "then",
    "there", "such", "let", "rightarrow", "leftarrow", "otimes", "oplus",
    "times", "maps", "map", "object", "objects", "element", "elements",
    "family", "between", "single", "two", "structure", "equivalence",
    "only", "end", "equation", "function", "functions", "set", "sets",
    "pair", "pairs", "fact",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


concept_shuttle = load_module("concept_shuttle", ROOT / "scripts" / "concept_shuttle.py")
concept_authority_mod = load_module("concept_authority", ROOT / "scripts" / "concept_authority.py")


def norm_label(value: str) -> str:
    value = re.sub(r"\$[^$]*\$", " ", value)
    value = re.sub(r"\\[A-Za-z@]+", " ", value)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_label(value))


def camel(value: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[\s-]+", value.strip()))


def is_reportable_term(term: str) -> bool:
    term = report_term(term)
    if not term or "\\" in term or "$" in term:
        return False
    label = norm_label(term)
    if not label:
        return False
    words = label.split()
    if len(words) == 1 and len(words[0]) < 5:
        return False
    if len(words) > 8:
        return False
    if not any(ch.isalpha() for ch in label):
        return False
    if re.search(r"\b(?:tex|mathcal|mathrm|phantom|hspace|vspace)\b", label):
        return False
    return True


def is_macro_origin(term: str) -> bool:
    return term.strip().startswith("\\")


def is_noise_concept_label(term: str) -> bool:
    label = norm_label(term)
    words = label.split()
    if not words:
        return True
    if any(re.fullmatch(r"[A-Za-z]", word) for word in words):
        return True
    if any(word in NON_CONCEPT_DEBT_WORDS for word in words):
        return True
    if len(words) == 1 and words[0] in {"map", "set", "class", "case"}:
        return True
    return False


def concept_debt_label(original_term: str, term: str, authority) -> str | None:
    if is_macro_origin(original_term):
        name = original_term.strip()[1:]
        alias = MACRO_CONCEPT_ALIASES.get(name) or MACRO_CONCEPT_ALIASES.get(name.lower())
        if not alias:
            return None
        if is_noise_concept_label(alias):
            return None
        return alias
    if not is_reportable_term(term):
        return None
    if is_noise_concept_label(term):
        return None
    label = norm_label(term)
    words = label.split()
    if len(words) >= 2:
        return term
    hit = authority.resolve(term) if authority is not None else None
    if not hit:
        return None
    target = str(hit.get("term") or hit.get("target") or "")
    if not target:
        return None
    return term if re.search(r"[A-Za-z]", target) else None


def split_macro_name(name: str) -> str:
    if name.isupper():
        return name
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return spaced.replace("_", " ")


def report_term(term: str) -> str | None:
    if term.startswith("\\"):
        name = term[1:]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            return None
        if name in MACRO_NOISE or name.lower() in MACRO_NOISE:
            return None
        if re.match(r"^(bb|cal|frak|scr|sf|rm|bf|it)[A-Z]$", name):
            return None
        if re.match(r"^(mc|mcal|bb|bo|b|u|s|Pr|i|c|l)[A-Z].*", name):
            return None
        if name.startswith("bbone"):
            return None
        if re.match(r"^(cat|Cat|obj|Obj|mor|Mor)[A-Z]$", name):
            return None
        if len(name) < 3 and name not in {"Ab", "Ch", "Grp", "Mod", "Set", "Top"}:
            return None
        return split_macro_name(name)
    return term


def iter_concordance_terms(path: Path) -> Iterator[tuple[str, list[dict]]]:
    """Stream ``terms`` members without loading the whole concordance object."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        buf = ""
        eof = False
        pos = 0

        def fill() -> bool:
            nonlocal buf, eof
            if eof:
                return False
            chunk = fh.read(CHUNK)
            if not chunk:
                eof = True
                return False
            buf += chunk
            return True

        def ensure() -> None:
            if pos >= len(buf) and not eof:
                fill()

        while '"terms"' not in buf:
            if not fill():
                raise ValueError(f"no terms object found in {path}")
        pos = buf.index('"terms"')
        while "{" not in buf[pos:]:
            if not fill():
                raise ValueError(f"unterminated terms object in {path}")
        pos = buf.index("{", pos) + 1

        def skip_ws() -> None:
            nonlocal pos
            while True:
                ensure()
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf) or eof:
                    return

        def raw_decode():
            nonlocal buf, pos, eof
            while True:
                try:
                    value, end = decoder.raw_decode(buf, pos)
                    pos = end
                    return value
                except json.JSONDecodeError:
                    if not fill():
                        raise

        while True:
            skip_ws()
            ensure()
            if pos < len(buf) and buf[pos] == "}":
                return
            if pos < len(buf) and buf[pos] == ",":
                pos += 1
                skip_ws()
            term = raw_decode()
            skip_ws()
            ensure()
            if pos >= len(buf) or buf[pos] != ":":
                raise ValueError(f"expected ':' after term {term!r}")
            pos += 1
            skip_ws()
            rows = raw_decode()
            yield term, rows
            if pos > CHUNK:
                buf = buf[pos:]
                pos = 0


def load_mathlib_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [str(row.get("name", "")) for row in json.loads(path.read_text(encoding="utf-8"))]


def load_planetmath_index(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for f in root.rglob("*.tex"):
        stem = f.stem.split("-", 1)[-1]
        key = compact(stem)
        if key and key not in out:
            out[key] = str(f.relative_to(root))
    return out


def load_nlab_index(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for page in json.loads(path.read_text(encoding="utf-8")):
        name = str(page.get("page_name", ""))
        key = compact(name)
        if key and key not in out:
            out[key] = str(page.get("page_id", ""))
        for link in page.get("typed_links") or []:
            content = link.get("hx/content") or {}
            target = str(content.get("target_name") or "")
            tkey = compact(target)
            if tkey and tkey not in out:
                out[tkey] = str(link.get("hx/target") or "")
    return out


def planetmath_hit(term: str, pm_index: dict[str, str]) -> str | None:
    cam = compact(camel(term))
    return pm_index.get(cam)


def nlab_hit(term: str, nlab_index: dict[str, str]) -> str | None:
    key = compact(term)
    if not key:
        return None
    return nlab_index.get(key)


def coverage(term: str, mathlib_names: list[str], pm_index: dict[str, str], nlab_index: dict[str, str]) -> dict:
    lean = concept_shuttle.in_mathlib(term, mathlib_names)
    pm = planetmath_hit(term, pm_index)
    nlab = nlab_hit(term, nlab_index)
    return {"lean": lean, "planetmath": pm, "nlab": nlab, "covered": bool(lean or pm or nlab)}


def summarize_rows(rows: list[dict]) -> dict:
    used_papers: set[str] = set()
    defined_papers: set[str] = set()
    used_count = 0
    defined_count = 0
    for row in rows:
        role = row.get("role")
        paper = str(row.get("paper", ""))
        count = int(row.get("count") or 0)
        if role == "used":
            used_papers.add(paper)
            used_count += count
        elif role == "defined":
            defined_papers.add(paper)
            defined_count += count
    return {
        "used_papers": used_papers,
        "defined_papers": defined_papers,
        "used_count": used_count,
        "defined_count": defined_count,
    }


def row_for(term: str, original_term: str, summary: dict, cov: dict) -> dict:
    used_papers = sorted(summary["used_papers"])
    defined_papers = sorted(summary["defined_papers"])
    return {
        "term": term,
        "concordance_term": original_term,
        "used_papers": len(used_papers),
        "defined_papers": len(defined_papers),
        "used_count": summary["used_count"],
        "defined_count": summary["defined_count"],
        "sample_used_papers": used_papers[:10],
        "sample_defined_papers": defined_papers[:10],
        "coverage": cov,
    }


def add_frontier_acc(acc: dict[str, dict], term: str, original_term: str, summary: dict, cov: dict) -> None:
    row = acc.setdefault(
        term,
        {
            "term": term,
            "concordance_terms": set(),
            "used_papers": set(),
            "defined_papers": set(),
            "used_count": 0,
            "defined_count": 0,
            "coverage": cov,
        },
    )
    row["concordance_terms"].add(original_term)
    row["used_papers"].update(summary["used_papers"])
    row["defined_papers"].update(summary["defined_papers"])
    row["used_count"] += summary["used_count"]
    row["defined_count"] += summary["defined_count"]
    if not row["coverage"].get("covered") and cov.get("covered"):
        row["coverage"] = cov


def debt_row_from_acc(row: dict) -> dict:
    used_papers = sorted(row["used_papers"])
    defined_papers = sorted(row["defined_papers"])
    terms = sorted(row["concordance_terms"])
    return {
        "term": row["term"],
        "concordance_term": terms[0] if len(terms) == 1 else None,
        "concordance_terms": terms,
        "used_papers": len(used_papers),
        "defined_papers": len(defined_papers),
        "used_count": row["used_count"],
        "defined_count": row["defined_count"],
        "sample_used_papers": used_papers[:10],
        "sample_defined_papers": defined_papers[:10],
        "coverage": row["coverage"],
    }


def build(args: argparse.Namespace) -> dict:
    start = time.time()
    mathlib_names = load_mathlib_names(args.mathlib)
    pm_index = load_planetmath_index(args.planetmath)
    nlab_index = load_nlab_index(args.nlab)
    authority = concept_authority_mod.ConceptAuthority()
    external_debt_acc: dict[str, dict] = {}
    covered_undefined_acc: dict[str, dict] = {}
    core: list[dict] = []
    stats = {
        "terms_seen": 0,
        "reportable_terms": 0,
        "covered_reportable_terms": 0,
        "external_debt_candidates": 0,
        "externally_covered_corpus_undefined_candidates": 0,
        "corpus_undefined_concordance_candidates": 0,
        "corpus_undefined_unique_candidates": 0,
        "core_candidates": 0,
        "mathlib_defs": len(mathlib_names),
        "planetmath_terms": len(pm_index),
        "nlab_terms": len(nlab_index),
    }

    for term, rows in iter_concordance_terms(args.concordance):
        stats["terms_seen"] += 1
        term_label = report_term(term)
        if not term_label or not is_reportable_term(term_label):
            continue
        summary = summarize_rows(rows)
        if len(summary["used_papers"]) < args.min_used_papers:
            continue
        stats["reportable_terms"] += 1
        debt_label = concept_debt_label(term, term_label, authority)
        cov = coverage(debt_label or term_label, mathlib_names, pm_index, nlab_index)
        if cov["covered"]:
            stats["covered_reportable_terms"] += 1
        if (
            summary["used_papers"]
            and not summary["defined_papers"]
            and debt_label
        ):
            stats["corpus_undefined_concordance_candidates"] += 1
            if cov["covered"]:
                add_frontier_acc(covered_undefined_acc, debt_label, term, summary, cov)
            else:
                add_frontier_acc(external_debt_acc, debt_label, term, summary, cov)
        if summary["used_papers"] and summary["defined_papers"]:
            stats["core_candidates"] += 1
            core.append(row_for(term_label, term, summary, cov))
        if args.progress and stats["terms_seen"] % args.progress == 0:
            print(
                f"[warp-debt] terms={stats['terms_seen']} external-debt={len(external_debt_acc)} covered-undefined={len(covered_undefined_acc)} core={len(core)}",
                file=sys.stderr,
                flush=True,
            )

    external_debt = [debt_row_from_acc(row) for row in external_debt_acc.values()]
    covered_undefined = [debt_row_from_acc(row) for row in covered_undefined_acc.values()]
    external_debt.sort(key=lambda r: (-r["used_papers"], -r["used_count"], r["term"].lower()))
    covered_undefined.sort(key=lambda r: (-r["used_papers"], -r["used_count"], r["term"].lower()))
    core.sort(key=lambda r: (-min(r["used_papers"], r["defined_papers"]), -r["used_papers"], -r["defined_papers"], r["term"].lower()))
    stats["external_debt_candidates"] = len(external_debt)
    stats["externally_covered_corpus_undefined_candidates"] = len(covered_undefined)
    stats["corpus_undefined_unique_candidates"] = len(external_debt) + len(covered_undefined)
    stats["elapsed_sec"] = round(time.time() - start, 3)
    stats["external_debt_returned"] = min(args.limit, len(external_debt))
    stats["externally_covered_corpus_undefined_returned"] = min(args.limit, len(covered_undefined))
    stats["core_returned"] = min(args.limit, len(core))
    return {
        "schema": "warp-corpus-debt-v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "concordance": str(args.concordance),
            "mathlib": str(args.mathlib),
            "planetmath": str(args.planetmath),
            "nlab": str(args.nlab),
        },
        "stats": stats,
        "external_debt_frontier": external_debt[: args.limit],
        "externally_covered_corpus_undefined": covered_undefined[: args.limit],
        "well_covered_core": core[: args.limit],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concordance", type=Path, default=DEFAULT_CONCORDANCE)
    ap.add_argument("--mathlib", type=Path, default=DEFAULT_MATHLIB)
    ap.add_argument("--planetmath", type=Path, default=DEFAULT_PLANETMATH)
    ap.add_argument("--nlab", type=Path, default=DEFAULT_NLAB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--min-used-papers", type=int, default=5)
    ap.add_argument("--progress", type=int, default=10000)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = build(args)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    print("\nTop external debt:")
    for row in result["external_debt_frontier"][:20]:
        print(f"{row['used_papers']:5d} papers  {row['used_count']:7d} uses  {row['term']}")
    print("\nTop externally covered corpus-undefined:")
    for row in result["externally_covered_corpus_undefined"][:20]:
        print(f"{row['used_papers']:5d} papers  {row['used_count']:7d} uses  {row['term']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
