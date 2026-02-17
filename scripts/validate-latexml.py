#!/usr/bin/env python3
"""Validate futon6 LaTeX→s-exp parser against LaTeXML Content MathML.

Designed as a Codex task: runs headless, produces a JSON report.

Prerequisites:
    sudo apt install latexml   # or: cpanm LaTeXML
    pip install lxml            # for parsing MathML

Usage:
    # Full validation against a thread hypergraph:
    python scripts/validate-latexml.py data/first-proof/thread-633512-hypergraph.json

    # Quick validation with built-in corpus only:
    python scripts/validate-latexml.py --builtin-only

    # Custom LaTeX expression:
    python scripts/validate-latexml.py --expr '\\frac{a}{b}'

Output:
    Writes validate-latexml-report.json with per-expression comparison results.
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from futon6.latex_sexp import parse as sexp_parse


# ---------------------------------------------------------------------------
# Built-in corpus: expressions spanning common LaTeX math constructs
# ---------------------------------------------------------------------------

BUILTIN_CORPUS = [
    # --- Identifiers and Greek ---
    (r"x", "single variable"),
    (r"\alpha", "Greek letter"),
    (r"\Gamma", "uppercase Greek"),

    # --- Binary operators ---
    (r"a + b", "addition"),
    (r"a - b", "subtraction"),
    (r"a \cdot b", "multiplication (cdot)"),
    (r"a \times b", "multiplication (times)"),

    # --- Relations ---
    (r"a = b", "equality"),
    (r"a \neq b", "inequality"),
    (r"a \leq b", "less-or-equal"),
    (r"x \in S", "set membership"),
    (r"A \subset B", "subset"),
    (r"A \subseteq B", "subset-or-equal"),

    # --- Fractions ---
    (r"\frac{a}{b}", "fraction"),
    (r"\frac{1}{n+1}", "fraction with sum in denom"),
    (r"\frac{x^2 + 1}{x - 1}", "fraction with polynomial"),

    # --- Sub/superscripts ---
    (r"x_i", "subscript"),
    (r"x^2", "superscript"),
    (r"x_i^2", "both sub and super"),
    (r"a_{ij}", "subscript group"),

    # --- Functions ---
    (r"f(x)", "function application"),
    (r"f(x, y)", "multi-arg function"),
    (r"\sin(x)", "named function"),

    # --- Arrows and morphisms ---
    (r"f : A \to B", "typed morphism"),
    (r"A \rightarrow B", "arrow"),
    (r"f \circ g", "composition"),
    (r"A \hookrightarrow B", "inclusion"),
    (r"f \mapsto g", "mapsto"),

    # --- Logic ---
    (r"\forall x", "universal quantifier"),
    (r"\exists x", "existential quantifier"),
    (r"A \land B", "conjunction"),
    (r"A \lor B", "disjunction"),

    # --- Set operations ---
    (r"A \cup B", "union"),
    (r"A \cap B", "intersection"),

    # --- Decorations ---
    (r"\bar{x}", "bar accent"),
    (r"\hat{x}", "hat accent"),
    (r"\tilde{x}", "tilde accent"),
    (r"\vec{v}", "vector"),
    (r"\overline{AB}", "overline"),

    # --- Font commands ---
    (r"\mathcal{C}", "mathcal"),
    (r"\mathbb{R}", "mathbb"),
    (r"\mathsf{Path}", "mathsf"),
    (r"\mathrm{id}", "mathrm"),

    # --- Delimiters ---
    (r"\left( x + y \right)", "sized parens"),
    (r"(a, b, c)", "tuple"),

    # --- Compound expressions (from thread #633512) ---
    (r"\Gamma=(V,E,s,t)", "graph definition"),
    (r"s(X(e))=X(s(e))", "functoriality"),
    (r"e : v \to w", "edge typing"),
    (r"X(\gamma):=\mathrm{id}_{X(v)}", "path definition"),
    (r"X(\gamma) := X(\beta) \circ X(e)", "path induction"),
    (r"\mathsf{Path}(\Gamma) \to \mathcal{C}", "functor type"),
    (r"D:\mathcal{I}\rightarrow\mathcal{A}", "diagram functor"),
]


# ---------------------------------------------------------------------------
# LaTeXML interface
# ---------------------------------------------------------------------------

def check_latexml() -> bool:
    """Check if latexmlmath is available."""
    try:
        result = subprocess.run(
            ["latexmlmath", "--VERSION"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def latexml_parse(latex: str, fmt: str = "cmml") -> str | None:
    """Parse a LaTeX expression with latexmlmath.

    Returns Content MathML (cmml) or XMath (xmath) as a string,
    or None on failure.
    """
    flag = f"--{fmt}=-"
    try:
        result = subprocess.run(
            ["latexmlmath", flag, "--", latex],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ---------------------------------------------------------------------------
# Content MathML → s-expression conversion
# ---------------------------------------------------------------------------

# MathML namespace
MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# Content MathML operator mapping to our s-exp operators
CMML_OP_MAP = {
    "plus": "+", "minus": "-", "times": "×", "divide": "/",
    "eq": "=", "neq": "≠", "leq": "≤", "geq": "≥",
    "lt": "<", "gt": ">",
    "in": "∈", "notin": "∉", "subset": "⊂", "prsubset": "⊂",
    "union": "∪", "intersect": "∩",
    "and": "∧", "or": "∨", "not": "¬",
    "forall": "∀", "exists": "∃",
    "compose": "∘",
    "sin": "sin", "cos": "cos", "tan": "tan", "log": "log",
    "exp": "exp",
}


def cmml_to_sexp(xml_str: str) -> str | None:
    """Convert Content MathML XML to an s-expression string.

    Returns None if the XML can't be parsed or has no content branch.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    # Navigate to the content MathML root
    content = _find_content_root(root)
    if content is None:
        return None

    return _cmml_node_to_sexp(content)


def _find_content_root(el: ET.Element) -> ET.Element | None:
    """Find the Content MathML root in a MathML tree.

    Handles both direct content markup and annotation-xml inside semantics.
    """
    tag = _strip_ns(el.tag)

    # Direct content element
    if tag in ("apply", "ci", "cn", "csymbol", "cerror", "bind"):
        return el

    # Inside <math>
    if tag == "math":
        children = list(el)
        if len(children) == 1:
            return _find_content_root(children[0])
        # Check for <semantics> wrapper
        for child in children:
            if _strip_ns(child.tag) == "semantics":
                return _find_content_root(child)
        # Fallback: first child
        if children:
            return _find_content_root(children[0])

    # Inside <semantics>: look for annotation-xml with content encoding
    if tag == "semantics":
        for child in el:
            ct = _strip_ns(child.tag)
            if ct == "annotation-xml":
                enc = child.get("encoding", "")
                if "MathML-Content" in enc or "content" in enc.lower():
                    children = list(child)
                    if children:
                        return children[0]
            # Direct content child
            if ct in ("apply", "ci", "cn", "csymbol", "bind"):
                return child
        # Fallback: first child
        children = list(el)
        if children:
            return _find_content_root(children[0])

    return el if tag in ("apply", "ci", "cn") else None


def _cmml_node_to_sexp(el: ET.Element) -> str:
    """Recursively convert a Content MathML element to s-expression."""
    tag = _strip_ns(el.tag)

    if tag == "ci":
        # Content identifier
        text = (el.text or "").strip()
        return text if text else "?"

    if tag == "cn":
        # Content number
        text = (el.text or "").strip()
        return text if text else "0"

    if tag == "csymbol":
        text = (el.text or "").strip()
        return CMML_OP_MAP.get(text, text)

    if tag == "apply":
        children = list(el)
        if not children:
            return "?"
        op_el = children[0]
        op = _cmml_op(op_el)
        args = [_cmml_node_to_sexp(c) for c in children[1:]]
        if not args:
            return op
        return f"({op} {' '.join(args)})"

    if tag == "bind":
        children = list(el)
        if not children:
            return "?"
        op = _cmml_op(children[0])
        args = [_cmml_node_to_sexp(c) for c in children[1:]]
        return f"({op} {' '.join(args)})"

    if tag == "bvar":
        children = list(el)
        if children:
            return _cmml_node_to_sexp(children[0])
        return "?"

    # Fallback
    text = (el.text or "").strip()
    return text if text else tag


def _cmml_op(el: ET.Element) -> str:
    """Extract operator name from a Content MathML element."""
    tag = _strip_ns(el.tag)
    if tag in CMML_OP_MAP:
        return CMML_OP_MAP[tag]
    if tag in ("ci", "csymbol"):
        text = (el.text or "").strip()
        return CMML_OP_MAP.get(text, text)
    return tag


def _strip_ns(tag: str) -> str:
    """Strip XML namespace from a tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


# ---------------------------------------------------------------------------
# Structural comparison
# ---------------------------------------------------------------------------

def compare_parses(our_sexp: str, latexml_sexp: str) -> dict:
    """Compare two s-expression parses structurally.

    Returns a dict with:
      - 'match': bool (exact string match)
      - 'structural': bool (same tree structure, possibly different names)
      - 'our': str
      - 'latexml': str
    """
    exact = our_sexp == latexml_sexp
    # Structural: same nesting depth and arity
    structural = _same_structure(our_sexp, latexml_sexp)
    return {
        'match': exact,
        'structural': structural,
        'our': our_sexp,
        'latexml': latexml_sexp,
    }


def _same_structure(a: str, b: str) -> bool:
    """Check if two s-expressions have the same tree skeleton."""
    def skeleton(s: str) -> str:
        """Replace all atoms with _ to get just the structure."""
        import re
        # Replace anything that's not parens or spaces with _
        return re.sub(r'[^()\s]+', '_', s)
    return skeleton(a) == skeleton(b)


# ---------------------------------------------------------------------------
# Corpus extraction
# ---------------------------------------------------------------------------

def extract_corpus_from_hypergraph(hg_path: str) -> list[tuple[str, str]]:
    """Extract (latex, description) pairs from a hypergraph JSON file."""
    with open(hg_path) as f:
        hg = json.load(f)

    corpus = []
    for node in hg.get("nodes", []):
        if node["type"] == "expression":
            latex = node["attrs"].get("latex", "")
            if latex and len(latex) > 1:
                desc = f"from {node['id']}"
                corpus.append((latex, desc))
    return corpus


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

def run_validation(corpus: list[tuple[str, str]],
                   use_latexml: bool = True) -> dict:
    """Run the full validation suite.

    Returns a report dict with results for each expression.
    """
    results = []
    stats = {
        'total': len(corpus),
        'our_parsed': 0,
        'latexml_parsed': 0,
        'both_parsed': 0,
        'exact_match': 0,
        'structural_match': 0,
        'our_only': 0,
        'latexml_only': 0,
        'neither': 0,
    }

    for latex, desc in corpus:
        entry = {
            'latex': latex,
            'description': desc,
            'our_sexp': None,
            'latexml_cmml': None,
            'latexml_sexp': None,
            'comparison': None,
        }

        # Our parser
        try:
            our = sexp_parse(latex)
            # Check if it's a fallback (quoted string)
            if our.startswith('"') and our.endswith('"'):
                our = None
            entry['our_sexp'] = our
        except Exception as e:
            entry['our_error'] = str(e)

        # LaTeXML parser
        if use_latexml:
            cmml = latexml_parse(latex)
            entry['latexml_cmml'] = cmml
            if cmml:
                latexml_sexp = cmml_to_sexp(cmml)
                entry['latexml_sexp'] = latexml_sexp

        # Statistics
        our_ok = entry['our_sexp'] is not None
        lml_ok = entry.get('latexml_sexp') is not None

        if our_ok:
            stats['our_parsed'] += 1
        if lml_ok:
            stats['latexml_parsed'] += 1

        if our_ok and lml_ok:
            stats['both_parsed'] += 1
            comp = compare_parses(entry['our_sexp'], entry['latexml_sexp'])
            entry['comparison'] = comp
            if comp['match']:
                stats['exact_match'] += 1
            if comp['structural']:
                stats['structural_match'] += 1
        elif our_ok and not lml_ok:
            stats['our_only'] += 1
        elif lml_ok and not our_ok:
            stats['latexml_only'] += 1
        else:
            stats['neither'] += 1

        results.append(entry)

    return {
        'stats': stats,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate futon6 LaTeX→s-exp parser against LaTeXML")
    parser.add_argument("hypergraph", nargs="?", default=None,
                        help="Path to thread hypergraph JSON (for corpus extraction)")
    parser.add_argument("--builtin-only", action="store_true",
                        help="Only use built-in corpus, skip hypergraph")
    parser.add_argument("--expr", default=None,
                        help="Validate a single LaTeX expression")
    parser.add_argument("--no-latexml", action="store_true",
                        help="Skip LaTeXML comparison (our parser only)")
    parser.add_argument("--output", default="validate-latexml-report.json",
                        help="Output report path")
    args = parser.parse_args()

    # Check LaTeXML
    use_latexml = not args.no_latexml
    if use_latexml and not check_latexml():
        print("WARNING: latexmlmath not found. Install with: sudo apt install latexml")
        print("         Running with --no-latexml (our parser only)")
        use_latexml = False

    # Build corpus
    corpus = list(BUILTIN_CORPUS)

    if args.expr:
        corpus = [(args.expr, "command-line expression")]
    elif not args.builtin_only and args.hypergraph:
        hg_corpus = extract_corpus_from_hypergraph(args.hypergraph)
        corpus.extend(hg_corpus)
        print(f"Corpus: {len(BUILTIN_CORPUS)} built-in + "
              f"{len(hg_corpus)} from hypergraph = {len(corpus)} total")

    # Run
    print(f"\nValidating {len(corpus)} expressions "
          f"(LaTeXML: {'yes' if use_latexml else 'no'})...\n")

    report = run_validation(corpus, use_latexml=use_latexml)
    stats = report['stats']

    # Print summary
    print(f"Results:")
    print(f"  Total expressions:    {stats['total']}")
    print(f"  Our parser parsed:    {stats['our_parsed']}")
    if use_latexml:
        print(f"  LaTeXML parsed:       {stats['latexml_parsed']}")
        print(f"  Both parsed:          {stats['both_parsed']}")
        print(f"  Exact match:          {stats['exact_match']}")
        print(f"  Structural match:     {stats['structural_match']}")
        print(f"  Our-only (superset):  {stats['our_only']}")
        print(f"  LaTeXML-only (gap):   {stats['latexml_only']}")
        print(f"  Neither parsed:       {stats['neither']}")

        # Show LaTeXML-only cases (our gaps)
        gaps = [r for r in report['results']
                if r.get('latexml_sexp') and not r.get('our_sexp')]
        if gaps:
            print(f"\n  === LaTeXML-only (our gaps) ===")
            for g in gaps[:10]:
                print(f"    {g['latex'][:60]}")
                print(f"      LaTeXML: {g['latexml_sexp']}")

        # Show disagreements
        disagree = [r for r in report['results']
                    if r.get('comparison') and not r['comparison']['structural']]
        if disagree:
            print(f"\n  === Structural disagreements ===")
            for d in disagree[:10]:
                print(f"    {d['latex'][:60]}")
                print(f"      Ours:    {d['comparison']['our']}")
                print(f"      LaTeXML: {d['comparison']['latexml']}")

    # Write report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
