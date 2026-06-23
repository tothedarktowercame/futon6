#!/usr/bin/env python3
"""Tier-3 of the 'improve as we run' spine: STRUCTURAL normalization of definitions.

Concepts aren't just labels — we build per-paper SFC :structure trees (sfc_def_structure).
Tier 1 (label) and Tier 2 (move cluster) are string/phrase normalization; definitions need
STRUCTURAL normalization: two definitions are the same shape modulo

  - α-renaming      (variables/symbols are positional: G,g,H,φ,e ≡ S,y,T,ψ,0)
  - operator-canon  (synonymous constructors map to one canonical operator)
  - grounding       (literals/constants are holes, like variables)

keeping only the STRUCTURAL operators (= ∈ ∀ ∃ conditional-set → ↦ ⊆ ≅ ∘ × ⊗ : …). Two
:structure trees with the same canonical key are the same definitional shape → merge into
one canonical encyclopedia entry instead of N near-duplicates.

  canon("(= G (conditional-set (∈ g H) (= (* φ g) e)))") ==
  canon("(= S (conditional-set (∈ y T) (= (* ψ y) 0)))")   # subset-by-equation

  futon6/.venv/bin/python scripts/sfc_struct_canon.py --self-test
  echo '<formula>' | bb scripts/sfc_def_structure.bb - | sfc_struct_canon.py -   # one
"""
import argparse
import re
import sys

# structural operators/constructors — kept verbatim; everything else is a hole (α-renamed).
OPERATORS = {
    "=", "∈", "∉", "⊆", "⊂", "≅", "≃", "≈", "≤", "≥", "<", ">", "→", "↦", "⟶", "⇒",
    "∀", "∃", "forall", "exists", "λ", "∘", "×", "⊗", "⊕", "∧", "∨", "¬", "∩", "∪",
    ":", "*", "conditional-set", "set", "tuple", "pair", "apply", "and", "or", "not",
    ":hole", "⊢", "↪", "≔", ":=",
}


def parse(s):
    toks = re.findall(r"\(|\)|[^()\s]+", s.strip())
    pos = [0]

    def walk():
        if pos[0] >= len(toks):
            return None
        t = toks[pos[0]]; pos[0] += 1
        if t == "(":
            node = []
            while pos[0] < len(toks) and toks[pos[0]] != ")":
                node.append(walk())
            pos[0] += 1  # skip ')'
            return tuple(node)
        return t

    return walk()


def canon(tree, mapping):
    if isinstance(tree, tuple):
        return tuple(canon(x, mapping) for x in tree)
    if tree in OPERATORS:
        return tree
    return mapping.setdefault(tree, f"v{len(mapping)}")   # α-rename hole/variable


def _ser(tree):
    return "(" + " ".join(_ser(x) for x in tree) + ")" if isinstance(tree, tuple) else str(tree)


def canon_key(structure_str):
    """:structure s-expr -> canonical structural key (α-renamed, operators kept)."""
    tree = parse(structure_str)
    return _ser(canon(tree, {})) if tree is not None else None


def extract_structure(edn_text):
    m = re.search(r":structure\s+(\(.*?\))\s*,?\s*:ungrounded", edn_text, re.S)
    if not m:
        m = re.search(r":structure\s+(\(.*\))", edn_text, re.S)
    return m.group(1).strip() if m else None


SELFTEST = [
    ("(= G (conditional-set (∈ g H) (= (* φ g) e)))", "subset-by-equation"),
    ("(= S (conditional-set (∈ y T) (= (* ψ y) 0)))", "subset-by-equation (renamed)"),
    ("(= X (conditional-set (∈ x A) (∈ (forall y) BPxy)))", "subset-by-∀-predicate"),
    ("(: f (→ A B))", "a typed map"),
]


def batch(formulae):
    """Run sfc_def_structure on each formula, canonicalize, cluster by structural key.
    Returns {key: [formulae]} — the canonical definition shapes."""
    import os
    import subprocess
    from collections import defaultdict
    bb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sfc_def_structure.bb")
    groups = defaultdict(list)
    for f in formulae:
        try:
            out = subprocess.run(["bb", bb, "-"], input=f, capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        st = extract_structure(out)
        k = canon_key(st) if st else None
        if k and k != "()":
            groups[k].append(f)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=None, help="EDN file (or - for stdin), else --self-test")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--formulae", help="file of definition formulae (one per line) -> normalization ratio")
    a = ap.parse_args()
    if a.formulae:
        forms = [l.strip() for l in open(a.formulae) if l.strip()]
        groups = batch(forms)
        nf = sum(len(v) for v in groups.values())
        print(f"=== structural normalization: {nf} definition formulae -> {len(groups)} canonical shapes "
              f"(ratio {nf / max(1, len(groups)):.2f}×) ===\n")
        for k, fs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
            print(f"×{len(fs)}  {k}")
            for f in fs[:2]:
                print(f"      ← {f[:64]}")
        return
    if a.self_test or not a.input:
        from collections import defaultdict
        groups = defaultdict(list)
        for s, label in SELFTEST:
            groups[canon_key(s)].append(label)
        print(f"=== structural canonicalization: {len(SELFTEST)} defs -> {len(groups)} shapes ===\n")
        for k, labels in groups.items():
            print(f"shape {k}")
            for l in labels:
                print(f"   ← {l}")
        merged = [g for g in groups.values() if len(g) > 1]
        print(f"\nmerges: {len(merged)} (e.g. the two subset-by-equation defs collapse to ONE shape)")
        return
    text = sys.stdin.read() if a.input == "-" else open(a.input).read()
    st = extract_structure(text) or text
    print(canon_key(st))


if __name__ == "__main__":
    main()
