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
import json
import os
import re
import sys

# structural operators/constructors — kept verbatim; everything else is a hole (α-renamed).
OPERATORS = {
    "=", "∈", "∉", "⊆", "⊂", "≅", "≃", "≈", "≤", "≥", "<", ">", "→", "↦", "⟶", "⇒",
    "∀", "∃", "forall", "exists", "λ", "∘", "×", "⊗", "⊕", "∧", "∨", "¬", "∩", "∪",
    ":", "*", "conditional-set", "set", "tuple", "pair", "apply", "and", "or", "not",
    "⊢", "↪", "≔", ":=",
    # type-formers kept DISTINCT (dependent product ≠ sum ≠ abstraction) once SFC emits them
    "Π", "Σ", "∏", "∑", "Pi", "Sigma", "lambda", "Lambda", "prod", "coprod", "∐",
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
    """:structure s-expr -> canonical structural key (α-renamed, operators kept).
    Returns None for an SFC COVERAGE GAP — a structure SFC left as :hole/unhandled — so
    unparsed structures are flagged, never falsely merged with one another (the Π/Σ trap)."""
    if structure_str and (":hole" in structure_str or "unhandled" in structure_str):
        return None
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


RELATION = re.compile(r"=|\\in\b|\\subseteq|\\subset|\\to\b|\\mapsto|\\cong|\\simeq|\\le|\\ge|<|>|:")


def _drop_group(s: str, command: str, keep: bool) -> str:
    """Remove (keep=False) or unwrap (keep=True) every \\command{...}, braces balanced."""
    out, i, key = [], 0, "\\" + command + "{"
    while True:
        j = s.find(key, i)
        if j < 0:
            return "".join(out) + s[i:]
        k, depth = j + len(key), 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        out.append(s[i:j] + (s[j + len(key):k - 1] if keep else " "))
        i = k


def split_formula(line: str) -> list[str]:
    """One extracted line as the formulae it holds.

    The structure reader takes one formula at a time. On the second APM run,
    19 of 247 lines held several side by side ("P_M^2 = P_M, \\qquad P_M^* = P_M",
    "H_0(M) = Z, \\quad H_1(M) = Z/2, ...") or a \\boxed result, and were read only
    in part; 3 used bracket-colon notation ([G:H], [K:k], [z_1 : z_2]) and 2 a
    \\tag or a phrase in \\text, and gave no structure at all.
    """
    s = _drop_group(_drop_group(line, "tag", keep=False), "boxed", keep=True)
    s = _drop_group(s, "text", keep=False)
    s = re.sub(r"\\ensuremath\{([^{}]*)\}", r"\1", s)
    # [A:B] -- an index, a degree, projective coordinates -- as a two-place operator
    s = re.sub(r"\[([^\[\]:,]+):([^\[\]:,]+)\]", r"\\operatorname{bracket}(\1,\2)", s)
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        depth += {"{": 1, "}": -1, "(": 1, ")": -1}.get(c, 0)
        sep = None
        if depth == 0:
            m = re.match(r",?\s*\\q?quad(?![A-Za-z])|,\s*\\\s", s[i:])
            if m:
                sep = m
        if sep:
            parts.append("".join(cur))
            cur = []
            i += sep.end()
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    out = []
    for part in parts:
        part = part.strip().strip(",.;").strip()
        if len(part) >= 3 and RELATION.search(part):
            out.append(part)
    return out or [line.strip()]


def batch(formulae):
    """Run sfc_def_structure on each formula, canonicalize, cluster by structural key.
    Returns {key: [formulae]} — the canonical definition shapes. A line holding
    several formulae is read as each of them (split_formula)."""
    formulae = [part for line in formulae for part in split_formula(line)]
    import os
    import subprocess
    from collections import defaultdict
    bb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sfc_def_structure.bb")
    groups = defaultdict(list)
    gaps = []   # SFC coverage gaps (:hole/unhandled) — flagged, not merged
    batch.unread = []   # (formula, why) the transducer gave no structure for
    for f in formulae:
        try:
            done = subprocess.run(["bb", bb, "-"], input=f, capture_output=True, text=True, timeout=30)
        except Exception as e:
            batch.unread.append((f, str(e)[:200]))
            continue
        out = done.stdout
        st = extract_structure(out)
        if not st:
            batch.unread.append((f, (done.stderr.strip().splitlines() or ["empty :structure"])[-1][:200]))
        k = canon_key(st) if st else None
        if k and k != "()":
            groups[k].append(f)
        elif st and (":hole" in st or "unhandled" in st):
            gaps.append(f)
    return groups, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=None, help="EDN file (or - for stdin), else --self-test")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--formulae", help="file of definition formulae (one per line) -> normalization ratio")
    ap.add_argument("--run-dir", help="persist the shape census here")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    a = ap.parse_args()
    if a.formulae:
        # RECORD A REFUSAL rather than dying into a void. This stage has never run:
        # nothing in the pipeline produces def-formulae.txt, and the transducer that
        # would (sfc_def_structure.bb) needs latexmlmath, which is not installed and
        # is documented nowhere. Because S11 chained its two scripts with `;`, the
        # FileNotFoundError was swallowed and the stage reported PASS on every run
        # (E-superpod-hardening H22). An artifact saying "not measured, because X"
        # is worth more than an exception nobody sees.
        if not os.path.exists(a.formulae):
            reason = (f"input {a.formulae} absent: no pipeline stage produces definition "
                      f"formulae. Producing them needs (a) an extraction step from "
                      f"the WARP layer's def-snippets.json prose and (b) the latexmlmath binary "
                      f"required by scripts/sfc_def_structure.bb, which is not installed "
                      f"and not listed as a dependency.")
            print(f"REFUSED: {reason}")
            if a.run_dir:
                out = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), a.run_dir)
                os.makedirs(out, exist_ok=True)
                with open(os.path.join(out, "structural-canon-defs.json"), "w") as fh:
                    json.dump({"run_id": a.run_id, "corpus_id": a.corpus_id,
                               "measured": False, "reason": reason}, fh, indent=1)
                print(f"wrote {a.run_dir}/structural-canon-defs.json (refusal recorded)")
            return
        forms = [l.strip() for l in open(a.formulae) if l.strip()]
        groups, gaps = batch(forms)
        nf = sum(len(v) for v in groups.values())
        if forms and not groups and not gaps:
            # Formulae went in and no structure came out: the transducer is broken
            # on this host, not the corpus empty. On the APM box (LaTeXML 0.8.6,
            # 2026-10-16) every formula read as an empty :structure and the stage
            # recorded 0 shapes as a measurement.
            first = batch.unread[0] if batch.unread else ("", "no output")
            raise SystemExit(f"REFUSED: 0 of {len(forms)} formulae gave a structure; "
                             f"sfc_def_structure.bb is not working here (first: {first[1]!r} "
                             f"for {first[0][:60]!r})")
        print(f"=== structural normalization: {nf} parsed defs -> {len(groups)} canonical shapes "
              f"(ratio {nf / max(1, len(groups)):.2f}×) ; {len(gaps)} SFC coverage-gaps flagged ===\n")
        for k, fs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
            print(f"×{len(fs)}  {k}")
            for f in fs[:2]:
                print(f"      ← {f[:64]}")
        if gaps:
            print(f"\nSFC coverage gaps ({len(gaps)} — :hole/unhandled, NOT merged; the constructs to teach SFC):")
            for f in gaps[:8]:
                print(f"   ⚑ {f[:64]}")
        if a.run_dir:
            out = a.run_dir if os.path.isabs(a.run_dir) else os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), a.run_dir)
            os.makedirs(out, exist_ok=True)
            payload = {"run_id": a.run_id, "corpus_id": a.corpus_id, "measured": True,
                       "n_formulae": nf, "n_canonical_shapes": len(groups),
                       "compression_ratio": round(nf / max(1, len(groups)), 4),
                       "shapes": [{"key": k, "n": len(v), "exemplars": v[:3]}
                                  for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))],
                       "sfc_coverage_gaps": gaps[:200]}
            with open(os.path.join(out, "structural-canon-defs.json"), "w") as fh:
                json.dump(payload, fh, indent=1)
            print(f"\nwrote {a.run_dir}/structural-canon-defs.json "
                  f"({len(groups)} shapes, {len(gaps)} coverage gaps)")
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
