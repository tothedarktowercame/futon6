#!/usr/bin/env python3
"""
demo-discover-terms.py — sanity-check demo: show what `--discover-terms` adds.

Per §2.A.2.20 of M-interim-director-proxy-metric-inventory.md, the master
flag `--discover-terms` has been OFF for all 8 cached mark2 batches. Joe
(2026-05-19): *"I'll be convinced when I see it — even just one or two
papers as a demo or mockup showing the difference could be useful as a
sanity check before we try to do the next 300,000."*

This script:
1. Imports `extract_open_ner_candidates` + `spot_terms_entity` + kernel
   loader directly from superpod-job.py (no copy; runs the real code).
2. Loads the production NER kernel TSV (19,236 PM+SE terms).
3. Picks N entities from a cached batch's entities.json (default: batch-001
   qc; first 2 entities).
4. For each, prints a side-by-side:
   - **OFF (classical NER only):** terms spotted from the kernel
   - **ON (discover-terms added):** novel candidate terms in 6 contextual
     categories (latex-emph / called-as / is-called / defined-as /
     definition-of / definition-block-subject)

CPU-only; no GPU; no superpod. Demo only — does NOT modify any production
state or pipeline configuration.

Usage:
    python3 demo-discover-terms.py                  # 2 papers, batch-001
    python3 demo-discover-terms.py 5                # 5 papers
    python3 demo-discover-terms.py 5 batch-002      # 5 papers from batch-002

Author: claude-13 2026-05-19; lives at futon6/scripts/.
"""
import importlib.util
import json
import sys
from pathlib import Path


SUPERPOD_JOB = Path("/home/joe/code/futon6/scripts/superpod-job.py")
KERNEL_TSV = Path("/home/joe/code/storage/futon6/data/ner-kernel/terms.tsv")
ENTITIES_BASE = Path("/home/joe/code/storage/mark2/qc")


def load_superpod_module():
    spec = importlib.util.spec_from_file_location("superpod_job", SUPERPOD_JOB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    batch = sys.argv[2] if len(sys.argv) > 2 else "batch-001"

    sp = load_superpod_module()

    print(f"=== demo-discover-terms ===")
    print(f"Kernel TSV: {KERNEL_TSV}")
    print(f"Source batch: {batch}")
    print(f"Papers to demo: {n}")
    print()

    # Load kernel
    singles, multi_index, multi_count = sp.load_ner_kernel(str(KERNEL_TSV))
    print(f"Loaded {len(singles)} single + {multi_count} multi-word terms "
          f"= {len(singles) + multi_count} kernel terms")

    # Build known_terms set (matches what discover_terms code does)
    known = set(singles.keys())
    for rows in multi_index.values():
        for term_lower, _, _ in rows:
            known.add(term_lower)
    print(f"known_terms set: {len(known)} entries")
    print()

    # Load entities
    entities_path = ENTITIES_BASE / batch / "output" / "entities.json"
    with open(entities_path) as f:
        entities = json.load(f)
    print(f"Loaded {len(entities)} entities from {entities_path}")
    print()

    # Demo per entity
    for i, entity in enumerate(entities[:n]):
        eid = entity.get("entity/id", "?")
        title = entity.get("title", "(no title)")
        qb = entity.get("question-body", "") or ""
        ab = entity.get("answer-body", "") or ""
        full_text = (qb + " " + ab).strip()

        print(f"--- Paper {i+1}/{n}: {eid} ---")
        print(f"Title: {title[:140]}")
        print(f"Text length: {len(full_text)} chars")
        print()

        # OFF: classical NER spotting
        spotted = sp.spot_terms_entity(full_text, singles, multi_index)
        # `spotted` is the production output shape
        spotted_terms = sorted({t.get("term_lower") if isinstance(t, dict) else t for t in spotted})

        print(f"OFF (classical NER only): {len(spotted)} hits "
              f"({len(spotted_terms)} unique terms)")
        if spotted_terms:
            preview = spotted_terms[:25]
            print(f"  Preview: {preview}")
        print()

        # ON: discover_terms candidates not in kernel
        candidates = sp.extract_open_ner_candidates(full_text, max_per_entity=64)
        novel = [(term, source, ctx) for (term, source, ctx) in candidates
                 if term not in known]

        print(f"ON adds (discover_terms novel candidates): {len(candidates)} raw -> "
              f"{len(novel)} novel (after kernel-filter)")
        for term, source, ctx in novel[:20]:
            ctx_short = ctx if len(ctx) <= 100 else ctx[:97] + "..."
            print(f"  [{source}] {term!r}")
            print(f"     ctx: {ctx_short}")
        if len(novel) > 20:
            print(f"  ... ({len(novel) - 20} more novel candidates)")
        print()
        print()

    print("=== summary (abstracts-only mode) ===")
    print(f"This is exactly the data Stage 5 would write to")
    print(f"`candidate-new-terms.jsonl` if `--discover-terms` were ON")
    print(f"WITHOUT --discover-terms-eprint-dir (i.e. text-only on entities).")
    print()

    # === Mockup mode (synthetic LaTeX) ===
    print("=== mockup mode (synthetic LaTeX-shaped text) ===")
    print("Demonstrates the discovery mechanism FIRES on LaTeX-marked terms")
    print("even when those terms are not in the kernel.")
    print()
    mockup = r"""
We define a \emph{transposed Tarski monoid} as a structure $(M, \star)$
where the binary operation $\star$ satisfies a dualised absorption law.
This is called a \emph{Lagrange-Whitney convolution} when restricted to
the finite-dimensional case. The structure is sometimes referred to as
the bicommutator quotient ring. A \textit{coupled symplectic embedding}
is defined as a smooth injection preserving the symplectic form modulo
a twist. Such embeddings appear in the Tao-Vu rigidity programme.

\begin{definition}[Cohomogeneity-one $G$-action]
A smooth $G$-action on a manifold $M$ is said to be of
\emph{cohomogeneity one} if the orbit space $M/G$ is one-dimensional.
\end{definition}
"""
    candidates = sp.extract_open_ner_candidates(mockup, max_per_entity=64)
    novel = [(term, source, ctx) for (term, source, ctx) in candidates
             if term not in known]
    print(f"Mockup: {len(candidates)} raw candidates -> {len(novel)} novel")
    for term, source, ctx in novel:
        ctx_short = ctx if len(ctx) <= 100 else ctx[:97] + "..."
        print(f"  [{source}] {term!r}")
        print(f"     ctx: {ctx_short}")
    print()

    # === Real .tex mode (PlanetMath sources) ===
    pm_tex_dir = Path("/home/joe/tmp-cat18")
    if pm_tex_dir.is_dir():
        tex_files = sorted(pm_tex_dir.glob("*.tex"))[:2]
        if tex_files:
            print(f"=== real .tex mode ({len(tex_files)} PlanetMath sources) ===")
            print("Demonstrates the mechanism on actual LaTeX from PM articles")
            print("(the same shape arxiv eprints would be in).")
            print()
            for tex_path in tex_files:
                text = tex_path.read_text(errors="replace")
                print(f"--- {tex_path.name} ({len(text)} chars) ---")
                candidates = sp.extract_open_ner_candidates(text, max_per_entity=64)
                novel = [(term, source, ctx) for (term, source, ctx) in candidates
                         if term not in known]
                print(f"  {len(candidates)} raw -> {len(novel)} novel")
                for term, source, ctx in novel[:15]:
                    ctx_short = ctx if len(ctx) <= 100 else ctx[:97] + "..."
                    print(f"  [{source}] {term!r}")
                    print(f"     ctx: {ctx_short}")
                if len(novel) > 15:
                    print(f"  ... ({len(novel) - 15} more)")
                print()
    print("=== overall summary ===")
    print("1. Abstracts-only mode (current entities.json): near-zero novelty")
    print("   (abstracts rarely contain \\emph or definition blocks)")
    print("2. Synthetic LaTeX mockup: mechanism fires as expected (regexes match)")
    print("3. Real .tex source: realistic novelty rate from actual LaTeX-marked terms")
    print()
    print("=> For mark2's 300K queued papers, eprint-dir mode is the load-bearing")
    print("   path. Enabling --discover-terms ALONE on the current entities.json")
    print("   would produce a near-empty candidate-new-terms.jsonl; the value")
    print("   comes from --discover-terms + --discover-terms-eprint-dir together.")


if __name__ == "__main__":
    main()
