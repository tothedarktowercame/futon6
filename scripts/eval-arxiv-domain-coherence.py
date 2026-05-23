#!/usr/bin/env python3
r"""Run the symbol-grounding engine on arxiv papers and measure
how well its emitted canons cohere with a target domain vocabulary
(by default, nLab's page-title set — a proxy for "is this a CT
concept the literature recognises?").

For Joe's "are arxiv math.CT papers canonicalising into the nLab
tagset?" question. Pipeline:

  1. Load nLab vocabulary: every nLab page title → CamelCased canon.
  2. Sample N arxiv eprints (extract text from each tarball).
  3. Run engine + arbitration with the supplied canon-store.
  4. Aggregate every emitted canon across all papers.
  5. Report:
     - Top emitted canons
     - Coverage: fraction of UNIQUE canons that are in nLab vocab
     - Fraction of EMISSIONS (with multiplicity) that are in nLab vocab
     - High-confidence-only counts so noise from low-posterior
       arbitrations gets segmented out

Usage:
    python scripts/eval-arxiv-domain-coherence.py \\
        --eprint-dir /home/joe/code/storage/futon6/data/arxiv-math-ct-eprints \\
        --ner-kernel /home/joe/code/storage/futon6/data/ner-kernel/terms.tsv \\
        --canon-store data/canon-store-pm-pw-wiki-nlab/aggregate.json \\
        --nlab-pages-dir /home/joe/code/nlab-content/pages \\
        --max-papers 50 \\
        --out data/arxiv-coherence-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
import tempfile
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import bayesian_grounding as _bg
from futon6 import canon_store as _cs
from futon6 import grounding as _grd
from futon6 import topic_prior as _tp


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERPOD_JOB = _load_module("superpod_job_coh", "scripts/superpod-job.py")
TERM_EVIDENCE = _load_module(
    "build_arxiv_ct_term_evidence_coh", "scripts/build-arxiv-ct-term-evidence.py"
)


def _normalize_nlab_canon(name: str) -> str:
    s = name.strip()
    s = s.split("#", 1)[0]
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def load_nlab_vocab(pages_dir: Path) -> set[str]:
    """Walk nlab-content/pages/**/name files, build CamelCased name set."""
    names: set[str] = set()
    for name_file in pages_dir.rglob("name"):
        try:
            raw = name_file.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if raw:
            names.add(_normalize_nlab_canon(raw))
    return names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eprint-dir", type=Path, required=True,
                        help="Directory of *.tar.gz arxiv eprints")
    parser.add_argument("--ner-kernel", type=Path, required=True)
    parser.add_argument("--canon-store", type=Path, default=None,
                        help="Optional canon-store aggregate.json for prior")
    parser.add_argument("--nlab-pages-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("arxiv-coherence-report.json"))
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--match-mode", choices=["loose", "strict", "ancestry"],
                        default="loose")
    parser.add_argument("--ancestry-index", type=Path, default=None)
    parser.add_argument("--disable-strategy", action="append", default=[],
                        dest="disable_strategies")
    parser.add_argument("--msc-prior", type=Path, default=None,
                        help="topic-prior-msc.json — MSC topic prior")
    parser.add_argument("--se-corpus-prior", type=Path, default=None,
                        help="topic-prior-se-corpus.json — SE corpus-frequency prior")
    parser.add_argument("--arxiv-metadata", type=Path, default=None,
                        help="arxiv-math-ct-metadata.jsonl — gives per-paper "
                             "`categories` for topic prior lookup")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Posterior probability threshold for "
                             "'high-confidence' counts in the report")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = random.Random(args.seed)

    print(f"[coherence] loading nLab vocabulary from {args.nlab_pages_dir}")
    nlab_vocab = load_nlab_vocab(args.nlab_pages_dir)
    print(f"[coherence] nLab vocabulary: {len(nlab_vocab)} canon names")

    print(f"[coherence] sampling {args.max_papers} papers from {args.eprint_dir}")
    all_tarballs = sorted(args.eprint_dir.glob("*.tar.gz"))
    sample = rng.sample(all_tarballs, min(args.max_papers, len(all_tarballs)))
    print(f"[coherence] {len(sample)} papers sampled "
          f"(from {len(all_tarballs)} total)")

    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)
    store: dict | None = None
    if args.canon_store:
        store = _cs.load_aggregate(args.canon_store)
        print(f"[coherence] canon-store: {len(store)} entries")
    msc_prior = None
    if args.msc_prior:
        msc_prior = _tp.MSCTopicPrior.load(args.msc_prior)
        print(f"[coherence] MSC topic prior: {len(msc_prior.counts)} canons")
    se_prior = None
    if args.se_corpus_prior:
        se_prior = _tp.SECorpusPrior.load(args.se_corpus_prior)
        print(f"[coherence] SE corpus prior: {len(se_prior.counts)} canons, "
              f"{se_prior.n_documents} docs scanned at build time")
    paper_categories: dict[str, list[str]] = {}
    if args.arxiv_metadata and args.arxiv_metadata.exists():
        for line in args.arxiv_metadata.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id", "")
            cats = rec.get("categories", []) or []
            if pid and cats:
                # Index by both "0711.1739" and "0711.1739v1" forms — the
                # tar filenames use the v-suffixed form, the metadata may
                # use either depending on the dump version.
                paper_categories[pid] = cats
                # If id has no v-suffix, also key any vN extensions implicitly
                # by prefix matching in lookup.
        print(f"[coherence] loaded categories for {len(paper_categories)} papers "
              f"from {args.arxiv_metadata}")

    disabled = set(args.disable_strategies) if args.disable_strategies else None
    # Uniform reliability priors — we're not doing precision-against-gold
    # here, we're measuring canon coherence. Per-strategy weights don't
    # change the canon EMITTED, only the posterior probabilities.
    rels = {
        n: _bg.StrategyReliability(name=n, alpha=10.0, beta=10.0)
        for n in ["newcommand", "color-channel", "notation-env",
                  "let-binding", "fix-pattern", "denotation",
                  "inline-is-a", "the-Y-X", "section-context",
                  "kernel-ambient", "learned-vocab"]
    }

    canon_counter: Counter[str] = Counter()
    hi_conf_counter: Counter[str] = Counter()
    per_paper = []
    n_processed = 0
    for tar_path in sample:
        try:
            text = TERM_EVIDENCE.LOAD_ARXIV._read_payload(tar_path)
        except Exception as exc:
            print(f"[coherence] skip {tar_path.name}: {exc}")
            continue
        n_processed += 1
        _, env, _ = _grd.detect_grounded_symbols(
            tar_path.stem, text, singles, multi_index,
            SUPERPOD_JOB.spot_terms_entity,
            disabled_strategies=disabled,
        )
        # Group strategies by symbol; arbitrate per-symbol; emit top-1
        on_sym: dict[str, list[tuple[str, str | None]]] = {}
        for b in env.all_bindings:
            on_sym.setdefault(b.symbol, []).append((b.strategy, b.canon))
        # Resolve paper's arxiv categories -> MSC primary codes for the
        # MSC topic prior. Falls back to no down-weight if unknown.
        pid = tar_path.stem
        cats = paper_categories.get(pid) or paper_categories.get(
            pid.rsplit("v", 1)[0] if "v" in pid else pid, []
        )
        msc_primaries = _tp.arxiv_categories_to_msc(cats) if cats else []
        context_factors = []
        if msc_prior is not None:
            context_factors.append(
                lambda c, _mp=msc_prior, _p=msc_primaries: _mp.prior(c, _p)
            )
        if se_prior is not None:
            context_factors.append(lambda c, _sp=se_prior: _sp.prior(c))
        paper_canons = []
        for sym, votes in on_sym.items():
            prior = (
                _cs.canon_prior(store, sym) if store else None
            ) or None
            post = _bg.combine_strategy_votes(
                sym, votes, rels, prior=prior,
                context_factors=context_factors or None,
            )
            top, prob = post.top1()
            if top is None:
                continue
            canon_counter[top] += 1
            if prob >= args.confidence_threshold:
                hi_conf_counter[top] += 1
            paper_canons.append((sym, top, prob))
        per_paper.append({
            "paper": tar_path.name,
            "n_canons": len(paper_canons),
            "in_nlab_vocab": sum(1 for _, c, _ in paper_canons if c in nlab_vocab),
        })
        if n_processed % 10 == 0:
            print(f"[coherence]   ...{n_processed}/{len(sample)} papers; "
                  f"{sum(canon_counter.values())} total emissions; "
                  f"{len(canon_counter)} unique canons")

    total_emissions = sum(canon_counter.values())
    total_in_nlab = sum(c for k, c in canon_counter.items() if k in nlab_vocab)
    hi_emissions = sum(hi_conf_counter.values())
    hi_in_nlab = sum(c for k, c in hi_conf_counter.items() if k in nlab_vocab)

    unique_canons = set(canon_counter.keys())
    unique_in_nlab = sum(1 for k in unique_canons if k in nlab_vocab)

    out = {
        "eprint_dir": str(args.eprint_dir),
        "nlab_pages_dir": str(args.nlab_pages_dir),
        "canon_store": str(args.canon_store) if args.canon_store else None,
        "nlab_vocab_size": len(nlab_vocab),
        "papers_processed": n_processed,
        "total_canon_emissions": total_emissions,
        "unique_canons_emitted": len(unique_canons),
        "emissions_in_nlab_vocab": total_in_nlab,
        "unique_canons_in_nlab_vocab": unique_in_nlab,
        "high_confidence_emissions": hi_emissions,
        "high_confidence_in_nlab": hi_in_nlab,
        "top_canons": canon_counter.most_common(30),
        "top_canons_in_nlab": [
            (c, n) for c, n in canon_counter.most_common(50) if c in nlab_vocab
        ][:30],
        "top_canons_not_in_nlab": [
            (c, n) for c, n in canon_counter.most_common(100) if c not in nlab_vocab
        ][:30],
        "per_paper_summary": per_paper,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print()
    print(f"[coherence] DONE: {n_processed} papers, "
          f"{total_emissions} total canon emissions, "
          f"{len(unique_canons)} unique")
    print(f"[coherence] In-vocab coverage:")
    print(f"  unique canons: {unique_in_nlab}/{len(unique_canons)} "
          f"= {unique_in_nlab/max(1,len(unique_canons))*100:.1f}%")
    print(f"  emissions (w/mult): {total_in_nlab}/{total_emissions} "
          f"= {total_in_nlab/max(1,total_emissions)*100:.1f}%")
    if hi_emissions:
        print(f"  high-confidence (p≥{args.confidence_threshold}): "
              f"{hi_in_nlab}/{hi_emissions} "
              f"= {hi_in_nlab/hi_emissions*100:.1f}%")
    print()
    print(f"[coherence] Top 15 canons emitted:")
    for canon, n in canon_counter.most_common(15):
        marker = "✓nlab" if canon in nlab_vocab else "     "
        print(f"  {marker}  {canon:34s} n={n}")
    return out


if __name__ == "__main__":
    main()
