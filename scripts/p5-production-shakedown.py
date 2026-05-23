#!/usr/bin/env python3
r"""Gate P5: production shakedown of the grounding pipeline.

Runs the symbol-grounding stack (the part Stage 5 would run on the
superpod) on two contrasting 100-paper samples from the arxiv math.CT
pool, and reports:

  - Throughput (papers/sec, mean per-paper wall time)
  - Memory ceiling (RSS at end of each pool)
  - Total bindings + canon emissions + arbitrated top-1 distribution
  - Per-strategy emit counts (does every strategy fire?)
  - Output-shape validation (each paper produces well-formed bindings)
  - Coherence (in-nLab-vocab fraction) for each pool
  - Topic-prior update: how many new canons land in the MSC prior
    when we run with --update-msc-prior

Pools (chosen to contrast on TOPIC BREADTH within the CT pool we have):

  recent: 100 papers dated 2024-01-01 or later. arxiv math.CT
    primary, but the recent slice carries more cross-listings into
    math.AT/math.RT/math.QA. Tests pipeline behaviour on a topically-
    diverse slice the priors will be shifting on aggressively.

  ct-pure: 100 papers whose categories list is exactly ["math.CT"].
    Narrow, "actually about category theory" papers. Tests pipeline
    behaviour where the priors should largely be no-ops.

Usage:
    python scripts/p5-production-shakedown.py \\
        --eprint-dir /home/joe/code/storage/futon6/data/arxiv-math-ct-eprints \\
        --metadata /home/joe/code/storage/futon6/data/arxiv-math-ct-metadata.jsonl \\
        --ner-kernel data/ner-kernel-clean.tsv \\
        --nlab-pages-dir /home/joe/code/nlab-content/pages \\
        --msc-prior data/topic-prior-msc.json \\
        --se-corpus-prior data/topic-prior-se-corpus.json \\
        --out data/p5-shakedown-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from futon6 import bayesian_grounding as _bg
from futon6 import grounding as _grd
from futon6 import topic_prior as _tp


def _load_module(name: str, rel_path: str):
    spec = spec_from_file_location(name, ROOT / rel_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPERPOD_JOB = _load_module("superpod_job_p5", "scripts/superpod-job.py")
TERM_EVIDENCE = _load_module(
    "build_arxiv_ct_term_evidence_p5", "scripts/build-arxiv-ct-term-evidence.py"
)


_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")


def _normalize_nlab(name: str) -> str:
    s = name.strip().split("#", 1)[0]
    s = re.sub(r"\s+", " ", s)
    return "".join(p[:1].upper() + p[1:] for p in s.split(" ") if p)


def load_nlab_vocab(pages_dir: Path) -> set[str]:
    out: set[str] = set()
    for name_file in pages_dir.rglob("name"):
        try:
            raw = name_file.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if raw:
            out.add(_normalize_nlab(raw))
    return out


def select_pool(metadata_path: Path, kind: str, n: int, seed: int):
    """Return list of (arxiv_id, categories) tuples for the chosen pool.

    Filter rules (applied to entries from `metadata_path`):
      - "all"     : all entries, random sample
      - "recent"  : entries dated 2024-01-01 or later
      - "ct-pure" : entries whose categories list is exactly ["math.CT"]
      - "non-ct"  : entries that do NOT include math.CT in categories
    """
    items: list[tuple[str, list[str]]] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        pid = rec.get("id", "")
        cats = rec.get("categories", []) or []
        date = rec.get("date", "")
        if kind == "recent" and date < "2024-01-01":
            continue
        if kind == "ct-pure" and cats != ["math.CT"]:
            continue
        if kind == "non-ct" and ("math.CT" in cats):
            continue
        items.append((pid, cats))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def resident_set_mb() -> float:
    """Linux-only RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


def find_tarball(eprint_dir: Path, arxiv_id: str) -> Path | None:
    """Match arxiv id to its tarball, allowing for v-suffix variation."""
    candidates = list(eprint_dir.glob(f"{arxiv_id}*.tar.gz"))
    return candidates[0] if candidates else None


def run_pool(name, pool, eprint_dir, singles, multi_index,
             msc_prior, se_prior, nlab_vocab, rels):
    print(f"[p5] === pool '{name}' ({len(pool)} papers) ===")
    t0 = time.time()
    mem0 = resident_set_mb()
    per_paper_wall: list[float] = []
    per_paper_n_canons: list[int] = []
    strategy_emit: Counter = Counter()
    canon_counter: Counter = Counter()
    hi_conf_canons: Counter = Counter()
    n_processed = 0
    n_skipped_no_tar = 0
    n_skipped_load_err = 0
    n_malformed_bindings = 0
    new_canons_in_msc_prior_at_start = len(msc_prior.counts) if msc_prior else 0

    for arxiv_id, cats in pool:
        tar = find_tarball(eprint_dir, arxiv_id)
        if tar is None:
            n_skipped_no_tar += 1
            continue
        try:
            text = TERM_EVIDENCE.LOAD_ARXIV._read_payload(tar)
        except Exception:
            n_skipped_load_err += 1
            continue
        n_processed += 1
        p_t0 = time.time()
        _, env, _ = _grd.detect_grounded_symbols(
            tar.stem, text, singles, multi_index,
            SUPERPOD_JOB.spot_terms_entity,
        )

        on_sym: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
        for b in env.all_bindings:
            # output-shape validation
            if not isinstance(b.symbol, str) or not b.strategy:
                n_malformed_bindings += 1
                continue
            strategy_emit[b.strategy] += 1
            on_sym[b.symbol].append((b.strategy, b.canon))

        msc_primaries = _tp.arxiv_categories_to_msc(cats)
        context_factors = []
        if msc_prior is not None and msc_primaries:
            context_factors.append(
                lambda c, _mp=msc_prior, _p=msc_primaries: _mp.prior(c, _p)
            )
        if se_prior is not None:
            context_factors.append(lambda c, _sp=se_prior: _sp.prior(c))

        paper_canons = 0
        for sym, votes in on_sym.items():
            post = _bg.combine_strategy_votes(
                sym, votes, rels,
                context_factors=context_factors or None,
            )
            top, prob = post.top1()
            if top is None:
                continue
            canon_counter[top] += 1
            if prob >= 0.5:
                hi_conf_canons[top] += 1
            paper_canons += 1
            if msc_prior is not None and msc_primaries and prob >= 0.5:
                msc_prior.add(top, msc_primaries[0], n=1)

        per_paper_wall.append(time.time() - p_t0)
        per_paper_n_canons.append(paper_canons)
        if n_processed % 25 == 0:
            print(f"[p5]   {name}: {n_processed}/{len(pool)} processed, "
                  f"{sum(canon_counter.values())} emissions, "
                  f"{len(canon_counter)} unique canons, "
                  f"RSS={resident_set_mb():.0f}MB")

    total_wall = time.time() - t0
    mem1 = resident_set_mb()
    total_em = sum(canon_counter.values())
    in_nlab = sum(c for k, c in canon_counter.items() if k in nlab_vocab)
    hi_em = sum(hi_conf_canons.values())
    hi_in_nlab = sum(c for k, c in hi_conf_canons.items() if k in nlab_vocab)
    new_canons_in_msc = (
        len(msc_prior.counts) - new_canons_in_msc_prior_at_start
        if msc_prior else 0
    )

    summary = {
        "pool": name,
        "n_papers_requested": len(pool),
        "n_skipped_no_tarball": n_skipped_no_tar,
        "n_skipped_load_error": n_skipped_load_err,
        "n_processed": n_processed,
        "wall_seconds": round(total_wall, 1),
        "throughput_papers_per_sec": round(n_processed / max(total_wall, 0.001), 2),
        "mean_paper_wall_sec": round(sum(per_paper_wall) / max(len(per_paper_wall), 1), 3),
        "rss_mb_start": round(mem0, 0),
        "rss_mb_end": round(mem1, 0),
        "rss_delta_mb": round(mem1 - mem0, 0),
        "n_malformed_bindings": n_malformed_bindings,
        "total_emissions": total_em,
        "unique_canons_emitted": len(canon_counter),
        "emissions_in_nlab": in_nlab,
        "emissions_in_nlab_pct": round(in_nlab / max(total_em, 1) * 100, 1),
        "high_conf_emissions": hi_em,
        "high_conf_in_nlab": hi_in_nlab,
        "high_conf_in_nlab_pct": round(hi_in_nlab / max(hi_em, 1) * 100, 1),
        "strategy_emit_counts": dict(strategy_emit.most_common()),
        "top_emitted_canons": canon_counter.most_common(20),
        "msc_prior_new_canons_added": new_canons_in_msc,
        "mean_canons_per_paper": round(sum(per_paper_n_canons) / max(len(per_paper_n_canons), 1), 1),
    }
    print(f"[p5] {name} DONE: {n_processed} papers in {total_wall:.1f}s "
          f"({summary['throughput_papers_per_sec']} papers/s); "
          f"emissions {total_em}; nLab-frac {summary['emissions_in_nlab_pct']}%")
    print(f"[p5]   strategies: {dict(strategy_emit.most_common(8))}")
    print(f"[p5]   malformed bindings: {n_malformed_bindings}")
    print(f"[p5]   new canons folded into MSC prior: {new_canons_in_msc}")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eprint-dir", type=Path, required=True,
                   help="Primary eprint dir (e.g. CT pool)")
    p.add_argument("--metadata", type=Path, required=True,
                   help="Primary metadata jsonl matching --eprint-dir")
    p.add_argument("--broad-eprint-dir", type=Path, default=None,
                   help="Secondary eprint dir for the broad/non-CT pool. "
                        "When supplied, the 'broad' pool replaces 'recent'.")
    p.add_argument("--broad-metadata", type=Path, default=None,
                   help="Secondary metadata jsonl for the broad pool. "
                        "Required if --broad-eprint-dir is supplied.")
    p.add_argument("--ner-kernel", type=Path, required=True)
    p.add_argument("--nlab-pages-dir", type=Path, required=True)
    p.add_argument("--msc-prior", type=Path, default=None)
    p.add_argument("--se-corpus-prior", type=Path, default=None)
    p.add_argument("--n-per-pool", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260523)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"[p5] loading nLab vocabulary from {args.nlab_pages_dir}")
    nlab_vocab = load_nlab_vocab(args.nlab_pages_dir)
    print(f"[p5] nLab vocabulary: {len(nlab_vocab)} canon names")

    print(f"[p5] selecting pools (n={args.n_per_pool} each)")
    pure_pool = select_pool(args.metadata, "ct-pure", args.n_per_pool, args.seed)
    if args.broad_eprint_dir and args.broad_metadata:
        broad_pool = select_pool(args.broad_metadata, "non-ct", args.n_per_pool, args.seed)
        pool_a_name = "broad-non-ct"
        pool_a_dir = args.broad_eprint_dir
        pool_a = broad_pool
        print(f"[p5]   broad-non-ct: {len(broad_pool)} papers from {args.broad_eprint_dir}")
    else:
        recent_pool = select_pool(args.metadata, "recent", args.n_per_pool, args.seed)
        pool_a_name = "recent-ct"
        pool_a_dir = args.eprint_dir
        pool_a = recent_pool
        print(f"[p5]   recent-ct: {len(recent_pool)} papers")
    print(f"[p5]   ct-pure : {len(pure_pool)} papers")

    singles, multi_index, _ = SUPERPOD_JOB.load_ner_kernel(args.ner_kernel)
    print(f"[p5] NER kernel: {len(singles)} singles, "
          f"{sum(len(v) for v in multi_index.values())} multi-word entries")

    msc_prior = (_tp.MSCTopicPrior.load(args.msc_prior)
                 if args.msc_prior else None)
    se_prior = (_tp.SECorpusPrior.load(args.se_corpus_prior)
                if args.se_corpus_prior else None)
    if msc_prior:
        print(f"[p5] MSC prior: {len(msc_prior.counts)} canons")
    if se_prior:
        print(f"[p5] SE corpus prior: {len(se_prior.counts)} canons")

    # Uniform reliability priors — the production pipeline gets
    # per-strategy reliabilities from the canon-store; this shakedown
    # just verifies the pipeline runs cleanly, not the precision.
    rels = {
        n: _bg.StrategyReliability(name=n, alpha=10.0, beta=10.0)
        for n in ["newcommand", "color-channel", "notation-env",
                  "let-binding", "fix-pattern", "denotation",
                  "inline-is-a", "the-Y-X", "section-context",
                  "kernel-ambient", "learned-vocab"]
    }

    pool_a_summary = run_pool(pool_a_name, pool_a, pool_a_dir,
                              singles, multi_index, msc_prior, se_prior,
                              nlab_vocab, rels)
    pure_summary = run_pool("ct-pure", pure_pool, args.eprint_dir,
                            singles, multi_index, msc_prior, se_prior,
                            nlab_vocab, rels)
    recent_summary = pool_a_summary  # back-compat alias for the report key

    print()
    print("[p5] === COMPARISON ===")
    print(f"  {'metric':28s}  {pool_a_name:>14s}  {'ct-pure':>12s}")
    for k in ["n_processed", "wall_seconds", "throughput_papers_per_sec",
              "mean_paper_wall_sec", "total_emissions",
              "mean_canons_per_paper", "unique_canons_emitted",
              "emissions_in_nlab_pct", "high_conf_in_nlab_pct",
              "rss_mb_end", "n_malformed_bindings",
              "msc_prior_new_canons_added"]:
        r = pool_a_summary.get(k, "n/a")
        c = pure_summary.get(k, "n/a")
        print(f"  {k:28s}  {str(r):>14s}  {str(c):>12s}")

    out = {
        "config": {
            "eprint_dir": str(args.eprint_dir),
            "broad_eprint_dir": str(args.broad_eprint_dir) if args.broad_eprint_dir else None,
            "ner_kernel": str(args.ner_kernel),
            "n_per_pool": args.n_per_pool,
            "seed": args.seed,
            "msc_prior": str(args.msc_prior) if args.msc_prior else None,
            "se_corpus_prior": str(args.se_corpus_prior) if args.se_corpus_prior else None,
        },
        "pool_a_name": pool_a_name,
        "pool_a": pool_a_summary,
        "ct_pure": pure_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n[p5] wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
