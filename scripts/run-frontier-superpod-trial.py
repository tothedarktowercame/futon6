#!/usr/bin/env python3
"""Empirical FrontierMath trial harness for superpod outputs.

This script turns frontier problem-state files into concrete retrieval trials
against a superpod run directory (e.g. `math-processed-gpu`).

Outputs:
- review JSON with ranked candidate threads and `judgement` placeholders
- summary markdown with accounting and failure signals

Scoring mode:
- reads a review JSON with completed judgements (`yes` / `no` / `unsure`)
- reports strict + weighted precision, P@k, and MAP
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
PROBLEM_ID_RE = re.compile(r"`problem_id`:\s*([A-Za-z0-9\-]+)")
TITLE_RE = re.compile(r"`title`:\s*(.+)")


EXTRA_QUERY_TOKENS = {
    "FM-001": {
        "ramsey",
        "book",
        "graph",
        "combinatoric",
        "extremal",
        "edge",
        "coloring",
        "clique",
    },
    "FM-002": {
        "ramsey",
        "hypergraph",
        "uniform",
        "combinatoric",
        "extremal",
        "coloring",
        "set",
    },
    "FM-003": {
        "steiner",
        "system",
        "design",
        "block",
        "combinatoric",
        "existence",
        "construct",
    },
}

PRIMARY_TOKENS = {
    "FM-001": {"ramsey", "book", "graph"},
    "FM-002": {"ramsey", "hypergraph"},
    "FM-003": {"steiner", "system", "design"},
}


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    title: str
    state_path: Path
    query_tokens: set[str]
    primary_tokens: set[str]


@dataclass
class SeedCandidate:
    score: float
    entity_index: int
    thread_id: int
    title: str
    tags: list[str]
    title_hits: list[str]
    tag_hits: list[str]


def normalize_token(token: str) -> str:
    tok = token.lower().strip()
    if tok.endswith("ies") and len(tok) > 4:
        tok = tok[:-3] + "y"
    elif tok.endswith("s") and len(tok) > 4:
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> set[str]:
    out = set()
    for raw in TOKEN_RE.findall(text.lower()):
        t = normalize_token(raw)
        if len(t) >= 2:
            out.add(t)
    return out


def parse_problem_state(path: Path) -> ProblemSpec:
    text = path.read_text(encoding="utf-8")
    id_match = PROBLEM_ID_RE.search(text)
    title_match = TITLE_RE.search(text)
    if not id_match or not title_match:
        raise ValueError(f"could not parse problem metadata from {path}")

    pid = id_match.group(1).strip()
    title = title_match.group(1).strip().strip("`")

    q_tokens = set(tokenize(title))
    q_tokens.update(EXTRA_QUERY_TOKENS.get(pid, set()))
    primary = set(PRIMARY_TOKENS.get(pid, set()))
    if not primary:
        primary = set(q_tokens)

    return ProblemSpec(
        problem_id=pid,
        title=title,
        state_path=path,
        query_tokens=q_tokens,
        primary_tokens=primary,
    )


def load_problem_specs(frontier_dir: Path) -> list[ProblemSpec]:
    files = sorted(frontier_dir.glob("FM-*-state.md"))
    specs = [parse_problem_state(p) for p in files]
    if not specs:
        raise ValueError(f"no FM-*-state.md files found in {frontier_dir}")
    return specs


def extract_thread_id(entity_id: str) -> Optional[int]:
    if not isinstance(entity_id, str):
        return None
    tail = entity_id.rsplit("-", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return None


def ensure_compact_entities_jsonl(
    entities_path: Path,
    compact_path: Path,
    rebuild: bool,
) -> Path:
    """Materialize compact entity records via jq for fast repeated scans."""
    if compact_path.exists() and not rebuild:
        return compact_path

    compact_path.parent.mkdir(parents=True, exist_ok=True)
    jq_program = (
        '.[] | {'
        '"thread_id": (try (."entity/id" | split("-") | last | tonumber) catch null), '
        '"title": (.title // ""), '
        '"tags": ((.tags // []) | map(tostring))'
        "}"
    )
    cmd = ["jq", "-c", jq_program, str(entities_path)]
    with compact_path.open("w", encoding="utf-8") as out_f:
        subprocess.run(cmd, check=True, stdout=out_f)
    return compact_path


def iter_compact_entities(compact_path: Path) -> Iterator[dict]:
    with compact_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)



def score_entity(spec: ProblemSpec, title_tokens: set[str], tag_tokens: set[str]) -> tuple[float, list[str], list[str], int]:
    title_hits = sorted(spec.query_tokens & title_tokens)
    tag_hits = sorted(spec.query_tokens & tag_tokens)

    title_hit_count = len(title_hits)
    tag_hit_count = len(tag_hits)
    primary_hits = len(spec.primary_tokens & (title_tokens | tag_tokens))

    score = 0.0
    score += 3.0 * title_hit_count
    score += 2.0 * tag_hit_count
    if primary_hits > 0:
        score += 4.0 + float(primary_hits)
    if title_hit_count >= 2:
        score += 1.5
    if tag_hit_count >= 2:
        score += 1.0

    return score, title_hits, tag_hits, primary_hits


def select_lexical_seeds(
    compact_entities_path: Path,
    specs: Sequence[ProblemSpec],
    seed_k: int,
    min_seed_score: float,
) -> tuple[dict[str, list[SeedCandidate]], dict[str, dict[str, int]]]:
    heaps: dict[str, list[tuple[float, int, SeedCandidate]]] = {
        spec.problem_id: [] for spec in specs
    }
    counters: dict[str, dict[str, int]] = {
        spec.problem_id: {"scanned": 0, "positive": 0} for spec in specs
    }

    for idx, ent in enumerate(iter_compact_entities(compact_entities_path)):
        thread_id = ent.get("thread_id")
        if thread_id is None:
            continue
        thread_id = int(thread_id)

        title = (ent.get("title") or "").strip()
        if not title:
            continue

        tags_raw = ent.get("tags") or []
        if not isinstance(tags_raw, list):
            tags_raw = []
        tags = [str(t) for t in tags_raw if isinstance(t, str)]

        title_tokens = tokenize(title)
        tag_tokens: set[str] = set()
        for tag in tags:
            tag_tokens.update(tokenize(tag))

        for spec in specs:
            counters[spec.problem_id]["scanned"] += 1
            score, title_hits, tag_hits, _ = score_entity(spec, title_tokens, tag_tokens)
            if score < min_seed_score:
                continue

            counters[spec.problem_id]["positive"] += 1
            cand = SeedCandidate(
                score=score,
                entity_index=idx,
                thread_id=thread_id,
                title=title,
                tags=tags,
                title_hits=title_hits,
                tag_hits=tag_hits,
            )

            heap = heaps[spec.problem_id]
            entry = (score, -idx, cand)
            if len(heap) < seed_k:
                heapq.heappush(heap, entry)
            else:
                if entry > heap[0]:
                    heapq.heapreplace(heap, entry)

    out: dict[str, list[SeedCandidate]] = {}
    for pid, heap in heaps.items():
        best = [h[2] for h in sorted(heap, key=lambda x: (-x[0], x[1]))]
        out[pid] = best

    return out, counters


def map_seed_threads_to_struct_rows(struct_ids: Sequence[int], seed_threads: set[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    if not seed_threads:
        return out
    for row, tid in enumerate(struct_ids):
        if tid in seed_threads and tid not in out:
            out[tid] = row
            if len(out) == len(seed_threads):
                break
    return out


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-8)


def topk_structural_neighbours(
    emb_path: Path,
    query_rows: list[int],
    topk_each: int,
    chunk_rows: int,
) -> dict[int, list[tuple[int, float]]]:
    emb = np.load(str(emb_path), mmap_mode="r")
    n_rows = int(emb.shape[0])

    if not query_rows:
        return {}

    qmat = np.asarray(emb[query_rows], dtype=np.float32)
    qmat = normalize_rows(qmat)
    qcount = qmat.shape[0]

    heaps: list[list[tuple[float, int]]] = [[] for _ in range(qcount)]
    local_topk = max(topk_each * 3, 64)

    for start in range(0, n_rows, chunk_rows):
        end = min(n_rows, start + chunk_rows)
        chunk = np.asarray(emb[start:end], dtype=np.float32)
        chunk = normalize_rows(chunk)

        sims = chunk @ qmat.T  # [chunk, qcount]

        for qi in range(qcount):
            col = sims[:, qi]
            k = min(local_topk, col.shape[0])
            if k <= 0:
                continue
            idx = np.argpartition(col, -k)[-k:]
            heap = heaps[qi]
            for local_row in idx:
                row = start + int(local_row)
                sim = float(col[local_row])
                entry = (sim, row)
                if len(heap) < topk_each:
                    heapq.heappush(heap, entry)
                else:
                    if entry > heap[0]:
                        heapq.heapreplace(heap, entry)

    out: dict[int, list[tuple[int, float]]] = {}
    for qi, qrow in enumerate(query_rows):
        ranked = sorted(heaps[qi], key=lambda x: (-x[0], x[1]))
        out[qrow] = [(row, sim) for sim, row in ranked]
    return out


def fetch_thread_metadata(entities_path: Path, needed_threads: set[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not needed_threads:
        return out

    for ent in iter_compact_entities(entities_path):
        tid = ent.get("thread_id")
        if tid is None:
            continue
        tid = int(tid)
        if tid not in needed_threads:
            continue
        title = (ent.get("title") or "").strip()
        tags_raw = ent.get("tags") or []
        if not isinstance(tags_raw, list):
            tags_raw = []
        tags = [str(t) for t in tags_raw if isinstance(t, str)]
        out[tid] = {"title": title, "tags": tags}
        if len(out) == len(needed_threads):
            break

    return out


def classify_proxy_usefulness(
    spec: ProblemSpec,
    title: str,
    tags: list[str],
    anchor_count: int,
) -> tuple[str, list[str]]:
    title_tokens = tokenize(title)
    tag_tokens = set()
    for tag in tags:
        tag_tokens.update(tokenize(tag))

    overlap = sorted(spec.query_tokens & (title_tokens | tag_tokens))
    primary_overlap = sorted(spec.primary_tokens & (title_tokens | tag_tokens))

    if len(primary_overlap) >= 2 or len(overlap) >= 3:
        return "likely", overlap
    if len(overlap) == 0 and anchor_count == 0:
        return "unlikely", overlap
    return "unclear", overlap


def p_at_k(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    k_eff = min(k, len(values))
    if k_eff == 0:
        return 0.0
    return sum(values[:k_eff]) / float(k_eff)


def map_yes_only(binary: list[int]) -> float:
    total_yes = sum(binary)
    if total_yes == 0:
        return 0.0
    c = 0
    ap = 0.0
    for i, v in enumerate(binary, start=1):
        if v:
            c += 1
            ap += c / i
    return ap / total_yes


def score_review(review_obj: dict) -> dict:
    per_problem = []
    overall_records = []
    overall_by_source = {
        "lexical_seed": [],
        "structural_neighbor": [],
        "structural_neighbor_zero_overlap": [],
        "structural_neighbor_with_overlap": [],
    }

    for problem in review_obj.get("problems", []):
        pid = problem.get("problem_id", "?")
        ranked = sorted(problem.get("candidates", []), key=lambda r: r.get("rank", 10**9))

        labels = extract_labels(ranked)
        for j in labels:
            overall_records.append(j)

        lexical_recs = [r for r in ranked if r.get("source") == "lexical_seed"]
        struct_recs = [r for r in ranked if r.get("source") == "structural_neighbor"]
        struct_zero = [r for r in struct_recs if len(r.get("token_overlap") or []) == 0]
        struct_with = [r for r in struct_recs if len(r.get("token_overlap") or []) > 0]

        lexical_labels = extract_labels(lexical_recs)
        struct_labels = extract_labels(struct_recs)
        struct_zero_labels = extract_labels(struct_zero)
        struct_with_labels = extract_labels(struct_with)

        overall_by_source["lexical_seed"].extend(lexical_labels)
        overall_by_source["structural_neighbor"].extend(struct_labels)
        overall_by_source["structural_neighbor_zero_overlap"].extend(struct_zero_labels)
        overall_by_source["structural_neighbor_with_overlap"].extend(struct_with_labels)

        metrics = compute_label_metrics(labels)
        metrics["problem_id"] = pid
        metrics["candidate_counts"] = {
            "all": len(ranked),
            "lexical_seed": len(lexical_recs),
            "structural_neighbor": len(struct_recs),
            "structural_neighbor_zero_overlap": len(struct_zero),
            "structural_neighbor_with_overlap": len(struct_with),
        }
        metrics["labeled_counts"] = {
            "all": len(labels),
            "lexical_seed": len(lexical_labels),
            "structural_neighbor": len(struct_labels),
            "structural_neighbor_zero_overlap": len(struct_zero_labels),
            "structural_neighbor_with_overlap": len(struct_with_labels),
        }
        metrics["by_source"] = {
            "lexical_seed": compute_label_metrics(lexical_labels),
            "structural_neighbor": compute_label_metrics(struct_labels),
            "structural_neighbor_zero_overlap": compute_label_metrics(struct_zero_labels),
            "structural_neighbor_with_overlap": compute_label_metrics(struct_with_labels),
        }
        per_problem.append(metrics)

    overall = compute_label_metrics(overall_records)
    overall["by_source"] = {
        "lexical_seed": compute_label_metrics(overall_by_source["lexical_seed"]),
        "structural_neighbor": compute_label_metrics(overall_by_source["structural_neighbor"]),
        "structural_neighbor_zero_overlap": compute_label_metrics(
            overall_by_source["structural_neighbor_zero_overlap"]
        ),
        "structural_neighbor_with_overlap": compute_label_metrics(
            overall_by_source["structural_neighbor_with_overlap"]
        ),
    }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "per_problem": per_problem,
        "overall": overall,
    }


def extract_labels(records: Sequence[dict]) -> list[str]:
    labels = []
    for rec in records:
        j = (rec.get("judgement") or "").strip().lower()
        if j in {"yes", "no", "unsure"}:
            labels.append(j)
    return labels


def compute_label_metrics(labels: Sequence[str]) -> dict:
    n = len(labels)
    yes = sum(1 for x in labels if x == "yes")
    no = sum(1 for x in labels if x == "no")
    unsure = sum(1 for x in labels if x == "unsure")

    strict = yes / (yes + no) if (yes + no) > 0 else 0.0
    weighted = (yes + 0.5 * unsure) / n if n > 0 else 0.0

    yes_values = [1.0 if x == "yes" else 0.0 for x in labels]
    weighted_values = [1.0 if x == "yes" else (0.5 if x == "unsure" else 0.0) for x in labels]

    return {
        "pairs": n,
        "yes": yes,
        "no": no,
        "unsure": unsure,
        "strict_precision": round(strict, 3),
        "weighted_score": round(weighted, 3),
        "p5_yes": round(p_at_k(yes_values, 5), 3),
        "p10_yes": round(p_at_k(yes_values, 10), 3),
        "p20_yes": round(p_at_k(yes_values, 20), 3),
        "p5_weighted": round(p_at_k(weighted_values, 5), 3),
        "p10_weighted": round(p_at_k(weighted_values, 10), 3),
        "p20_weighted": round(p_at_k(weighted_values, 20), 3),
        "map_yes_only": round(map_yes_only([int(v) for v in yes_values]), 3),
    }


def build_trial(
    outdir: Path,
    frontier_dir: Path,
    compact_entities_path: Path,
    rebuild_compact: bool,
    out_review: Path,
    out_summary: Path,
    seed_k: int,
    n_anchors: int,
    struct_k: int,
    struct_min_overlap: int,
    allow_blank_struct_titles: bool,
    min_seed_score: float,
    chunk_rows: int,
) -> dict:
    entities_path = outdir / "entities.json"
    struct_emb_path = outdir / "hypergraph-embeddings.npy"
    struct_ids_path = outdir / "hypergraph-thread-ids.json"

    for p in [entities_path, struct_emb_path, struct_ids_path]:
        if not p.exists():
            raise FileNotFoundError(f"required artifact not found: {p}")

    compact_entities_path = ensure_compact_entities_jsonl(
        entities_path=entities_path,
        compact_path=compact_entities_path,
        rebuild=rebuild_compact,
    )

    specs = load_problem_specs(frontier_dir)

    seeds_by_problem, counters = select_lexical_seeds(
        compact_entities_path=compact_entities_path,
        specs=specs,
        seed_k=seed_k,
        min_seed_score=min_seed_score,
    )

    struct_ids = json.loads(struct_ids_path.read_text(encoding="utf-8"))
    struct_ids = [int(x) for x in struct_ids]

    all_seed_threads = {
        cand.thread_id
        for cands in seeds_by_problem.values()
        for cand in cands
    }
    seed_thread_to_row = map_seed_threads_to_struct_rows(struct_ids, all_seed_threads)

    problem_records = []
    needed_thread_meta: set[int] = set()

    for spec in specs:
        seeds = seeds_by_problem.get(spec.problem_id, [])
        anchors = []
        for cand in seeds:
            row = seed_thread_to_row.get(cand.thread_id)
            if row is None:
                continue
            anchors.append((cand, row))
            if len(anchors) >= n_anchors:
                break

        query_rows = [r for _, r in anchors]
        neighbor_by_anchor = topk_structural_neighbours(
            emb_path=struct_emb_path,
            query_rows=query_rows,
            topk_each=max(struct_k * 2, 80),
            chunk_rows=chunk_rows,
        ) if query_rows else {}

        anchor_rows = {r for _, r in anchors}
        anchor_threads = {cand.thread_id for cand, _ in anchors}
        agg: dict[int, dict] = {}

        for cand, row in anchors:
            ranked = neighbor_by_anchor.get(row, [])
            for nb_row, sim in ranked:
                if nb_row in anchor_rows:
                    continue
                tid = int(struct_ids[nb_row])
                if tid in anchor_threads:
                    continue
                rec = agg.get(tid)
                if rec is None:
                    agg[tid] = {
                        "thread_id": tid,
                        "structural_similarity": sim,
                        "anchor_thread_ids": [cand.thread_id],
                        "anchor_titles": [cand.title],
                    }
                else:
                    if sim > rec["structural_similarity"]:
                        rec["structural_similarity"] = sim
                    if cand.thread_id not in rec["anchor_thread_ids"]:
                        rec["anchor_thread_ids"].append(cand.thread_id)
                        rec["anchor_titles"].append(cand.title)

        structural_candidates = sorted(
            agg.values(), key=lambda x: (-x["structural_similarity"], x["thread_id"])
        )[:struct_k]

        needed_thread_meta.update(c.thread_id for c in seeds)
        needed_thread_meta.update(c["thread_id"] for c in structural_candidates)

        problem_records.append(
            {
                "problem_id": spec.problem_id,
                "title": spec.title,
                "query_tokens": sorted(spec.query_tokens),
                "primary_tokens": sorted(spec.primary_tokens),
                "seed_scan": counters[spec.problem_id],
                "lexical_seeds": seeds,
                "anchors": anchors,
                "structural_candidates": structural_candidates,
            }
        )

    metadata_by_tid = fetch_thread_metadata(compact_entities_path, needed_thread_meta)

    review_obj = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "frontier_dir": str(frontier_dir),
        "settings": {
            "seed_k": seed_k,
            "n_anchors": n_anchors,
            "struct_k": struct_k,
            "struct_min_overlap": struct_min_overlap,
            "allow_blank_struct_titles": allow_blank_struct_titles,
            "min_seed_score": min_seed_score,
            "chunk_rows": chunk_rows,
            "compact_entities_jsonl": str(compact_entities_path),
        },
        "problems": [],
    }

    summary_lines = [
        "# Frontier Superpod Empirical Trial",
        "",
        f"Generated: `{review_obj['generated_utc']}`",
        f"Outdir: `{outdir}`",
        f"Frontier dir: `{frontier_dir}`",
        "",
    ]

    for prec in problem_records:
        spec = next(s for s in specs if s.problem_id == prec["problem_id"])
        candidates = []
        rank = 1
        seen = set()
        filtered_struct_no_overlap = 0
        filtered_struct_missing_meta = 0

        for seed in prec["lexical_seeds"]:
            if seed.thread_id in seen:
                continue
            seen.add(seed.thread_id)
            meta = metadata_by_tid.get(seed.thread_id, {"title": seed.title, "tags": seed.tags})
            proxy, overlap = classify_proxy_usefulness(
                spec,
                meta.get("title", seed.title),
                meta.get("tags", seed.tags),
                anchor_count=0,
            )
            candidates.append(
                {
                    "rank": rank,
                    "source": "lexical_seed",
                    "thread": {
                        "thread_id": seed.thread_id,
                        "title": meta.get("title", seed.title),
                        "tags": meta.get("tags", seed.tags),
                    },
                    "scores": {
                        "lexical": round(float(seed.score), 3),
                        "structural": None,
                    },
                    "token_overlap": overlap,
                    "proxy_usefulness": proxy,
                    "anchor_thread_ids": [],
                    "judgement": None,
                    "notes": "",
                }
            )
            rank += 1

        for c in prec["structural_candidates"]:
            tid = int(c["thread_id"])
            if tid in seen:
                continue
            seen.add(tid)
            meta = metadata_by_tid.get(tid, {"title": "", "tags": []})
            if not allow_blank_struct_titles and not meta.get("title"):
                filtered_struct_missing_meta += 1
                continue
            proxy, overlap = classify_proxy_usefulness(
                spec,
                meta.get("title", ""),
                meta.get("tags", []),
                anchor_count=len(c.get("anchor_thread_ids", [])),
            )
            if len(overlap) < struct_min_overlap:
                filtered_struct_no_overlap += 1
                continue
            candidates.append(
                {
                    "rank": rank,
                    "source": "structural_neighbor",
                    "thread": {
                        "thread_id": tid,
                        "title": meta.get("title", ""),
                        "tags": meta.get("tags", []),
                    },
                    "scores": {
                        "lexical": None,
                        "structural": round(float(c["structural_similarity"]), 4),
                    },
                    "token_overlap": overlap,
                    "proxy_usefulness": proxy,
                    "anchor_thread_ids": c.get("anchor_thread_ids", []),
                    "judgement": None,
                    "notes": "",
                }
            )
            rank += 1

        likely = sum(1 for r in candidates if r["proxy_usefulness"] == "likely")
        unclear = sum(1 for r in candidates if r["proxy_usefulness"] == "unclear")
        unlikely = sum(1 for r in candidates if r["proxy_usefulness"] == "unlikely")
        missing_meta = sum(1 for r in candidates if not r["thread"]["title"])

        review_obj["problems"].append(
            {
                "problem_id": prec["problem_id"],
                "title": prec["title"],
                "state_file": str(next(s.state_path for s in specs if s.problem_id == prec["problem_id"])),
                "seed_scan": prec["seed_scan"],
                "n_lexical_seeds": len(prec["lexical_seeds"]),
                "n_anchors": len(prec["anchors"]),
                "n_structural_candidates": len(prec["structural_candidates"]),
                "accounting": {
                    "candidates_total": len(candidates),
                    "proxy_likely": likely,
                    "proxy_unclear": unclear,
                    "proxy_unlikely": unlikely,
                    "missing_metadata": missing_meta,
                    "filtered_struct_no_overlap": filtered_struct_no_overlap,
                    "filtered_struct_missing_meta": filtered_struct_missing_meta,
                },
                "candidates": candidates,
            }
        )

        summary_lines.append(f"## {prec['problem_id']}: {prec['title']}")
        summary_lines.append("")
        summary_lines.append(
            f"- seed scan: `{prec['seed_scan']['positive']}` positive over `{prec['seed_scan']['scanned']}` scanned"
        )
        summary_lines.append(
            f"- lexical seeds: `{len(prec['lexical_seeds'])}`; anchors with structural rows: `{len(prec['anchors'])}`"
        )
        summary_lines.append(
            f"- structural candidates: `{len(prec['structural_candidates'])}`"
        )
        if filtered_struct_no_overlap or filtered_struct_missing_meta:
            summary_lines.append(
                f"- structural filtered (no-overlap/missing-meta): `{filtered_struct_no_overlap}/{filtered_struct_missing_meta}`"
            )
        summary_lines.append(
            f"- accounting proxy (likely/unclear/unlikely): `{likely}/{unclear}/{unlikely}`"
        )
        if missing_meta:
            summary_lines.append(f"- missing metadata records: `{missing_meta}`")
        summary_lines.append("")
        summary_lines.append("Top candidates:")
        summary_lines.append("")
        summary_lines.append("| Rank | Source | Thread ID | Structural | Proxy | Title |")
        summary_lines.append("|---:|---|---:|---:|---|---|")
        for rec in candidates[:10]:
            st = rec["scores"]["structural"]
            st_s = "" if st is None else f"{st:.4f}"
            title = (rec["thread"]["title"] or "").replace("|", " ")
            summary_lines.append(
                f"| {rec['rank']} | {rec['source']} | {rec['thread']['thread_id']} | {st_s} | {rec['proxy_usefulness']} | {title[:120]} |"
            )
        summary_lines.append("")

    out_review.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_review.write_text(json.dumps(review_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return review_obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FrontierMath empirical trial on superpod outputs")
    parser.add_argument("--outdir", type=Path, default=Path("/home/joe/code/storage/math-processed-gpu"))
    parser.add_argument(
        "--frontier-dir",
        type=Path,
        default=Path("data/first-proof/frontiermath-pilot"),
    )
    parser.add_argument(
        "--out-review",
        type=Path,
        default=Path("data/first-proof/frontiermath-pilot/superpod-frontier-trial-review.json"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("data/first-proof/frontiermath-pilot/superpod-frontier-trial-summary.md"),
    )
    parser.add_argument("--seed-k", type=int, default=12)
    parser.add_argument("--n-anchors", type=int, default=4)
    parser.add_argument("--struct-k", type=int, default=20)
    parser.add_argument(
        "--struct-min-overlap",
        type=int,
        default=1,
        help="Minimum query-token overlap required to keep structural candidates",
    )
    parser.add_argument(
        "--allow-blank-struct-titles",
        action="store_true",
        help="Keep structural candidates even when metadata title is missing",
    )
    parser.add_argument("--min-seed-score", type=float, default=5.0)
    parser.add_argument("--chunk-rows", type=int, default=120000)
    parser.add_argument(
        "--compact-entities-jsonl",
        type=Path,
        default=None,
        help="Path to compact entities JSONL cache; default is <outdir>/entities.compact.jsonl",
    )
    parser.add_argument(
        "--rebuild-compact",
        action="store_true",
        help="Rebuild compact entities cache even if it already exists",
    )
    parser.add_argument(
        "--score-judgements",
        type=Path,
        default=None,
        help="Score a completed review JSON instead of building a new one",
    )
    parser.add_argument(
        "--out-score",
        type=Path,
        default=None,
        help="Optional path to write score JSON when using --score-judgements",
    )
    args = parser.parse_args()

    if args.score_judgements:
        review = json.loads(args.score_judgements.read_text(encoding="utf-8"))
        scored = score_review(review)
        print(json.dumps(scored, indent=2, ensure_ascii=False))
        if args.out_score:
            args.out_score.parent.mkdir(parents=True, exist_ok=True)
            args.out_score.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    compact_path = args.compact_entities_jsonl
    if compact_path is None:
        compact_path = args.outdir / "entities.compact.jsonl"

    review_obj = build_trial(
        outdir=args.outdir,
        frontier_dir=args.frontier_dir,
        compact_entities_path=compact_path,
        rebuild_compact=args.rebuild_compact,
        out_review=args.out_review,
        out_summary=args.out_summary,
        seed_k=args.seed_k,
        n_anchors=args.n_anchors,
        struct_k=args.struct_k,
        struct_min_overlap=args.struct_min_overlap,
        allow_blank_struct_titles=args.allow_blank_struct_titles,
        min_seed_score=args.min_seed_score,
        chunk_rows=args.chunk_rows,
    )

    print(f"wrote review: {args.out_review}")
    print(f"wrote summary: {args.out_summary}")
    print(f"problems: {len(review_obj.get('problems', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
