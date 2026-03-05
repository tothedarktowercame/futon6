#!/usr/bin/env python3
"""Retrieve relevant math.SE/MO threads for each proof node.

Pre-computes corpus context for proof-polish prompts, so the LLM gets
exactly the relevant threads instead of browsing the filesystem.

Retrieval strategy (staged):
  1. Tag filter: match proof node keywords against SE tags
  2. NER filter: match proof node terms against NER-discovered terms
  3. Embedding rerank: score survivors by cosine similarity to node text
  4. Return top-k with excerpts

Output: JSONL with one record per proof node, containing retrieved context.
This file is consumed by run-proof-polish-codex-p7.py --corpus-context.

Usage:
    python3 scripts/retrieve-proof-context.py
    python3 scripts/retrieve-proof-context.py --problem 7 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
STORAGE_MATH = Path(os.path.expanduser("~/code/storage/math-processed-gpu"))
STORAGE_MO = Path(os.path.expanduser("~/code/storage/mo-processed-gpu"))
STORAGE_ARSE = Path(os.path.expanduser("~/code/storage/arse"))

# Topic keywords per proof node — extracted from NODE_VERIFICATION_FOCUS
# and the node body text in problem7-wiring.json
NODE_QUERY_TERMS: dict[str, list[str]] = {
    "p7-problem": [
        "uniform lattice", "cocompact lattice", "semi-simple Lie group",
        "fundamental group", "rationally acyclic", "universal cover",
        "2-torsion", "torsion element",
    ],
    "p7-s1": [
        "symmetric space", "orbifold", "torsion", "fixed point",
        "lattice action", "non-positively curved", "isometry",
        "involution", "free action",
    ],
    "p7-s2": [
        "Poincare duality", "rational cohomology", "Bredon cohomology",
        "orbifold cohomology", "group cohomology", "uniform lattice",
        "Borel-Serre", "virtual cohomological dimension",
    ],
    "p7-s3": [
        "equivariant finiteness", "finite CW complex", "Fowler",
        "FH(Q)", "finiteness obstruction", "Wall finiteness",
        "Euler characteristic", "proper action",
    ],
    "p7-s3a": [
        "arithmetic lattice", "Fowler", "lattice extension",
        "torsion subgroup", "arithmetic group",
    ],
    "p7-s4": [
        "surgery theory", "surgery obstruction", "normal invariant",
        "L-group", "Wall surgery", "manifold realization",
        "pi-pi theorem", "Spivak normal fibration",
    ],
    "p7-s5": [
        "Smith theory", "fixed point", "mod-2 acyclic",
        "transfer homomorphism", "rational acyclic",
        "group action", "free action", "Z/2 action",
    ],
    "p7-s6": [
        "Poincare duality group", "surgery obstruction",
        "FH(Q)", "manifold realization", "closed manifold",
        "fundamental group", "rationally acyclic",
    ],
}


def load_entities(storage_dir: Path) -> list[dict]:
    with (storage_dir / "entities.json").open() as f:
        return json.load(f)


def load_tags(storage_dir: Path) -> list[dict]:
    """Load SE tags per entity."""
    path = storage_dir / "tags.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def load_pattern_tags(storage_dir: Path) -> list[dict]:
    path = storage_dir / "pattern-tags.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def load_ner_terms(storage_dir: Path) -> list[dict]:
    path = storage_dir / "ner-terms.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def load_embeddings(storage_dir: Path) -> np.ndarray:
    return np.load(storage_dir / "embeddings.npy")


def build_tag_index(entities: list[dict]) -> dict[str, list[int]]:
    """Map lowercase tag -> list of entity indices."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, ent in enumerate(entities):
        for tag in ent.get("tags", []):
            if isinstance(tag, str):
                idx[tag.lower()].append(i)
    return idx


def build_text_index(entities: list[dict]) -> list[str]:
    """Concatenate title + question + answer text for keyword search."""
    texts = []
    for ent in entities:
        parts = [
            ent.get("title", ""),
            ent.get("question-body", ""),
            ent.get("answer-body", ""),
        ]
        texts.append(" ".join(p for p in parts if p).lower())
    return texts


def keyword_match_score(text: str, terms: list[str]) -> int:
    """Count how many query terms appear in the text."""
    score = 0
    for term in terms:
        if term.lower() in text:
            score += 1
    return score


def retrieve_for_node(
    node_id: str,
    query_terms: list[str],
    entities: list[dict],
    text_index: list[str],
    tag_index: dict[str, list[int]],
    embeddings: np.ndarray | None,
    faiss_index=None,
    faiss_ids: list[int] | None = None,
    entity_id_to_idx: dict[str, int] | None = None,
    top_k: int = 5,
    candidate_limit: int = 500,
    structural_expand: int = 3,
) -> list[dict]:
    """Retrieve top-k relevant entities for a proof node.

    Four-stage pipeline:
      1. Tag filter: match relevant SE tags
      2. Keyword match: score entities by query term overlap
      3. Text embedding rerank: cosine similarity on BGE embeddings
      4. FAISS structural expansion: find structurally similar threads
         to the top text-ranked seeds via the GNN FAISS index
    """

    # Stage 1: Tag-based candidates
    candidate_scores: Counter = Counter()
    tag_terms = {
        "lattice", "lie-groups", "algebraic-topology", "group-theory",
        "homological-algebra", "differential-topology", "surgery-theory",
        "group-cohomology", "manifolds", "algebraic-groups",
        "homotopy-theory", "geometric-group-theory", "orbifolds",
        "fixed-point-theorems", "poincare-duality",
    }
    for tag in tag_terms:
        for idx in tag_index.get(tag, []):
            candidate_scores[idx] += 2

    # Stage 2: Keyword match on text
    for i, text in enumerate(text_index):
        score = keyword_match_score(text, query_terms)
        if score > 0:
            candidate_scores[i] += score

    if not candidate_scores:
        return []

    # Take top candidates by keyword/tag score
    top_candidates = candidate_scores.most_common(candidate_limit)
    candidate_indices = [idx for idx, _ in top_candidates]

    # Stage 3: Rerank by text embedding similarity if available
    if embeddings is not None and len(candidate_indices) > top_k:
        scores_arr = np.array([candidate_scores[i] for i in candidate_indices], dtype=np.float32)
        scores_arr /= scores_arr.sum()
        candidate_embs = embeddings[candidate_indices]
        query_vec = (candidate_embs.T @ scores_arr).reshape(1, -1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec /= query_norm
        emb_norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        emb_norms = np.maximum(emb_norms, 1e-8)
        candidate_embs_normed = candidate_embs / emb_norms
        sims = (candidate_embs_normed @ query_vec.T).flatten()
        kw_scores = np.array([candidate_scores[i] for i in candidate_indices], dtype=np.float32)
        kw_max = kw_scores.max()
        if kw_max > 0:
            kw_scores /= kw_max
        combined = 0.5 * kw_scores + 0.5 * sims
        ranked = np.argsort(-combined)
        text_top = [candidate_indices[j] for j in ranked[:top_k]]
    else:
        text_top = candidate_indices[:top_k]

    # Stage 4: FAISS structural expansion
    # Use text-ranked seeds to find structurally similar threads
    structural_results = []
    if faiss_index is not None and faiss_ids is not None and entity_id_to_idx is not None:
        # Map seed entity indices -> thread IDs in the FAISS index
        faiss_id_set = set(faiss_ids)
        faiss_id_to_pos = {tid: pos for pos, tid in enumerate(faiss_ids)}
        seed_faiss_positions = []
        for eidx in text_top:
            ent = entities[eidx]
            eid = ent.get("entity/id", "")
            # Entity IDs are like "se-math-8" -> thread ID 8
            try:
                tid = int(eid.split("-")[-1])
            except (ValueError, IndexError):
                continue
            if tid in faiss_id_to_pos:
                seed_faiss_positions.append(faiss_id_to_pos[tid])

        if seed_faiss_positions:
            # Get seed vectors from the FAISS index
            seed_vectors = np.zeros((len(seed_faiss_positions), faiss_index.d), dtype=np.float32)
            for i, pos in enumerate(seed_faiss_positions):
                seed_vectors[i] = faiss_index.reconstruct(pos)

            # Mean of seed vectors as query
            query_structural = seed_vectors.mean(axis=0, keepdims=True)
            import faiss as _faiss
            _faiss.normalize_L2(query_structural)

            # Search for neighbors
            n_search = structural_expand + len(seed_faiss_positions) + 5
            D, I = faiss_index.search(query_structural, n_search)

            seed_tids = set()
            for pos in seed_faiss_positions:
                seed_tids.add(faiss_ids[pos])
            text_top_ids = {entities[i].get("entity/id", "") for i in text_top}

            for pos_idx, sim in zip(I[0], D[0]):
                if pos_idx < 0:
                    continue
                tid = faiss_ids[pos_idx]
                if tid in seed_tids:
                    continue
                # Map thread ID back to entity
                for prefix in ("se-math-", "se-mo-"):
                    eid_candidate = f"{prefix}{tid}"
                    if eid_candidate in entity_id_to_idx:
                        eidx = entity_id_to_idx[eid_candidate]
                        if entities[eidx].get("entity/id", "") not in text_top_ids:
                            structural_results.append((eidx, float(sim)))
                            text_top_ids.add(entities[eidx].get("entity/id", ""))
                        break
                if len(structural_results) >= structural_expand:
                    break

    # Build results: text seeds first, then structural expansions
    results = []
    for idx in text_top:
        ent = entities[idx]
        results.append(_build_result(ent, idx, candidate_scores[idx], source="text"))

    for idx, sim in structural_results:
        ent = entities[idx]
        r = _build_result(ent, idx, 0, source="structural")
        r["structural_similarity"] = round(sim, 4)
        results.append(r)

    return results


def _build_result(ent: dict, idx: int, kw_score: int, source: str) -> dict:
    return {
        "entity_id": ent.get("entity/id", str(idx)),
        "title": ent.get("title", ""),
        "question_excerpt": ent.get("question-body", "")[:600],
        "answer_excerpt": ent.get("answer-body", "")[:800],
        "tags": ent.get("tags", [])[:10],
        "score": ent.get("score", 0),
        "retrieval_keyword_score": kw_score,
        "retrieval_source": source,
    }


def format_context_for_prompt(retrieved: list[dict]) -> str:
    """Format retrieved threads as prompt context."""
    if not retrieved:
        return ""
    lines = ["## Relevant Corpus Threads", ""]
    for i, r in enumerate(retrieved, 1):
        source_tag = f" [structural]" if r.get("retrieval_source") == "structural" else ""
        lines.append(f"### [{i}] {r['title']}{source_tag}")
        lines.append(f"**ID**: {r['entity_id']} | **Tags**: {', '.join(r['tags'])} | **Score**: {r['score']}")
        lines.append("")
        if r["question_excerpt"]:
            lines.append(f"**Question**: {r['question_excerpt']}")
            lines.append("")
        if r["answer_excerpt"]:
            lines.append(f"**Answer**: {r['answer_excerpt']}")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", type=int, default=7,
                        help="Problem number (default: 7)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Threads to retrieve per node (default: 5)")
    parser.add_argument("--candidate-limit", type=int, default=500,
                        help="Max candidates after keyword filter (default: 500)")
    parser.add_argument("--sources", nargs="+", default=["math", "mo"],
                        choices=["math", "mo", "arse"])
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path (default: data/first-proof/problem{N}-corpus-context.jsonl)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding reranking (faster, keyword-only)")
    parser.add_argument("--no-faiss", action="store_true",
                        help="Skip FAISS structural expansion")
    parser.add_argument("--structural-expand", type=int, default=3,
                        help="Extra threads to retrieve via FAISS structural similarity (default: 3)")
    args = parser.parse_args()

    if args.output is None:
        args.output = REPO_ROOT / "data" / "first-proof" / f"problem{args.problem}-corpus-context.jsonl"

    wiring_path = REPO_ROOT / "data" / "first-proof" / f"problem{args.problem}-wiring.json"
    if not wiring_path.exists():
        print(f"Wiring not found: {wiring_path}", file=sys.stderr)
        return 2

    wiring = json.loads(wiring_path.read_text())
    nodes = wiring["nodes"]

    # Load corpora (real + synthetic)
    storage_dirs = []
    if "math" in args.sources:
        storage_dirs.append(("math", STORAGE_MATH))
    if "mo" in args.sources:
        storage_dirs.append(("mo", STORAGE_MO))
    if "arse" in args.sources or STORAGE_ARSE.exists():
        if (STORAGE_ARSE / "entities.json").exists():
            storage_dirs.append(("arse", STORAGE_ARSE))

    all_entities: list[dict] = []
    source_offsets: list[tuple[str, int, int]] = []
    for source_name, sdir in storage_dirs:
        if not (sdir / "entities.json").exists():
            print(f"Skipping {source_name}: no entities.json")
            continue
        start = len(all_entities)
        ents = load_entities(sdir)
        all_entities.extend(ents)
        end = len(all_entities)
        source_offsets.append((source_name, start, end))
        print(f"Loaded {len(ents)} entities from {source_name}")

    if not all_entities:
        print("No entities loaded.", file=sys.stderr)
        return 2

    print("Building indexes...")
    t0 = time.time()
    tag_index = build_tag_index(all_entities)
    text_index = build_text_index(all_entities)
    print(f"  Tag index: {len(tag_index)} unique tags")
    print(f"  Text index: {len(text_index)} entries")

    # Load embeddings if requested
    embeddings = None
    if not args.no_embeddings:
        emb_parts = []
        for source_name, sdir in storage_dirs:
            emb_path = sdir / "embeddings.npy"
            if emb_path.exists():
                emb = np.load(emb_path)
                emb_parts.append(emb)
                print(f"  Loaded {source_name} embeddings: {emb.shape}")
            else:
                # Pad with zeros if missing
                n = sum(1 for sn, s, e in source_offsets if sn == source_name for _ in range(e - s))
                emb_parts.append(np.zeros((n, 1024), dtype=np.float32))
                print(f"  Warning: no embeddings for {source_name}, using zeros")
        if emb_parts:
            embeddings = np.vstack(emb_parts)
            print(f"  Combined embeddings: {embeddings.shape}")

    # Load FAISS structural similarity index if available
    faiss_index = None
    faiss_ids = None
    entity_id_to_idx = None
    if not args.no_faiss:
        # Build entity ID -> index mapping
        entity_id_to_idx = {}
        for i, ent in enumerate(all_entities):
            eid = ent.get("entity/id", "")
            if eid:
                entity_id_to_idx[eid] = i

        # Try loading FAISS from each source (use math as primary)
        for source_name, sdir in storage_dirs:
            faiss_path = sdir / "structural-similarity-index.faiss"
            ids_path = sdir / "structural-similarity-index.ids.json"
            if faiss_path.exists() and ids_path.exists():
                try:
                    import faiss
                    faiss_index = faiss.read_index(str(faiss_path))
                    with ids_path.open() as f:
                        faiss_ids = json.load(f)
                    print(f"  FAISS index ({source_name}): {faiss_index.ntotal} vectors, dim={faiss_index.d}")
                    break  # use first available
                except ImportError:
                    print("  Warning: faiss-cpu not installed, skipping structural expansion")
                    break
                except Exception as e:
                    print(f"  Warning: failed to load FAISS index from {source_name}: {e}")

    print(f"  Indexing took {time.time()-t0:.1f}s")

    # Retrieve for each node
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_retrieved = 0

    with args.output.open("w") as fout:
        for node in nodes:
            nid = node["id"]
            query_terms = NODE_QUERY_TERMS.get(nid, [])
            if not query_terms:
                # Fallback: extract terms from body text
                body = node.get("body_text", "")
                query_terms = [w for w in re.findall(r'\b[A-Za-z][a-z]{3,}\b', body)]

            retrieved = retrieve_for_node(
                node_id=nid,
                query_terms=query_terms,
                entities=all_entities,
                text_index=text_index,
                tag_index=tag_index,
                embeddings=embeddings,
                faiss_index=faiss_index,
                faiss_ids=faiss_ids,
                entity_id_to_idx=entity_id_to_idx,
                top_k=args.top_k,
                candidate_limit=args.candidate_limit,
                structural_expand=args.structural_expand,
            )

            prompt_context = format_context_for_prompt(retrieved)
            rec = {
                "node_id": nid,
                "n_retrieved": len(retrieved),
                "retrieved": retrieved,
                "prompt_context": prompt_context,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_retrieved += len(retrieved)

            titles = [r["title"][:60] for r in retrieved[:3]]
            print(f"  {nid:20s}: {len(retrieved)} threads — {titles}")

    print(f"\nWrote {args.output} ({total_retrieved} total retrievals)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
