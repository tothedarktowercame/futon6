#!/usr/bin/env python3
"""mark3 H8 embedding pipeline: per-MSC norms + global connections.

This stage builds BGE embeddings over concept and passage texts, with explicit
hard-negative mining for the train surface. Full-scale execution is meant for
Rob's GPU box; local/sample execution is parameterized by env knobs and can run
without mutating tracked data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp" / "mark3-embed" / "ct-sample"
STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EmbedItem:
    item_id: str
    item_type: str
    msc: str
    label: str
    text: str
    paper: str | None = None
    meta: dict | None = None


@dataclass(frozen=True)
class HardNegative:
    anchor_id: str
    positive_id: str
    negative_id: str
    score: float
    reason: str


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def cgroup_cpu_quota_count() -> int | None:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_s, period_s = cpu_max.read_text().strip().split()[:2]
    except (OSError, ValueError):
        return None
    if quota_s == "max":
        return None
    try:
        quota = int(quota_s)
        period = int(period_s)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.ceil(quota / period))


def available_cpu_count() -> int:
    """CPU count available to this job, honoring Slurm/cpuset/cgroup limits."""
    counts: list[int] = []
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        counts.append(int(slurm_cpus))
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    quota = cgroup_cpu_quota_count()
    if quota:
        counts.append(quota)
    return max(1, min(c for c in counts if c > 0)) if counts else 1


def cpu_default() -> int:
    if os.environ.get("NUM_CPU_WORKERS"):
        return env_int("NUM_CPU_WORKERS", 16)
    return min(16, available_cpu_count())


def visible_cuda_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d for d in visible.split(",") if d.strip() and d.strip() != "-1"]
        return len(devices)
    return 0


def resolve_embed_workers(device: str | None, requested: int, num_gpus: int) -> int:
    if requested < 0:
        raise ValueError("embed workers must be >= 0")
    if requested > 0:
        return requested
    if device and str(device).startswith("cuda"):
        visible = visible_cuda_count()
        if visible:
            return max(1, min(num_gpus, visible))
        return max(1, num_gpus)
    return 1


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def norm_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def singularize(label: str) -> str:
    words = label.split()
    if not words:
        return label
    last = words[-1]
    if last.endswith("ies"):
        words[-1] = last[:-3] + "y"
    elif last.endswith("s") and len(last) > 3:
        words[-1] = last[:-1]
    return " ".join(words)


def load_concept_entries(path: Path, limit: int | None = None) -> list[dict]:
    data = json.loads(path.read_text())
    entries = data.get("entries", data if isinstance(data, list) else [])
    if limit is not None:
        entries = entries[:limit]
    return list(entries)


def concept_item(entry: dict, msc: str) -> EmbedItem:
    label = entry.get("concept") or entry.get("name") or str(entry.get("concept/id", "unknown"))
    gloss = (entry.get("gloss") or {}).get("text") or ""
    deps = " ".join(entry.get("depends_on") or entry.get("depends-on") or [])
    kind = entry.get("kind") or "concept"
    df = entry.get("df")
    text = (
        f"Represent this mathematical concept for retrieval: {label}. "
        f"Kind: {kind}. Definition/gloss: {gloss}. Dependencies: {deps}."
    )
    return EmbedItem(
        item_id=f"concept:{norm_label(label)}",
        item_type="concept",
        msc=msc,
        label=label,
        text=text,
        paper=(entry.get("gloss") or {}).get("paper"),
        meta={"df": df, "kind": kind, "defined_in": entry.get("defined_in") or entry.get("defined-in")},
    )


def load_term_prior_items(path: Path, msc: str, limit: int) -> list[EmbedItem]:
    if limit <= 0 or not path.exists():
        return []
    data = json.loads(path.read_text())
    bigrams = data.get("bigram_df", {})
    terms = sorted(bigrams.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    out = []
    for term, df in terms:
        out.append(
            EmbedItem(
                item_id=f"term:{norm_label(term)}",
                item_type="term",
                msc=msc,
                label=term,
                text=f"Represent this mathematical term for retrieval: {term}. Corpus document frequency: {df}.",
                meta={"df": df},
            )
        )
    return out


def paper_id_from_path(path: Path) -> str:
    name = path.name
    if name.startswith("fable-") and name.endswith("-dp-emacs.json"):
        return name[len("fable-") : -len("-dp-emacs.json")]
    return path.stem


def split_passages(text: str, max_passages: int, max_chars: int = 900) -> list[str]:
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < 160:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            sentences = SENTENCE_RE.split(para)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) > max_chars and buf:
                    chunks.append(buf.strip())
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                chunks.append(buf.strip())
        if len(chunks) >= max_passages:
            break
    return chunks[:max_passages]


def load_passage_items(golden_dir: Path, msc: str, paper_limit: int, passages_per_paper: int) -> list[EmbedItem]:
    paths = sorted(golden_dir.glob("fable-*-dp-emacs.json"))[:paper_limit]
    out: list[EmbedItem] = []
    for path in paths:
        data = json.loads(path.read_text())
        paper = str(data.get("paper") or paper_id_from_path(path))
        for idx, passage in enumerate(split_passages(data.get("text", ""), passages_per_paper)):
            out.append(
                EmbedItem(
                    item_id=f"passage:{paper}:{idx}",
                    item_type="passage",
                    msc=msc,
                    label=f"{paper} passage {idx}",
                    text=f"Represent this mathematical passage for retrieval: {passage}",
                    paper=paper,
                    meta={"source": str(path.relative_to(ROOT))},
                )
            )
    return out


def build_items(args: argparse.Namespace) -> list[EmbedItem]:
    concepts = [concept_item(e, args.msc) for e in load_concept_entries(args.concepts, args.concept_limit)]
    terms = load_term_prior_items(args.term_prior, args.msc, args.term_limit)
    passages = load_passage_items(args.golden_dir, args.msc, args.paper_limit, args.passages_per_paper)
    items = concepts + terms + passages
    if not items:
        raise SystemExit("no embedding items built; check input paths")
    return items


def synonym_clusters(items: Sequence[EmbedItem]) -> list[list[str]]:
    by_label = {i.label.lower(): i.item_id for i in items if i.item_type in {"concept", "term"}}
    wanted = [
        ["natural transformation", "natural transformations"],
        ["monoidal category", "monoidal categories"],
        ["abelian category", "abelian categories"],
        ["model category", "model categories"],
        ["monoidal functor", "monoidal functors"],
        ["identity morphism", "identity morphisms"],
    ]
    clusters: list[list[str]] = []
    seen = set()
    for labels in wanted:
        ids = [by_label[x] for x in labels if x in by_label]
        if len(ids) >= 2:
            key = tuple(sorted(ids))
            if key not in seen:
                clusters.append(ids)
                seen.add(key)
    for label, item_id in by_label.items():
        singular = singularize(label)
        if singular != label and singular in by_label:
            key = tuple(sorted([item_id, by_label[singular]]))
            if key not in seen:
                clusters.append(list(key))
                seen.add(key)
    return clusters


def lexical_score(a: EmbedItem, b: EmbedItem) -> float:
    at = Counter(tokens(a.label + " " + a.text))
    bt = Counter(tokens(b.label + " " + b.text))
    if not at or not bt:
        return 0.0
    inter = sum((at & bt).values())
    union = sum((at | bt).values())
    return inter / union if union else 0.0


def mine_hard_negatives(
    items: Sequence[EmbedItem],
    clusters: Sequence[Sequence[str]],
    negatives_per_pair: int,
) -> list[HardNegative]:
    by_id = {i.item_id: i for i in items}
    cluster_for: dict[str, int] = {}
    for idx, cluster in enumerate(clusters):
        for item_id in cluster:
            cluster_for[item_id] = idx
    candidates = [i for i in items if i.item_type in {"concept", "term"}]
    triples: list[HardNegative] = []
    for cluster in clusters:
        for anchor_id in cluster:
            for positive_id in cluster:
                if anchor_id == positive_id:
                    continue
                anchor = by_id[anchor_id]
                scored = []
                for candidate in candidates:
                    if candidate.item_id == anchor_id:
                        continue
                    if cluster_for.get(candidate.item_id) == cluster_for.get(anchor_id):
                        continue
                    score = lexical_score(anchor, candidate)
                    scored.append((score, candidate.item_id))
                scored.sort(key=lambda kv: (-kv[0], kv[1]))
                for score, negative_id in scored[:negatives_per_pair]:
                    triples.append(
                        HardNegative(
                            anchor_id=anchor_id,
                            positive_id=positive_id,
                            negative_id=negative_id,
                            score=float(score),
                            reason="lexically-close-outside-synonym-cluster",
                        )
                    )
    return triples


def shard_ranges(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    n_shards = max(1, n_shards)
    width = math.ceil(n_items / n_shards)
    return [(i, min(n_items, i + width)) for i in range(0, n_items, width)]


def batched(items: Sequence[EmbedItem], batch_size: int) -> Iterable[list[EmbedItem]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def stable_hash_embeddings(texts: Sequence[str], dim: int) -> np.ndarray:
    arr = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for tok in tokens(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            arr[row, bucket] += sign
    return normalize_rows(arr)


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def load_bge_model(model_name_or_path: str, device: str | None):
    from sentence_transformers import SentenceTransformer

    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name_or_path, **kwargs)


def bge_device(args: argparse.Namespace) -> str | None:
    if args.device:
        return args.device
    if visible_cuda_count() > 0:
        return "cuda"
    return None


def encode_items(items: Sequence[EmbedItem], args: argparse.Namespace) -> np.ndarray:
    texts = [i.text for i in items]
    backend = args.backend
    if backend == "auto":
        backend = "bge"
    if backend == "hash":
        return stable_hash_embeddings(texts, args.hash_dim)
    model_path = args.model_out if args.model_out.exists() and any(args.model_out.iterdir()) else args.model
    device = bge_device(args)
    embed_workers = resolve_embed_workers(device, args.embed_workers, args.num_gpus)
    model_device = device
    if embed_workers > 1 and device and device.startswith("cuda"):
        # SentenceTransformer.encode(device="cuda") is single-device. Keep the
        # parent model off cuda:0; the pool below places one replica per target.
        model_device = "cpu"
    model = load_bge_model(str(model_path), model_device)
    if embed_workers > 1:
        target_devices = (
            [f"cuda:{i}" for i in range(embed_workers)]
            if device and device.startswith("cuda")
            else [device or "cpu"] * embed_workers
        )
        pool = model.start_multi_process_pool(target_devices=target_devices)
        try:
            emb = model.encode_multi_process(
                texts,
                pool,
                batch_size=args.batch_size,
                normalize_embeddings=True,
            )
        finally:
            model.stop_multi_process_pool(pool)
        return np.asarray(emb, dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def train_stage(args: argparse.Namespace) -> dict:
    items = build_items(args)
    clusters = synonym_clusters(items)
    triples = mine_hard_negatives(items, clusters, args.hard_negatives)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "hard-negatives.jsonl", (asdict(t) for t in triples))
    write_json(args.output_dir / "train-items.json", [asdict(i) for i in items])
    report = {
        "stage": "train",
        "model": args.model,
        "backend": args.backend,
        "num_items": len(items),
        "num_clusters": len(clusters),
        "num_hard_negative_triples": len(triples),
        "epochs": args.epochs,
        "model_out": str(args.model_out),
    }
    if args.epochs > 0:
        if args.backend == "hash":
            raise SystemExit("cannot train with --backend hash; use --backend bge")
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        by_id = {i.item_id: i for i in items}
        examples = [
            InputExample(texts=[by_id[t.anchor_id].text, by_id[t.positive_id].text, by_id[t.negative_id].text])
            for t in triples
        ]
        if not examples:
            raise SystemExit("no hard-negative triples available for training")
        model = load_bge_model(args.model, args.device)
        loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
        loss = losses.TripletLoss(model=model)
        model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, warmup_steps=0, output_path=str(args.model_out))
        report["trained"] = True
    else:
        report["trained"] = False
        report["note"] = "epochs=0: emitted hard-negative training surface without fine-tuning"
    write_json(args.output_dir / "train-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def infer_stage(args: argparse.Namespace) -> dict:
    items = build_items(args)
    embeddings = encode_items(items, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concept_idx = [idx for idx, item in enumerate(items) if item.item_type in {"concept", "term"}]
    passage_idx = [idx for idx, item in enumerate(items) if item.item_type == "passage"]
    np.save(args.output_dir / f"{args.space}-embeddings.npy", embeddings)
    np.save(args.output_dir / f"{args.space}-concept-embeddings.npy", embeddings[concept_idx])
    np.save(args.output_dir / f"{args.space}-passage-embeddings.npy", embeddings[passage_idx])
    write_json(args.output_dir / f"{args.space}-ids.json", [i.item_id for i in items])
    write_json(args.output_dir / f"{args.space}-items.json", [asdict(i) for i in items])
    report = {
        "stage": "infer",
        "space": args.space,
        "backend": args.backend,
        "model": args.model,
        "items": len(items),
        "concept_or_term_items": len(concept_idx),
        "passage_items": len(passage_idx),
        "embedding_dim": int(embeddings.shape[1]),
        "output_dir": str(args.output_dir),
        "env": {
            "NUM_GPUS": args.num_gpus,
            "EMBED_BATCH": args.batch_size,
            "NUM_CPU_WORKERS": args.num_workers,
            "NUM_SHARDS": args.num_shards,
            "EMBED_WORKERS": resolve_embed_workers(bge_device(args), args.embed_workers, args.num_gpus),
        },
    }
    write_json(args.output_dir / "infer-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    normed = normalize_rows(embeddings.astype(np.float32))
    return normed @ normed.T


def eval_stage(args: argparse.Namespace) -> dict:
    items_data = json.loads((args.output_dir / f"{args.space}-items.json").read_text())
    items = [EmbedItem(**row) for row in items_data]
    ids = [i.item_id for i in items]
    id_to_idx = {item_id: idx for idx, item_id in enumerate(ids)}
    embeddings = np.load(args.output_dir / f"{args.space}-embeddings.npy")
    sims = cosine_matrix(embeddings)
    clusters = synonym_clusters(items)
    pair_sims: list[float] = []
    ranks: list[int] = []
    cluster_examples: list[dict] = []
    for cluster in clusters:
        cluster_indices = [id_to_idx[x] for x in cluster if x in id_to_idx]
        cluster_ranks: list[int] = []
        cluster_sims: list[float] = []
        for idx in cluster_indices:
            target = set(cluster_indices) - {idx}
            if not target:
                continue
            order = np.argsort(-sims[idx])
            order = [int(x) for x in order if int(x) != idx]
            best_rank = min(order.index(t) + 1 for t in target)
            ranks.append(best_rank)
            cluster_ranks.append(best_rank)
            pair_sims.extend(float(sims[idx, t]) for t in target)
            cluster_sims.extend(float(sims[idx, t]) for t in target)
        if cluster_ranks and len(cluster_examples) < 12:
            cluster_examples.append(
                {
                    "labels": [items[idx].label for idx in cluster_indices],
                    "mean_pair_similarity": float(np.mean(cluster_sims)),
                    "best_member_rank_mean": float(np.mean(cluster_ranks)),
                    "recall_at_5": float(sum(1 for r in cluster_ranks if r <= 5) / len(cluster_ranks)),
                }
            )
    rng = random.Random(42)
    random_pairs = []
    for _ in range(min(2000, max(1, len(items) * 4))):
        a, b = rng.sample(range(len(items)), 2)
        random_pairs.append(float(sims[a, b]))
    concept_indices = [idx for idx, item in enumerate(items) if item.item_type in {"concept", "term"}]
    passage_indices = [idx for idx, item in enumerate(items) if item.item_type == "passage"]
    retrieval_hits = 0
    retrieval_total = 0
    for cidx in concept_indices:
        item = items[cidx]
        label_toks = set(tokens(item.label))
        if not label_toks:
            continue
        scored = sorted(((float(sims[cidx, pidx]), pidx) for pidx in passage_indices), reverse=True)
        top = scored[: args.retrieval_k]
        if not top:
            continue
        retrieval_total += 1
        for _, pidx in top:
            passage = items[pidx]
            p_toks = set(tokens(passage.text))
            if label_toks <= p_toks or (item.paper and item.paper == passage.paper):
                retrieval_hits += 1
                break
    report = {
        "stage": "eval",
        "space": args.space,
        "num_items": len(items),
        "num_synonym_clusters": len(clusters),
        "synonym_pair_mean": float(np.mean(pair_sims)) if pair_sims else None,
        "random_pair_mean": float(np.mean(random_pairs)) if random_pairs else None,
        "synonym_minus_random_margin": (
            float(np.mean(pair_sims) - np.mean(random_pairs)) if pair_sims and random_pairs else None
        ),
        "synonym_recall_at_5": float(sum(1 for r in ranks if r <= 5) / len(ranks)) if ranks else None,
        "mean_synonym_rank": float(np.mean(ranks)) if ranks else None,
        "cluster_examples": cluster_examples,
        "concept_to_passage_recall_at_k": float(retrieval_hits / retrieval_total) if retrieval_total else None,
        "retrieval_k": args.retrieval_k,
    }
    write_json(args.output_dir / "eval-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_metrics:
        if report["synonym_recall_at_5"] is None or report["synonym_recall_at_5"] < args.min_synonym_recall:
            raise SystemExit(f"synonym recall gate failed: {report['synonym_recall_at_5']}")
        if (
            report["synonym_minus_random_margin"] is None
            or report["synonym_minus_random_margin"] < args.min_synonym_margin
        ):
            raise SystemExit(f"synonym margin gate failed: {report['synonym_minus_random_margin']}")
    return report


def run_sample(args: argparse.Namespace) -> dict:
    train_stage(args)
    infer_stage(args)
    report = eval_stage(args)
    print(f"sample report: {args.output_dir / 'eval-report.json'}")
    return report


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--msc", default="ct", help="MSC tag/name for this run (default: ct)")
    parser.add_argument("--space", choices=["per-msc", "global"], default="per-msc", help="embedding space to emit")
    parser.add_argument("--concepts", type=Path, default=ROOT / "data" / "concept-encyclopedia-ct.json")
    parser.add_argument("--term-prior", type=Path, default=ROOT / "data" / "ct-term-prior.json")
    parser.add_argument("--golden-dir", type=Path, default=ROOT / "data" / "showcases" / "ct-anatomy" / "golden")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("MARK3_EMBED_OUT", DEFAULT_OUT)))
    parser.add_argument("--model", default=os.environ.get("BGE_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--model-out", type=Path, default=Path(os.environ.get("MARK3_EMBED_MODEL_OUT", DEFAULT_OUT / "bge-ft")))
    parser.add_argument("--backend", choices=["auto", "bge", "hash"], default=os.environ.get("MARK3_EMBED_BACKEND", "auto"))
    parser.add_argument("--device", default=os.environ.get("EMBED_DEVICE"))
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=env_int("EMBED_WORKERS", 0),
        help=(
            "BGE replica fanout workers. Default 0 = auto: all visible CUDA "
            "GPUs up to NUM_GPUS; 1 = single SentenceTransformer.encode path."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=env_int("EMBED_BATCH", 256))
    parser.add_argument("--num-gpus", type=int, default=env_int("NUM_GPUS", 8))
    parser.add_argument("--num-workers", type=int, default=cpu_default())
    parser.add_argument("--num-shards", type=int, default=env_int("NUM_SHARDS", 8))
    parser.add_argument("--concept-limit", type=int, default=env_int("MARK3_CONCEPT_LIMIT", 200))
    parser.add_argument("--term-limit", type=int, default=env_int("MARK3_TERM_LIMIT", 120))
    parser.add_argument("--paper-limit", type=int, default=env_int("MARK3_PAPER_LIMIT", 24))
    parser.add_argument("--passages-per-paper", type=int, default=env_int("MARK3_PASSAGES_PER_PAPER", 2))
    parser.add_argument("--hash-dim", type=int, default=env_int("MARK3_HASH_DIM", 384))


def build_parser() -> argparse.ArgumentParser:
    epilog = """Environment knobs:
  NUM_GPUS=8                 default GPU count hint for Rob's 8-GPU box
  EMBED_BATCH=256            BGE encode/train batch size
  EMBED_WORKERS=0            0 = auto fanout over visible CUDA GPUs, up to NUM_GPUS
  NUM_CPU_WORKERS=16         dataloader workers; honors Slurm/cpuset/cgroup affinity
  NUM_SHARDS=8               shard hint for full-scale corpus partitioning
  BGE_MODEL=BAAI/bge-small-en-v1.5
  MARK3_EMBED_OUT=tmp/mark3-embed/ct-sample
  MARK3_CONCEPT_LIMIT=200 MARK3_TERM_LIMIT=120 MARK3_PAPER_LIMIT=24

Stages:
  train       emit hard-negative triplets; optionally fine-tune BGE with TripletLoss
  infer       produce concept/term and passage embeddings for per-MSC or global space
  eval        synonym-cluster cohesion + basic concept->passage retrieval sanity
  run-sample  train(epochs default 0) + infer + eval with gates on sample metrics
"""
    parser = argparse.ArgumentParser(
        description="mark3 H8 BGE embedding stage for per-MSC norms and global connections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ["train", "infer", "eval", "run-sample"]:
        p = sub.add_parser(name, formatter_class=argparse.RawDescriptionHelpFormatter)
        add_common_args(p)
        p.add_argument("--hard-negatives", type=int, default=env_int("HARD_NEGATIVES", 3))
        p.add_argument("--epochs", type=int, default=0 if name == "run-sample" else env_int("EMBED_EPOCHS", 0))
        p.add_argument("--retrieval-k", type=int, default=5)
        p.add_argument("--require-metrics", action="store_true", default=name == "run-sample")
        p.add_argument("--min-synonym-recall", type=float, default=float(os.environ.get("MIN_SYNONYM_RECALL", 0.60)))
        p.add_argument("--min-synonym-margin", type=float, default=float(os.environ.get("MIN_SYNONYM_MARGIN", 0.05)))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage == "train":
        train_stage(args)
    elif args.stage == "infer":
        infer_stage(args)
    elif args.stage == "eval":
        eval_stage(args)
    elif args.stage == "run-sample":
        run_sample(args)
    else:
        parser.error(f"unknown stage: {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
