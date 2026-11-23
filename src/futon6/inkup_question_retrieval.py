"""Streaming lexical and dense retrieval of provenance-bearing reader questions.

Two independent methods over a local Stack Exchange store directory that holds
``entities.json`` (one JSON array; may be gigabytes, so it is stream-decoded)
and ``embeddings.npy`` (shape ``(N, 1024)`` float32, normalised bge-large-en-v1.5,
memory-mapped and scored in chunks).

* lexical: BM25 over title + question body. The answer body is never read.
* embedding: cosine against the stored vectors. A stored vector was made from
  title + first 500 question chars + first 500 answer chars, so an embedding hit
  may be driven by answer text; every embedding candidate says so.

The embedder is injected so tests never load the BGE model.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

MODEL_NAME = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
WINDOW = 500
TOP_PER_METHOD = 100
MIN_ALIGNMENT_SAMPLE = 200
ALIGNMENT_THRESHOLD = 0.95
STORES_ENV = "INKUP_QA_STORES"
CORPUS_LIMITS = [
    "One answer per question.",
    "No comments: questions expressed only in comments cannot be retrieved.",
    "Stored vectors mix in answer text: an embedding hit may be driven by the answer.",
]
EMBEDDING_MATCH_SCOPE = "title + first 500 chars of question + first 500 chars of answer"

# A query term marked "anchor": true (see holes/inkup/pilots/*/retrieval-query.json) must be
# matched by every lexical candidate. An anchor is a multi-word term with LaTeX, so it is
# rarely a contiguous phrase in a post: it is matched when the document (title + question
# body) contains at least this share of its distinct content tokens, and never fewer than
# ANCHOR_MIN_TOKENS of them (a one- or two-token anchor needs what it has). Content tokens
# are the tokenizer's tokens of length >= 2: LaTeX leaves single letters (f, p, l) and
# command names (alpha, infty) behind, and single letters would match almost every post.
ANCHOR_TOKEN_SHARE = 0.6
ANCHOR_MIN_TOKENS = 2

Encode = Callable[[list[str]], np.ndarray]


def encode_texts(encode, texts: list[str]) -> np.ndarray:
    """`encode` is a callable or any object with an ``encode(texts)`` method."""
    return encode(texts) if callable(encode) else encode.encode(texts)


def encode_query_text(encode, text: str) -> np.ndarray:
    """Query vector; an embedder may define ``encode_query`` (BGE uses a query prefix)."""
    if hasattr(encode, "encode_query"):
        return np.asarray(encode.encode_query(text), dtype=np.float32)
    return encode_texts(encode, [text])[0]


# --- streaming JSON -------------------------------------------------------

def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Yield the objects of a JSON array file without materialising the array."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as fh:
        buf, pos, eof, started, need_more = "", 0, False, False, False
        while True:
            # Read on demand: an object larger than the buffer asks for more when a
            # decode attempt fails, so chunk size bounds memory, never object size.
            if not eof and (need_more or len(buf) - pos < 1):
                part = fh.read(chunk_size)
                if part:
                    buf, pos = buf[pos:] + part, 0
                else:
                    eof = True
                need_more = False
            length = len(buf)
            while pos < length and buf[pos] in " \t\r\n":
                pos += 1
            if pos >= length:
                if eof:
                    break
                need_more = True
                continue
            if not started:
                if buf[pos] != "[":
                    raise ValueError(f"expected JSON array: {path}")
                started, pos = True, pos + 1
                continue
            if buf[pos] == "]":
                return
            if buf[pos] == ",":
                pos += 1
                continue
            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                need_more = True  # object incomplete: read more, then try again
                continue
            if not isinstance(obj, dict):
                raise ValueError(f"non-object member in {path}")
            yield obj
            pos = end
    if not started:
        raise ValueError(f"empty file: {path}")
    raise ValueError(f"truncated JSON array: {path}")


# --- embedder -------------------------------------------------------------

def embedding_text(entity: dict) -> str:
    """The text a stored vector was made from (see compute_qa_embeddings)."""
    parts = [entity.get("title") or "",
             (entity.get("question-body") or "")[:WINDOW],
             (entity.get("answer-body") or "")[:WINDOW]]
    return ". ".join(part for part in parts if part)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class BgeEmbedder:
    """The production embedder; loaded lazily so import stays cheap."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None,
                 batch_size: int = 64):
        self.model_name, self.device, self.batch_size = model_name, device, batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=self.device)
        return normalize_rows(np.asarray(model.encode(texts, batch_size=self.batch_size,
                                                      show_progress_bar=False)))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([QUERY_PREFIX + text])[0]


# --- row-alignment check --------------------------------------------------

def sample_indices(size: int, count: int) -> list[int]:
    """`count` evenly spaced rows; `count` is never below the minimum sample."""
    return sorted(set(np.linspace(0, size - 1, max(1, min(size, count)),
                                  dtype=np.int64).tolist()))


def entities_at_indices(path: Path, indices: Sequence[int]) -> list[dict]:
    wanted, found = set(indices), {}
    for row, entity in enumerate(iter_json_array(path)):
        if row in wanted:
            found[row] = entity
        if row >= max(wanted):
            break
    if wanted != found.keys():
        raise ValueError(f"missing entity rows: {sorted(wanted - found.keys())[:5]}")
    return [found[row] for row in indices]


def check_row_alignment(entities_path: Path, embeddings_path: Path, encode: Encode,
                        sample_count: int = MIN_ALIGNMENT_SAMPLE,
                        sample_entities: list[dict] | None = None) -> dict:
    """Re-embed sampled rows and report the fraction whose nearest stored vector is their own.

    The nearest stored vector is searched over the whole store, in chunks. `sample_entities`
    lets a caller that has already streamed the file hand over the sampled rows.
    """
    stored = np.load(embeddings_path, mmap_mode="r")
    if stored.ndim != 2:
        raise ValueError(f"embeddings must be 2-dimensional: {embeddings_path}")
    requested = max(MIN_ALIGNMENT_SAMPLE, sample_count)
    indices = sample_indices(len(stored), requested)
    entities = sample_entities if sample_entities is not None else entities_at_indices(
        entities_path, indices)
    if len(entities) != len(indices):
        raise ValueError("sample_entities does not match the sampled rows")
    texts = [embedding_text(e) for e in entities]
    generated = normalize_rows(encode_texts(encode, texts))
    if generated.shape[1] != stored.shape[1]:
        raise ValueError(f"embedding width {generated.shape[1]} != stored {stored.shape[1]}")
    own = np.stack([np.asarray(stored[i]) for i in indices])
    nearest = np.full(len(indices), -1)
    best = np.full(len(indices), -np.inf)
    chunk = max(1, (1 << 22) // (stored.shape[1] * 4))  # about 4 MiB of vectors
    for start in range(0, len(stored), chunk):
        block = normalize_rows(np.asarray(stored[start:start + chunk]))
        scores = generated @ block.T
        local = scores.argmax(axis=1)
        value = scores[np.arange(len(indices)), local]
        take = value > best
        nearest[take], best[take] = start + local[take], value[take]
    hits = int(np.sum(nearest == np.asarray(indices)))
    return {
        "sample_count": len(indices),
        "sample_count_requested": requested,
        "sample_capped_by_store_size": len(indices) < requested,
        "sample_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "own_row_nearest_count": hits,
        "own_row_nearest_fraction": hits / len(indices),
        "mean_own_cosine": float(np.mean(np.sum(generated * normalize_rows(own), axis=1))),
        "trusted": hits / len(indices) >= ALIGNMENT_THRESHOLD,
        "threshold": ALIGNMENT_THRESHOLD,
    }


# --- query ----------------------------------------------------------------

def load_query(path: Path) -> dict:
    """Accept {"vocabulary"|"terms": [{term, source, category?}, ...]} or a bare list."""
    query = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(query, list):
        query = {"vocabulary": query}
    if not isinstance(query, dict):
        raise ValueError("query must be a JSON object or a list of terms")
    items = query.get("vocabulary") or query.get("terms")
    if not isinstance(items, list) or not items:
        raise ValueError("query needs a non-empty list of terms")
    for item in items:
        if not isinstance(item, dict) or not str(item.get("term") or "").strip():
            raise ValueError("each query term needs a non-empty 'term'")
        if item.get("source") in (None, "", [], {}):
            raise ValueError(f"query term {item.get('term')!r} needs a source note")
    query["vocabulary"] = items
    return query


def anchor_terms(query: dict) -> list[str]:
    """The query terms marked "anchor": true, in query order; empty when none is marked."""
    return list(dict.fromkeys(item["term"].strip().lower() for item in query["vocabulary"]
                              if item.get("anchor") is True))


def query_terms(query: dict) -> list[str]:
    return list(dict.fromkeys(item["term"].strip().lower() for item in query["vocabulary"]))


def query_embedding_text(query: dict) -> str:
    """Terms grouped by category when categories are given; the text is reported verbatim."""
    categories: dict[str, list[str]] = {}
    for item in query["vocabulary"]:
        group = str(item.get("category") or item.get("kind") or "term")
        categories.setdefault(group, []).append(item["term"])
    return ". ".join(f"{group}: " + "; ".join(terms) for group, terms in categories.items())


def query_embedding_texts(query: dict) -> list[str]:
    """The texts to embed. A query file may state them itself under "embedding_queries"
    ([{"text": ...}, ...]): a sentence about the problem is a better dense query than a
    list of terms. Without that key the grouped term list is the single query text."""
    stated = query.get("embedding_queries")
    if stated is None:
        return [query_embedding_text(query)]
    if (not isinstance(stated, list) or not stated or
            not all(isinstance(q, dict) and str(q.get("text") or "").strip() for q in stated)):
        raise ValueError("embedding_queries must be a non-empty list of {text: ...}")
    return [q["text"].strip() for q in stated]


# --- lexical: BM25 over title + question body -----------------------------

K1, B, TITLE_BOOST = 1.5, 0.75, 3.0


def _fold(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_fold(token) for token in re.findall(r"\w+", text.lower())]


def _phrase_count(tokens: list[str], phrase: list[str]) -> int:
    """Occurrences of `phrase` as a contiguous run in `tokens` (the caller has checked that
    its first token is present, which is what keeps this cheap on a large corpus)."""
    if len(phrase) == 1:
        return tokens.count(phrase[0])
    n, count, start = len(phrase), 0, 0
    while True:
        try:
            start = tokens.index(phrase[0], start)
        except ValueError:
            return count
        if tokens[start:start + n] == phrase:
            count += 1
        start += 1


def _counts(title: list[str], body: list[str], phrases: dict[str, list[str]]) -> dict[str, int]:
    """Per-term occurrences (title + body) for the terms whose first token is present."""
    present = set(title) | set(body)
    counts = {}
    for term, phrase in phrases.items():
        if phrase[0] in present:
            count = _phrase_count(title, phrase) + _phrase_count(body, phrase)
            if count:
                counts[term] = count
    return counts


def anchor_required_tokens(anchor: str) -> tuple[set[str], int]:
    """(distinct content tokens, how many a document must contain) for one anchor term."""
    tokens = {token for token in tokenize(anchor) if len(token) >= 2}
    if not tokens:
        raise ValueError(f"anchor term has no content token of length >= 2: {anchor!r}")
    return tokens, min(len(tokens), max(ANCHOR_MIN_TOKENS, int(np.ceil(ANCHOR_TOKEN_SHARE * len(tokens)))))


def anchors_admitting(document_tokens: set[str], anchors: dict[str, tuple[set[str], int]]) -> list[str]:
    """The anchor terms that `document_tokens` matches, in the order the anchors were given."""
    return [anchor for anchor, (tokens, required) in anchors.items()
            if len(tokens.intersection(document_tokens)) >= required]


ANCHOR_RULE = (f"a document is a lexical candidate only if, for at least one anchor term, it contains "
               f"at least {ANCHOR_TOKEN_SHARE:.0%} of that term's distinct tokens of length >= 2 "
               f"(never fewer than {ANCHOR_MIN_TOKENS}), in title or question body")


def _bm25(title_n: int, body_n: int, doc_len: int, avgdl: float, df: int, docs: int) -> float:
    """BM25 term weight; a title occurrence counts TITLE_BOOST times a body occurrence."""
    idf = float(np.log(1.0 + (docs - df + 0.5) / (df + 0.5)))
    tf = body_n + TITLE_BOOST * title_n
    if not tf:
        return 0.0
    return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc_len / (avgdl or 1.0)))


# --- retrieval ------------------------------------------------------------

def _push(heap: list, item: tuple, limit: int) -> None:
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def lexical_retrieve(path: Path, terms: Sequence[str], limit: int,
                     stats: dict | None = None, rows_to_grab: Sequence[int] = (),
                     anchors: Sequence[str] = ()) -> dict:
    """BM25 over title + question body (the answer body is never read).

    `anchors`: terms that must be matched (see ANCHOR_RULE). With none, every document that
    matches any term is scored; with one or more, only documents matching an anchor are.
    Scoring is unchanged; the anchor rule only filters, and each hit records the anchors
    that admitted it under "admitted_by".

    `stats` from lexical_stats supplies document frequencies; without it a first pass runs.
    Returns {"hits": [(score, row, entity, matched_terms, passage), ...] best first,
             "grabbed": {row: entity} for the requested rows, "docs": documents seen}.
    """
    phrases = _phrases(terms)
    grabbed = dict((stats or {}).get("grabbed", {}))
    wanted = set(rows_to_grab) - grabbed.keys()
    if not phrases:
        return {"hits": [], "grabbed": grabbed, "docs": 0}
    if stats is None:
        stats = lexical_stats(path, phrases)
    docs, total, df = stats["docs"], stats["total"], stats["df"]
    unknown = [a for a in anchors if a not in phrases]
    if unknown:
        raise ValueError(f"anchor terms must be among the query terms: {unknown}")
    anchor_specs = {a: anchor_required_tokens(a) for a in anchors}
    heap, passing = [], 0
    for row, entity in enumerate(iter_json_array(path)):
        if row in wanted:
            grabbed[row] = entity
        title = tokenize(entity.get("title") or "")
        body = tokenize(entity.get("question-body") or "")
        admitted = anchors_admitting(set(title).union(body), anchor_specs)
        if anchor_specs and not admitted:
            continue
        passing += 1
        score, matched = 0.0, []
        for term in _counts(title, body, phrases):
            phrase = phrases[term]
            part = _bm25(_phrase_count(title, phrase), _phrase_count(body, phrase),
                         len(title) + len(body), total / docs if docs else 0.0, df[term], docs)
            if part:
                score, matched = score + part, matched + [term]
        if score:
            passage = question_passage(entity.get("question-body") or "", matched)
            _push(heap, (score, -row, row, entity, matched, passage, admitted), limit)
    wanted = wanted - grabbed.keys()
    if wanted:
        raise ValueError(f"store rows missing from entities.json: {sorted(wanted)[:5]}")
    return {"hits": [(x[0], x[2], x[3], x[4], x[5]) for x in sorted(heap, reverse=True)],
            "admitted_by": {x[2]: x[6] for x in heap},
            "anchor_filter": ({"anchors": list(anchors), "rule": ANCHOR_RULE,
                               "required_tokens": {a: required for a, (_t, required) in anchor_specs.items()},
                               "documents_passing": passing, "documents": docs}
                              if anchors else None),
            "grabbed": grabbed, "docs": docs}


def _phrases(terms: Sequence[str]) -> dict[str, list[str]]:
    return {term: phrase for term in terms if (phrase := tokenize(term))}


def lexical_stats(path: Path, phrases: dict[str, list[str]],
                  grab: Sequence[int] = ()) -> dict:
    """One streamed pass: document count, total length, per-phrase document frequency, and
    optionally the entities at `grab` rows (used for the alignment sample)."""
    docs, total, grabbed = 0, 0, {}
    df = {term: 0 for term in phrases}
    wanted = set(grab)
    for row, entity in enumerate(iter_json_array(path)):
        title = tokenize(entity.get("title") or "")
        body = tokenize(entity.get("question-body") or "")
        docs, total = docs + 1, total + len(title) + len(body)
        for term in _counts(title, body, phrases):
            df[term] += 1
        if row in wanted:
            grabbed[row] = entity
    return {"docs": docs, "total": total, "df": df, "grabbed": grabbed}


def question_passage(body: str, matched: Sequence[str], width: int = 700) -> str:
    """A verbatim window of the question body around the first matched term."""
    if not body:
        return ""
    for term in matched:
        patterns = [r"(?<!\w)" + re.escape(term) + r"(?!\w)"]
        first = tokenize(term)[:1]
        if first:
            patterns.append(r"(?<!\w)" + re.escape(first[0]) + r"\w*")
        for pattern in patterns:
            found = re.search(pattern, body, re.I)
            if found:
                start = max(0, found.start() - width // 4)
                return body[start:start + width]
    return body[:width]


def embedding_top_rows(path: Path, query_vector: np.ndarray, limit: int,
                       chunk_size: int = 8192,
                       best_query: dict[int, int] | None = None) -> list[tuple[float, int]]:
    """Top `limit` rows by positive cosine, memory-mapped and scored in chunks.

    `query_vector` is one vector or a (queries, width) matrix. With several queries a row's
    score is its best cosine over them, and `best_query`, when given, is filled with
    row -> index of the query that produced that score.
    Rows with no similarity to any query (cosine <= 0) are not candidates.
    """
    matrix = np.load(path, mmap_mode="r")
    queries = np.asarray(query_vector, dtype=np.float32)
    queries = normalize_rows(queries.reshape(1, -1) if queries.ndim == 1 else queries)
    if matrix.ndim != 2 or matrix.shape[1] != queries.shape[1]:
        raise ValueError(f"stored vectors {matrix.shape} do not match query width {queries.shape[1]}")
    heap = []
    which: dict[int, int] = {}
    for start in range(0, len(matrix), chunk_size):
        chunk = np.asarray(matrix[start:start + chunk_size], dtype=np.float32)
        every = normalize_rows(chunk) @ queries.T
        scores, winners = every.max(axis=1), every.argmax(axis=1)
        take = min(limit, len(scores))
        for local in np.argpartition(scores, len(scores) - take)[len(scores) - take:]:
            if scores[local] > 0.0:
                row = start + int(local)
                which[row] = int(winners[local])
                _push(heap, (float(scores[local]), -row, row), limit)
    top = [(x[0], x[2]) for x in sorted(heap, reverse=True) if x[0] > 0.0]
    if best_query is not None:
        best_query.update({row: which[row] for _score, row in top})
    return top


def make_candidate(entity: dict, site: str, method: str, rank: int, score: float,
                   reason: dict, passage: str) -> dict:
    eid = str(entity.get("entity/id") or "")
    digits = re.search(r"(\d+)$", eid)
    candidate = {
        "entity_id": eid,
        "site": site,
        "thread_id": digits.group(1) if digits else eid,
        "title": entity.get("title") or "",
        "passage": passage,
        "tags": list(entity.get("tags") or []),
        "created": entity.get("created"),
        "method": method,
        "rank": rank,
        "score": float(score),
        "retrieval_reason": reason,
    }
    if method == "embedding":
        candidate["matched_vector_scope"] = EMBEDDING_MATCH_SCOPE
        candidate["answer_text_may_drive_match"] = True
    return candidate


# --- stores and the run ---------------------------------------------------

def parse_stores(values: Sequence[str] | None = None, env: dict | None = None) -> list[tuple[str, Path]]:
    """`SITE=PATH` entries from arguments, else INKUP_QA_STORES (os.pathsep-separated). No default."""
    raw = [value for value in (values or []) if value and value.strip()]
    if not raw:
        setting = (os.environ if env is None else env).get(STORES_ENV, "")
        raw = [part for part in setting.split(os.pathsep) if part.strip()]
    if not raw:
        raise ValueError(f"no store supplied: use --store SITE=PATH or set {STORES_ENV}")
    stores = []
    for value in raw:
        site, sep, raw_path = value.partition("=")
        site, raw_path = site.strip(), raw_path.strip()
        if not sep or not site or not raw_path:
            raise ValueError(f"store must be SITE=PATH, got {value!r}")
        path = Path(raw_path)
        for name in ("entities.json", "embeddings.npy"):
            if not (path / name).is_file():
                raise ValueError(f"incomplete store {site}={raw_path}: missing {name}")
        stores.append((site, path))
    return stores


def _file_facts(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def git_sha(repo: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo or "."), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def retrieve_site(site: str, store: Path, terms: Sequence[str], query_vector: np.ndarray,
                  encode: Encode, limit: int = TOP_PER_METHOD,
                  sample_count: int = MIN_ALIGNMENT_SAMPLE,
                  anchors: Sequence[str] = ()) -> tuple[list[dict], dict]:
    """Both methods over one store. entities.json is streamed exactly twice: pass one collects
    document frequencies and the alignment sample, pass two scores and fetches the embedding rows."""
    entities, embeddings = store / "entities.json", store / "embeddings.npy"
    phrases = _phrases(terms)
    stored_rows = int(np.load(embeddings, mmap_mode="r").shape[0])
    sample = sample_indices(stored_rows, max(MIN_ALIGNMENT_SAMPLE, sample_count))
    stats = lexical_stats(entities, phrases, sample)
    rows_match = stats["docs"] == stored_rows
    if rows_match:
        alignment = check_row_alignment(entities, embeddings, encode, sample_count,
                                        sample_entities=[stats["grabbed"][row] for row in sample])
    else:
        alignment = {"trusted": False, "not_run": "row count mismatch"}
    report = {
        "site": site,
        "files": {"entities.json": _file_facts(entities), "embeddings.npy": _file_facts(embeddings)},
        "row_alignment": alignment,
    }
    dense = []
    best_query: dict[int, int] = {}
    if not rows_match:
        # Vectors are tied to entities by row position; unequal counts mean positions disagree.
        report["embedding_skipped"] = (
            f"row count mismatch: {stats['docs']} entities but {stored_rows} stored vectors")
    elif alignment["trusted"]:
        dense = embedding_top_rows(embeddings, query_vector, limit, best_query=best_query)
    else:
        report["embedding_skipped"] = ("row alignment below threshold: stored vectors "
                                       "could not be tied to their entity rows")
    lexical = lexical_retrieve(entities, terms, limit, stats, rows_to_grab=[row for _, row in dense],
                                anchors=anchors)
    report["row_count"] = {"entities": lexical["docs"], "stored_vectors": stored_rows}
    report["row_count"]["entities_match_vectors"] = (
        report["row_count"]["entities"] == report["row_count"]["stored_vectors"])
    candidates = [make_candidate(entity, site, "lexical", rank, score,
                                 ({"matched_terms": matched, "admitted_by": lexical["admitted_by"][row]}
                                  if anchors else {"matched_terms": matched}), passage)
                  for rank, (score, row, entity, matched, passage) in enumerate(lexical["hits"], 1)]
    candidates += [make_candidate(lexical["grabbed"][row], site, "embedding", rank, score,
                                  {"query_text": None, "query_index": best_query.get(row, 0),
                                   "match_scope": EMBEDDING_MATCH_SCOPE},
                                  (lexical["grabbed"][row].get("question-body") or "")[:WINDOW])
                   for rank, (score, row) in enumerate(dense, 1)]
    if anchors:
        report["anchor_filter"] = lexical["anchor_filter"]
    report["counts"] = {"lexical": len(lexical["hits"]), "embedding": len(dense)}
    return candidates, report


def run_retrieval(query_path: Path, stores: Sequence[tuple[str, Path]], output_dir: Path,
                  encode: Encode, encode_query: Callable[[str], np.ndarray] | None = None,
                  run_id: str | None = None, limit: int = TOP_PER_METHOD,
                  sample_count: int = MIN_ALIGNMENT_SAMPLE,
                  repo: Path | None = None) -> Path:
    """Write question-candidates.jsonl and manifest.json into `output_dir`."""
    query = load_query(query_path)
    if not stores:
        raise ValueError(f"no store supplied: use --store SITE=PATH or set {STORES_ENV}")
    output_dir = Path(output_dir)
    for name in ("question-candidates.jsonl", "manifest.json"):
        if (output_dir / name).exists():
            raise ValueError(f"refusing to overwrite {output_dir / name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    encode_query = encode_query or (lambda text: encode_query_text(encode, text))
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    terms = query_terms(query)
    anchors = anchor_terms(query)
    unknown = [a for a in anchors if a not in terms]
    if unknown:
        raise ValueError(f"anchor terms must be query terms: {unknown}")
    query_texts = query_embedding_texts(query)
    query_vector = np.stack([np.asarray(encode_query(text), dtype=np.float32)
                             for text in query_texts])
    candidates, reports = [], []
    for site, store in stores:
        found, report = retrieve_site(site, store, terms, query_vector, encode, limit, sample_count, anchors)
        for candidate in found:
            if candidate["method"] == "embedding":
                reason = candidate["retrieval_reason"]
                reason["query_text"] = query_texts[reason["query_index"]]
        candidates += found
        reports.append(report)
    with (output_dir / "question-candidates.jsonl").open("w", encoding="utf-8") as fh:
        for item in candidates:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "inkup.question-retrieval-run/v0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(repo),
        "model": MODEL_NAME,
        "query_sha256": hashlib.sha256(Path(query_path).read_bytes()).hexdigest(),
        "query_texts": query_texts,
        "terms": terms,
        "anchors": anchors,
        "lexical_anchor_rule": ANCHOR_RULE if anchors else None,
        "problem": query.get("problem"),
        "top_per_method_per_site": limit,
        "stores": reports,
        "corpus_limits": CORPUS_LIMITS,
        "labels": "none: no relevance labels and no question-function labels are assigned",
        "counts": {
            "total": len(candidates),
            "by_method": {m: sum(c["method"] == m for c in candidates)
                          for m in ("lexical", "embedding")},
            "by_site": {site: sum(c["site"] == site for c in candidates) for site, _ in stores},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")
    return output_dir
