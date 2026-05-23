"""Canon fingerprint store — Billey-Tenner instantiation for symbol grounding.

Implements F1+F2 of M-canon-fingerprint-store.md:

  F1 (schema + writer): per-binding `CanonFingerprint` records,
      written append-only as JSONL per batch.
  F2 (reducer + query): `aggregate_canon_store()` incrementally
      state-merges batch files into a persistent `CanonAggregate`
      dict; `canon_distribution()` queries by symbol.

The store is the persistent layer that survives across batches —
the OEIS-shaped knowledge that the per-binding Bayesian posterior
(slice F3, already shipped) will consume as a prior in slice F5.

File layout convention (caller's choice; defaults shown):
    data/canon-store/fingerprints/batch-XXX.jsonl   [per-binding records]
    data/canon-store/aggregate.json                 [reduced state]

JSONL was chosen over SQLite for v1 because it matches existing
infrastructure (`learned-newcommand-vocab.json`) and stays cheap
under append-heavy MAP load. SQLite is the natural promotion path
if Stage 5 needs to query during the run rather than at startup.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# Source-paper IDs are stored as a bounded sample so a popular
# (symbol, canon) like (G, Group) doesn't blow up the aggregate.
# The cap is per-aggregate; the n_occurrences counter remains exact.
SAMPLE_PAPER_LIMIT = 50


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class CanonFingerprint:
    """A single binding's persistent fingerprint.

    Frozen so it can serve as a dict key for in-memory dedup,
    though the actual store dedups by (symbol, canon) at the
    aggregate level.
    """
    symbol: str
    canon: str | None
    paper_id: str
    strategy: str
    confidence: str = "medium"
    constructor: str = "single"
    timestamp: str = ""  # caller can leave blank; writer fills it in

    def to_jsonable(self) -> dict:
        d = asdict(self)
        if not d["timestamp"]:
            d["timestamp"] = _now_iso()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CanonFingerprint":
        return cls(
            symbol=d["symbol"],
            canon=d.get("canon"),
            paper_id=d["paper_id"],
            strategy=d.get("strategy", ""),
            confidence=d.get("confidence", "medium"),
            constructor=d.get("constructor", "single"),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class CanonAggregate:
    """Aggregated evidence for one (symbol, canon) pair across batches."""
    symbol: str
    canon: str
    n_occurrences: int = 0
    source_paper_ids: list[str] = field(default_factory=list)
    strategy_breakdown: dict[str, int] = field(default_factory=dict)
    first_seen: str = ""
    last_seen: str = ""

    def merge_fingerprint(self, fp: CanonFingerprint) -> None:
        """In-place update of this aggregate by one new fingerprint."""
        self.n_occurrences += 1
        if fp.paper_id and len(self.source_paper_ids) < SAMPLE_PAPER_LIMIT:
            if fp.paper_id not in self.source_paper_ids:
                self.source_paper_ids.append(fp.paper_id)
        self.strategy_breakdown[fp.strategy] = (
            self.strategy_breakdown.get(fp.strategy, 0) + 1
        )
        ts = fp.timestamp or _now_iso()
        if not self.first_seen or ts < self.first_seen:
            self.first_seen = ts
        if not self.last_seen or ts > self.last_seen:
            self.last_seen = ts

    def to_jsonable(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CanonAggregate":
        return cls(
            symbol=d["symbol"],
            canon=d["canon"],
            n_occurrences=d.get("n_occurrences", 0),
            source_paper_ids=list(d.get("source_paper_ids") or []),
            strategy_breakdown=dict(d.get("strategy_breakdown") or {}),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
        )


# ============================================================
# F1: writer
# ============================================================

def write_batch_fingerprints(
    records: Iterable[CanonFingerprint],
    out_path: Path,
) -> int:
    """Append-only JSONL writer. Returns count written.

    Each line is one JSON-encoded CanonFingerprint. Caller picks the
    file path (typically `data/canon-store/fingerprints/batch-NNN.jsonl`);
    this function only writes records, doesn't enforce filename schema.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for fp in records:
            f.write(json.dumps(fp.to_jsonable(), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_batch_fingerprints(jsonl_path: Path) -> Iterable[CanonFingerprint]:
    """Stream fingerprints from a single batch JSONL file."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield CanonFingerprint.from_dict(json.loads(line))
            except json.JSONDecodeError:
                continue


# ============================================================
# F2: reducer + query
# ============================================================

def aggregate_canon_store(
    batch_jsonl_paths: Iterable[Path],
    prior_aggregate: dict[tuple[str, str], CanonAggregate] | None = None,
) -> dict[tuple[str, str], CanonAggregate]:
    """State-merge `batch_jsonl_paths` into `prior_aggregate` (or a fresh
    empty dict). Idempotent: re-running with the same inputs produces
    the same output, modulo the `last_seen` timestamp resolution.

    Returned dict is keyed by (symbol, canon). The aggregate ignores
    fingerprints whose canon is None (those don't denote anything to
    aggregate). Caller serializes via `save_aggregate()`.
    """
    aggregate: dict[tuple[str, str], CanonAggregate] = dict(prior_aggregate or {})
    for path in batch_jsonl_paths:
        for fp in iter_batch_fingerprints(path):
            if fp.canon is None:
                continue
            key = (fp.symbol, fp.canon)
            existing = aggregate.get(key)
            if existing is None:
                existing = CanonAggregate(symbol=fp.symbol, canon=fp.canon)
                aggregate[key] = existing
            existing.merge_fingerprint(fp)
    return aggregate


def save_aggregate(
    aggregate: dict[tuple[str, str], CanonAggregate],
    out_path: Path,
) -> None:
    """Serialize aggregate to JSON. The (symbol, canon) tuple keys
    become a list of records so the JSON stays clean for human reading."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v1",
        "n_aggregates": len(aggregate),
        "records": [agg.to_jsonable() for agg in aggregate.values()],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def load_aggregate(
    path: Path,
) -> dict[tuple[str, str], CanonAggregate]:
    """Inverse of `save_aggregate`. Missing file → empty dict."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], CanonAggregate] = {}
    for rec in data.get("records", []):
        agg = CanonAggregate.from_dict(rec)
        out[(agg.symbol, agg.canon)] = agg
    return out


def canon_distribution(
    aggregate: dict[tuple[str, str], CanonAggregate],
    symbol: str,
) -> dict[str, CanonAggregate]:
    """Query: for a symbol, return all canons it's been bound to and
    the aggregate evidence for each. Empty dict if the symbol is
    unseen in the store.
    """
    return {
        canon: agg
        for (sym, canon), agg in aggregate.items()
        if sym == symbol
    }


def canon_prior(
    aggregate: dict[tuple[str, str], CanonAggregate],
    symbol: str,
    smoothing: float = 0.1,
) -> dict[str, float]:
    """Convert the canon distribution for `symbol` into a normalised
    prior over canons. The prior is `(n_occurrences + smoothing) /
    (total + len * smoothing)` — additive smoothing so unseen canons
    aren't strictly zero in the downstream posterior calculation.

    Used by slice F5 when the per-binding posterior asks the store
    "what does X usually mean across the literature we've seen?"
    Returns an empty dict if the symbol is unseen.
    """
    dist = canon_distribution(aggregate, symbol)
    if not dist:
        return {}
    total = sum(a.n_occurrences for a in dist.values())
    n_options = len(dist)
    denom = total + smoothing * n_options
    return {
        canon: (agg.n_occurrences + smoothing) / denom
        for canon, agg in dist.items()
    }
