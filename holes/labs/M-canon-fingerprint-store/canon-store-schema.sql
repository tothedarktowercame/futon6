-- Canon fingerprint store schema
-- Mission: M-canon-fingerprint-store (Billey-Tenner instantiation for symbol grounding)
-- Source: futon6/holes/missions/M-canon-fingerprint-store.md §2, §2.1, §8 (resolved decisions)
--
-- §8(a) resolved decisions:
--   - Drop `position`; locator is strategy_anchor (Arxana-style structural locator)
--   - Add scope fields: role (binder role), scope (binding scope, nullable)
--   - Lift per-theorem grouping so {scoped-inputs → output-relation} is recoverable
-- §8(c): SQLite (in-run-queryable; Stage 5 opens read/write per run)
-- Evidence-ledger discipline (fold ft-canon-fingerprint-store-003):
--   paper_id, strategy, strategy_anchor are NOT NULL — a record without evidence is lore

-- ── CanonFingerprint: per-binding record (the MAP step output) ──────────────

CREATE TABLE IF NOT EXISTS canon_fingerprint (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,               -- e.g. '\sigma(T)' — primary inference key
    canon           TEXT    DEFAULT NULL,           -- e.g. 'Spectrum' or NULL (un-canonical) — secondary key
    paper_id        TEXT    NOT NULL,               -- e.g. 'arxiv-2305.01234v2' — EVIDENCE (not lore)
    strategy        TEXT    NOT NULL,               -- e.g. 'let-binding' — EVIDENCE
    strategy_anchor TEXT    NOT NULL,               -- Arxana-style structural locator (replaces §2 `position`)
    confidence      TEXT    NOT NULL DEFAULT 'medium',  -- 'high' | 'medium' | 'low'
    constructor     TEXT    NOT NULL DEFAULT 'single',  -- 'single' | 'comma-list' | 'relation-chain' | ...
    role            TEXT    DEFAULT NULL,           -- §2.1 scope binding: binder role (e.g. 'hypotenuse-of-right-triangle')
    scope           TEXT    DEFAULT NULL,           -- §2.1 scope binding: binding scope
    theorem_group   TEXT    DEFAULT NULL,           -- §2.1 per-theorem grouping: groups scoped-inputs → output-relation
    timestamp       TEXT    NOT NULL                -- ISO-8601 when emitted
);

-- Indexes per §8: symbol (primary inference key) and canon (secondary)
CREATE INDEX IF NOT EXISTS idx_fingerprint_symbol ON canon_fingerprint(symbol);
CREATE INDEX IF NOT EXISTS idx_fingerprint_canon  ON canon_fingerprint(canon);
CREATE INDEX IF NOT EXISTS idx_fingerprint_pair   ON canon_fingerprint(symbol, canon);
-- Frequency-ordered REDUCE support (§3.1 most-cited-first)
CREATE INDEX IF NOT EXISTS idx_fingerprint_paper   ON canon_fingerprint(paper_id);

-- ── CanonAggregate: per (symbol, canon) aggregate (the REDUCE step output) ───
-- Mirrors the CanonAggregate dataclass from §2.
-- source_paper_ids and strategy_breakdown stored as JSON (SQLite JSON1 or TEXT).

CREATE TABLE IF NOT EXISTS canon_aggregate (
    symbol              TEXT    NOT NULL,
    canon               TEXT    DEFAULT NULL,           -- NULL = un-canonical (matches canon_fingerprint)
    n_occurrences       INTEGER NOT NULL,
    source_paper_ids    TEXT    NOT NULL,           -- JSON array of paper_ids
    strategy_breakdown  TEXT    NOT NULL,           -- JSON object {strategy: count}
    first_seen          TEXT    NOT NULL,           -- ISO timestamp
    last_seen           TEXT    NOT NULL,           -- ISO timestamp
    computed_at         TEXT    NOT NULL,           -- ISO timestamp of this REDUCE run
    aggregate_hash      TEXT    NOT NULL,           -- sha256 of the deterministic serialisation (jiji vigilance)
    PRIMARY KEY (symbol, canon)
);
