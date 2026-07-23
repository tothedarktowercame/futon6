#!/usr/bin/env python3
"""
smoke_canon_store.py — B3-F1 smoke test for the canon fingerprint store.
Mission: M-canon-fingerprint-store (Billey-Tenner instantiation).
Deposit grounding: ft-canon-fingerprint-store-002 (b2 jiji vigilance + b3 snapshot-witness).

stdlib sqlite3 only; file-relative paths; runs from any cwd, exit 0.

Acceptance:
  1. Script runs from any cwd, exit 0.
  2. Re-run aggregate hash identical (print both).
  3. NULL-evidence insert REJECTED (show the error).
"""

import hashlib
import json
import os
import sqlite3
import sys
import tempfile

# ── file-relative path to schema ────────────────────────────────────────────
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canon-store-schema.sql")


def load_schema():
    """Read the schema SQL from the file-relative path."""
    with open(SCHEMA_PATH, "r") as f:
        return f.read()


def create_store(db_path):
    """Create a fresh SQLite store from the schema."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(load_schema())
    conn.commit()
    return conn


def insert_fingerprint(conn, symbol, canon, paper_id, strategy, strategy_anchor,
                       confidence="medium", constructor="single", role=None,
                       scope=None, theorem_group=None, timestamp="2026-07-10T12:00:00Z"):
    """Insert a single CanonFingerprint record."""
    conn.execute(
        """INSERT INTO canon_fingerprint
           (symbol, canon, paper_id, strategy, strategy_anchor,
            confidence, constructor, role, scope, theorem_group, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, canon, paper_id, strategy, strategy_anchor,
         confidence, constructor, role, scope, theorem_group, timestamp),
    )
    conn.commit()


# ── Fixture fingerprints (≥5, covering the mission's resolved schema) ───────
# These exercise: scope bindings (§2.1), strategy_anchor (§8a replaces position),
# multiple symbols/canons, multiple papers/strategies.

FIXTURES = [
    # σ(T) → Spectrum, two papers, let-binding strategy
    dict(symbol="\\sigma(T)", canon="Spectrum", paper_id="arxiv-2305.01234v2",
         strategy="let-binding", strategy_anchor="lemma-3-defines-sigma-T",
         confidence="high", constructor="single",
         role="operator-on-Banach-space", scope="functional-analysis",
         theorem_group="spectral-theorem", timestamp="2026-07-10T10:00:00Z"),
    dict(symbol="\\sigma(T)", canon="Spectrum", paper_id="arxiv-2401.00567v1",
         strategy="let-binding", strategy_anchor="section-2-notation",
         confidence="high", constructor="single",
         role="operator-on-Hilbert-space", scope="functional-analysis",
         theorem_group="spectral-theorem", timestamp="2026-07-10T10:05:00Z"),
    # σ(T) → Spectrum, different strategy (relation-chain)
    dict(symbol="\\sigma(T)", canon="Spectrum", paper_id="pm:Spectrum",
         strategy="relation-chain", strategy_anchor="proofwiki-spectrum-definition",
         confidence="medium", constructor="relation-chain",
         role="set-of-eigenvalues", scope="functional-analysis",
         theorem_group="spectral-theorem", timestamp="2026-07-10T10:10:00Z"),
    # α → Continuity, single paper
    dict(symbol="\\alpha", canon="Continuity", paper_id="arxiv-2305.01234v2",
         strategy="comma-list", strategy_anchor="section-1-preliminaries",
         confidence="low", constructor="comma-list",
         role="index-parameter", scope=None,
         theorem_group=None, timestamp="2026-07-10T10:15:00Z"),
    # ∂f → Derivative, different symbol/canon
    dict(symbol="\\partial f", canon="Derivative", paper_id="arxiv-2401.00567v1",
         strategy="let-binding", strategy_anchor="definition-4-1",
         confidence="high", constructor="single",
         role="frechet-derivative", scope="optimization",
         theorem_group="first-order-optimality", timestamp="2026-07-10T10:20:00Z"),
    # σ(T) → Null (un-canonical), tests NULL canon handling
    dict(symbol="\\sigma(T)", canon=None, paper_id="arxiv-2305.01234v2",
         strategy="single", strategy_anchor="remark-after-thm-2",
         confidence="low", constructor="single",
         role=None, scope=None,
         theorem_group=None, timestamp="2026-07-10T10:25:00Z"),
]


def run_reduce(conn, computed_at):
    """
    The REDUCE step: aggregate canon_fingerprint rows into canon_aggregate.

    This is a FULL re-aggregation (not incremental) to test jiji idempotence:
    re-running on the same fingerprints must produce a byte-identical aggregate.

    Returns the aggregate_hash (sha256 of the deterministic serialisation).
    """
    # Clear existing aggregates
    conn.execute("DELETE FROM canon_aggregate")

    # Query all (symbol, canon) pairs — canon can be NULL (un-canonical)
    rows = conn.execute("""
        SELECT symbol, canon, paper_id, strategy, timestamp
        FROM canon_fingerprint
        ORDER BY symbol, canon, paper_id, strategy, timestamp
    """).fetchall()

    # Group by (symbol, canon)
    groups = {}
    for symbol, canon, paper_id, strategy, timestamp in rows:
        key = (symbol, canon if canon is not None else "__UN_CANONICAL__")
        if key not in groups:
            groups[key] = {"papers": set(), "strategies": {}, "timestamps": [], "n": 0}
        g = groups[key]
        g["papers"].add(paper_id)
        g["strategies"][strategy] = g["strategies"].get(strategy, 0) + 1
        g["timestamps"].append(timestamp)
        g["n"] += 1

    # Build deterministic serialisation for hashing
    # (sorted keys, sorted sets, sorted strategies → stable byte output)
    serial_parts = []
    for (symbol, canon_key), g in sorted(groups.items()):
        canon_display = canon_key if canon_key != "__UN_CANONICAL__" else None
        paper_ids_sorted = sorted(g["papers"])
        strategy_sorted = sorted(g["strategies"].items())
        first_seen = min(g["timestamps"])
        last_seen = max(g["timestamps"])

        # Insert into aggregate table
        conn.execute(
            """INSERT OR REPLACE INTO canon_aggregate
               (symbol, canon, n_occurrences, source_paper_ids, strategy_breakdown,
                first_seen, last_seen, computed_at, aggregate_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, canon_display, g["n"],
             json.dumps(paper_ids_sorted),
             json.dumps(strategy_sorted),
             first_seen, last_seen, computed_at, ""),  # hash filled after
        )

        # Deterministic serialisation part for this aggregate
        part = json.dumps({
            "symbol": symbol,
            "canon": canon_display,
            "n_occurrences": g["n"],
            "source_paper_ids": paper_ids_sorted,
            "strategy_breakdown": strategy_sorted,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }, sort_keys=True, separators=(",", ":"))
        serial_parts.append(part)

    # Compute aggregate hash over ALL aggregate rows (deterministic order)
    full_serial = "\n".join(serial_parts)
    aggregate_hash = hashlib.sha256(full_serial.encode("utf-8")).hexdigest()

    # Update all rows with the hash
    conn.execute("UPDATE canon_aggregate SET aggregate_hash = ?", (aggregate_hash,))
    conn.commit()

    return aggregate_hash


def test_null_evidence_rejected(conn):
    """
    jiji vigilance + evidence-ledger discipline:
    A fingerprint record without evidence (paper_id, strategy, or strategy_anchor)
    MUST be rejected by the NOT NULL constraint.

    Returns the error message string.
    """
    try:
        conn.execute(
            """INSERT INTO canon_fingerprint
               (symbol, canon, paper_id, strategy, strategy_anchor, timestamp)
               VALUES ('\\test', 'Test', NULL, 'let-binding', 'some-anchor', '2026-07-10T00:00:00Z')""",
        )
        conn.commit()
        # If we get here, the constraint failed to fire
        print("FAIL: NULL paper_id was accepted — NOT NULL constraint missing")
        sys.exit(1)
    except sqlite3.IntegrityError as e:
        return str(e)


def main():
    print("=== B3-F1 smoke_canon_store.py ===")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Python: {sys.executable}")
    print()

    # Use a temp dir so the script runs from any cwd
    tmpdir = tempfile.mkdtemp(prefix="canon_store_smoke_")
    db_path = os.path.join(tmpdir, "canon_store.db")

    # ── 1. Create store and insert fixtures ─────────────────────────────────
    conn = create_store(db_path)
    print(f"Store created: {db_path}")

    for i, fx in enumerate(FIXTURES):
        insert_fingerprint(conn, **fx)
    print(f"Fixtures inserted: {len(FIXTURES)} fingerprint records")
    print()

    # ── 2. First REDUCE run ─────────────────────────────────────────────────
    hash1 = run_reduce(conn, "2026-07-10T12:00:00Z")
    print(f"REDUCE run 1 aggregate_hash: {hash1}")

    # Verify aggregate contents
    agg_rows = conn.execute(
        "SELECT symbol, canon, n_occurrences FROM canon_aggregate ORDER BY symbol, canon"
    ).fetchall()
    print(f"Aggregates: {len(agg_rows)} (symbol, canon) pairs")
    for row in agg_rows:
        sym, can, n = row
        print(f"  {sym} → {can}: {n} occurrences")
    print()

    # ── 3. Second REDUCE run (jiji idempotence check) ───────────────────────
    # Re-run on the SAME fingerprints — must produce byte-identical aggregate.
    # The second run clears and re-derives; the hash must match.
    hash2 = run_reduce(conn, "2026-07-10T12:00:01Z")  # different computed_at, same data
    print(f"REDUCE run 2 aggregate_hash: {hash2}")
    print()

    if hash1 == hash2:
        print("JlJI VIGILANCE CHECK: PASS — re-run aggregate is byte-identical")
    else:
        print("JlJI VIGILANCE CHECK: FAIL — aggregates differ!")
        print(f"  hash1: {hash1}")
        print(f"  hash2: {hash2}")
        conn.close()
        sys.exit(1)
    print()

    # ── 4. NULL-evidence rejection (evidence-ledger discipline) ─────────────
    error_msg = test_null_evidence_rejected(conn)
    print(f"NULL-evidence rejection: PASS")
    print(f"  Error: {error_msg}")
    print()

    # ── Done ────────────────────────────────────────────────────────────────
    conn.close()
    print("All checks passed. Exit 0.")
    sys.exit(0)


if __name__ == "__main__":
    main()
