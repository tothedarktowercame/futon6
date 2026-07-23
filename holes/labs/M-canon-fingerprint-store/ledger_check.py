#!/usr/bin/env python3
"""
ledger_check.py — B3-F2 evidence-ledger integrity validator for the canon fingerprint store.
Mission: M-canon-fingerprint-store.
Deposit grounding: ft-canon-fingerprint-store-003 (library-coherence/library-evidence-ledger).

The evidence-ledger pattern requires every pattern entry to cite concrete, verifiable evidence.
SQL NOT NULL catches missing fields (NULL); this validator ALSO catches blank/whitespace-only
fields — the case the schema constraint cannot reach. A record with paper_id = '' or paper_id = '   '
passes NOT NULL but is still lore: the citation is present in form but absent in substance.

stdlib sqlite3 only; file-relative paths; runs from any cwd, exit 0.

Acceptance:
  1. Real fixtures green (all records in B3-F1's fixture store pass the ledger check).
  2. Two malformed records REJECTED with named errors:
     - one missing-field (NULL — caught by SQL, but the ledger names it too)
     - one blank-field (empty/whitespace string — NOT caught by SQL NOT NULL)
  3. Any cwd; exit 0.
"""

import os
import sqlite3
import sys
import tempfile
import hashlib

# ── file-relative paths ─────────────────────────────────────────────────────
LAB_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(LAB_DIR, "canon-store-schema.sql")
SMOKE_PATH = os.path.join(LAB_DIR, "smoke_canon_store.py")

# Evidence fields that must be non-null AND non-blank (the ledger's citation requirement)
EVIDENCE_FIELDS = ["paper_id", "strategy", "strategy_anchor"]


def load_schema():
    with open(SCHEMA_PATH, "r") as f:
        return f.read()


def is_blank(val):
    """True if val is None or whitespace-only string."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def check_record(row):
    """
    Validate a single canon_fingerprint row against the evidence-ledger discipline.
    Returns list of error strings (empty = valid).
    """
    errors = []
    # row is a sqlite3.Row; convert to dict for field access
    d = {key: row[key] for key in row.keys()}

    for field in EVIDENCE_FIELDS:
        val = d[field]
        if val is None:
            errors.append(f"MISSING: field '{field}' is NULL — record cites no evidence for this field")
        elif isinstance(val, str) and val.strip() == "":
            errors.append(f"BLANK: field '{field}' is whitespace-only — citation present in form but absent in substance (SQL NOT NULL cannot catch this)")
    return errors


def run_ledger_check(conn, label=""):
    """
    Check all canon_fingerprint rows in the store.
    Returns (n_records, n_valid, errors_list).
    """
    rows = conn.execute(
        "SELECT * FROM canon_fingerprint ORDER BY id"
    ).fetchall()

    all_errors = []
    n_valid = 0

    for row in rows:
        errs = check_record(row)
        if errs:
            rid = row["id"] if "id" in row.keys() else "?"
            for e in errs:
                all_errors.append(f"  record id={rid} symbol={row['symbol']}: {e}")
        else:
            n_valid += 1

    return len(rows), n_valid, all_errors


def create_fixture_store(db_path):
    """Create a store with B3-F1's real fixtures (reuses smoke_canon_store logic)."""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(load_schema())
    conn.commit()

    # Insert the same 6 fixtures from smoke_canon_store.py
    fixtures = [
        ("\\sigma(T)", "Spectrum", "arxiv-2305.01234v2", "let-binding", "lemma-3-defines-sigma-T",
         "high", "single", "operator-on-Banach-space", "functional-analysis", "spectral-theorem", "2026-07-10T10:00:00Z"),
        ("\\sigma(T)", "Spectrum", "arxiv-2401.00567v1", "let-binding", "section-2-notation",
         "high", "single", "operator-on-Hilbert-space", "functional-analysis", "spectral-theorem", "2026-07-10T10:05:00Z"),
        ("\\sigma(T)", "Spectrum", "pm:Spectrum", "relation-chain", "proofwiki-spectrum-definition",
         "medium", "relation-chain", "set-of-eigenvalues", "functional-analysis", "spectral-theorem", "2026-07-10T10:10:00Z"),
        ("\\alpha", "Continuity", "arxiv-2305.01234v2", "comma-list", "section-1-preliminaries",
         "low", "comma-list", "index-parameter", None, None, "2026-07-10T10:15:00Z"),
        ("\\partial f", "Derivative", "arxiv-2401.00567v1", "let-binding", "definition-4-1",
         "high", "single", "frechet-derivative", "optimization", "first-order-optimality", "2026-07-10T10:20:00Z"),
        ("\\sigma(T)", None, "arxiv-2305.01234v2", "single", "remark-after-thm-2",
         "low", "single", None, None, None, "2026-07-10T10:25:00Z"),
    ]
    for fx in fixtures:
        conn.execute(
            """INSERT INTO canon_fingerprint
               (symbol, canon, paper_id, strategy, strategy_anchor,
                confidence, constructor, role, scope, theorem_group, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            fx,
        )
    conn.commit()
    return conn


def create_relaxed_store(db_path):
    """
    Create a store with the schema but WITHOUT NOT NULL on evidence fields.
    This lets us test that the LEDGER CHECK itself catches bad records,
    independent of SQL's constraint — the ledger is the application-layer
    integrity that catches what the schema cannot (or in case the schema
    is ever relaxed/migrated).
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Create the tables without NOT NULL on evidence fields
    conn.executescript("""
        CREATE TABLE canon_fingerprint (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            canon           TEXT    DEFAULT NULL,
            paper_id        TEXT,
            strategy        TEXT,
            strategy_anchor TEXT,
            confidence      TEXT    NOT NULL DEFAULT 'medium',
            constructor     TEXT    NOT NULL DEFAULT 'single',
            role            TEXT    DEFAULT NULL,
            scope           TEXT    DEFAULT NULL,
            theorem_group   TEXT    DEFAULT NULL,
            timestamp       TEXT    NOT NULL
        );
        CREATE TABLE canon_aggregate (
            symbol              TEXT    NOT NULL,
            canon               TEXT    DEFAULT NULL,
            n_occurrences       INTEGER NOT NULL,
            source_paper_ids    TEXT    NOT NULL,
            strategy_breakdown  TEXT    NOT NULL,
            first_seen          TEXT    NOT NULL,
            last_seen           TEXT    NOT NULL,
            computed_at         TEXT    NOT NULL,
            aggregate_hash      TEXT    NOT NULL
        );
    """)
    conn.commit()
    return conn


def main():
    print("=== B3-F2 ledger_check.py ===")
    print(f"Schema: {SCHEMA_PATH}")
    print(f"Evidence fields checked: {EVIDENCE_FIELDS}")
    print()

    tmpdir = tempfile.mkdtemp(prefix="ledger_check_")

    # ── 1. Real fixtures: all must pass ─────────────────────────────────────
    db_path = os.path.join(tmpdir, "fixtures.db")
    conn = create_fixture_store(db_path)

    n_recs, n_valid, errors = run_ledger_check(conn, "real fixtures")
    print(f"REAL FIXTURES: {n_recs} records, {n_valid} valid, {len(errors)} errors")
    if errors:
        for e in errors:
            print(f"  REJECT: {e}")
        print("FAIL: real fixtures have ledger errors")
        conn.close()
        sys.exit(1)
    else:
        print("  All real fixtures GREEN — every record cites verifiable evidence")
    print()

    # ── 2. Malformed record 1: missing field (NULL paper_id) ────────────────
    # We use a relaxed store (no NOT NULL on evidence fields) to prove the
    # LEDGER CHECK itself catches this — the ledger is the application-layer
    # integrity, independent of the SQL constraint.
    relaxed_path = os.path.join(tmpdir, "relaxed.db")
    rconn = create_relaxed_store(relaxed_path)

    # Copy the 6 real fixtures into the relaxed store
    fixtures = [
        ("\\sigma(T)", "Spectrum", "arxiv-2305.01234v2", "let-binding", "lemma-3-defines-sigma-T",
         "high", "single", "operator-on-Banach-space", "functional-analysis", "spectral-theorem", "2026-07-10T10:00:00Z"),
        ("\\sigma(T)", "Spectrum", "arxiv-2401.00567v1", "let-binding", "section-2-notation",
         "high", "single", "operator-on-Hilbert-space", "functional-analysis", "spectral-theorem", "2026-07-10T10:05:00Z"),
        ("\\sigma(T)", "Spectrum", "pm:Spectrum", "relation-chain", "proofwiki-spectrum-definition",
         "medium", "relation-chain", "set-of-eigenvalues", "functional-analysis", "spectral-theorem", "2026-07-10T10:10:00Z"),
        ("\\alpha", "Continuity", "arxiv-2305.01234v2", "comma-list", "section-1-preliminaries",
         "low", "comma-list", "index-parameter", None, None, "2026-07-10T10:15:00Z"),
        ("\\partial f", "Derivative", "arxiv-2401.00567v1", "let-binding", "definition-4-1",
         "high", "single", "frechet-derivative", "optimization", "first-order-optimality", "2026-07-10T10:20:00Z"),
        ("\\sigma(T)", None, "arxiv-2305.01234v2", "single", "remark-after-thm-2",
         "low", "single", None, None, None, "2026-07-10T10:25:00Z"),
    ]
    for fx in fixtures:
        rconn.execute(
            """INSERT INTO canon_fingerprint
               (symbol, canon, paper_id, strategy, strategy_anchor,
                confidence, constructor, role, scope, theorem_group, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            fx,
        )

    # Insert malformed record 1: NULL paper_id
    rconn.execute(
        """INSERT INTO canon_fingerprint
           (symbol, canon, paper_id, strategy, strategy_anchor, timestamp)
           VALUES ('\\bad-null', 'Test', NULL, 'let-binding', 'some-anchor', '2026-07-10T00:00:00Z')""",
    )
    rconn.commit()

    _, _, errors_null = run_ledger_check(rconn, "null-field test")
    null_rejects = [e for e in errors_null if "MISSING" in e and "paper_id" in e]
    print(f"MALFORMED 1 (NULL paper_id):")
    if null_rejects:
        for e in null_rejects:
            print(f"  REJECT: {e}")
        print("  PASS — missing-field record rejected with named error")
    else:
        print("  FAIL — missing-field record was not caught")
        rconn.close()
        sys.exit(1)
    print()

    # ── 3. Malformed record 2: blank field (whitespace-only strategy) ───────
    # This is NOT caught by SQL NOT NULL — the value is '   ' (not NULL).
    # This is the case the ledger catches that the schema cannot.
    rconn.execute(
        """INSERT INTO canon_fingerprint
           (symbol, canon, paper_id, strategy, strategy_anchor, timestamp)
           VALUES ('\\bad-blank', 'Test', 'arxiv-9999', '   ', 'some-anchor', '2026-07-10T00:00:00Z')""",
    )
    rconn.commit()

    _, _, errors_blank = run_ledger_check(rconn, "blank-field test")
    blank_rejects = [e for e in errors_blank if "BLANK" in e and "strategy" in e]
    print(f"MALFORMED 2 (blank strategy = whitespace):")
    if blank_rejects:
        for e in blank_rejects:
            print(f"  REJECT: {e}")
        print("  PASS — blank-field record rejected (SQL NOT NULL cannot catch this)")
    else:
        print("  FAIL — blank-field record was not caught")
        rconn.close()
        sys.exit(1)
    print()

    # ── Summary ─────────────────────────────────────────────────────────────
    conn.close()
    rconn.close()
    print("Ledger check complete. All acceptance criteria met. Exit 0.")
    sys.exit(0)


if __name__ == "__main__":
    main()
