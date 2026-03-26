#!/usr/bin/env python3
"""Build/update arXiv manifest DB from arXiv OAI-PMH (metadataPrefix=arXiv).

Default scope: primary math set via OAI setSpec `math`, latest version only.
This keeps costs low and aligns with rate-limited, resumable harvesting.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BASE_URL_DEFAULT = "https://export.arxiv.org/oai2"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "arxiv-manifest" / "arxiv_manifest.sqlite"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}


@dataclass
class Record:
    arxiv_id: str
    created: str
    updated: str
    oai_datestamp: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    set_specs: list[str]
    doi: str
    license: str
    withdrawn: bool
    versions: list[int]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def req_xml(
    url: str,
    user_agent: str,
    timeout: int,
    retries: int,
    retry_sleep_seconds: float,
) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    attempts = retries + 1
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500:
                raise
            if attempt >= attempts:
                raise
            wait_seconds = retry_sleep_seconds * attempt
            print(
                f"[retry] attempt={attempt}/{attempts} wait={wait_seconds:.1f}s "
                f"url={url} error={exc}"
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"unreachable req_xml failure: {last_error}")


def normalize_text(s: str) -> str:
    return " ".join((s or "").split())


def parse_versions(arxiv_meta: ET.Element) -> list[int]:
    out: list[int] = []
    for v in arxiv_meta.findall("arxiv:version", NS):
        vname = v.attrib.get("version", "")
        if vname.startswith("v") and vname[1:].isdigit():
            out.append(int(vname[1:]))
    if not out:
        out = [1]
    return sorted(set(out))


def parse_record(rec_elem: ET.Element) -> Record | None:
    header = rec_elem.find("oai:header", NS)
    if header is None:
        return None
    if header.attrib.get("status") == "deleted":
        return None

    ident = header.findtext("oai:identifier", "", NS)
    if not ident.startswith("oai:arXiv.org:"):
        return None
    arxiv_id = ident.split(":", 2)[-1].strip()
    oai_datestamp = header.findtext("oai:datestamp", "", NS).strip()
    set_specs = [s.text.strip() for s in header.findall("oai:setSpec", NS) if s.text]

    md = rec_elem.find("oai:metadata", NS)
    if md is None:
        return None
    arxiv_meta = md.find("arxiv:arXiv", NS)
    if arxiv_meta is None:
        return None

    created = arxiv_meta.findtext("arxiv:created", "", NS).strip()
    updated = arxiv_meta.findtext("arxiv:updated", "", NS).strip()
    title = normalize_text(arxiv_meta.findtext("arxiv:title", "", NS))
    abstract = normalize_text(arxiv_meta.findtext("arxiv:abstract", "", NS))

    authors: list[str] = []
    for a in arxiv_meta.findall("arxiv:authors/arxiv:author", NS):
        key = " ".join(filter(None, [
            (a.findtext("arxiv:keyname", "", NS) or "").strip(),
            (a.findtext("arxiv:forenames", "", NS) or "").strip(),
        ])).strip()
        if key:
            authors.append(key)

    categories_raw = (arxiv_meta.findtext("arxiv:categories", "", NS) or "").strip()
    categories = [c for c in categories_raw.split() if c]
    primary = categories[0] if categories else ""

    doi = (arxiv_meta.findtext("arxiv:doi", "", NS) or "").strip()
    lic = (arxiv_meta.findtext("arxiv:license", "", NS) or "").strip()
    withdrawn = "withdrawn" in abstract.lower()

    versions = parse_versions(arxiv_meta)

    return Record(
        arxiv_id=arxiv_id,
        created=created,
        updated=updated,
        oai_datestamp=oai_datestamp,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        primary_category=primary,
        set_specs=set_specs,
        doi=doi,
        license=lic,
        withdrawn=withdrawn,
        versions=versions,
    )


def make_urls(arxiv_id: str, version: int) -> tuple[str, str, str]:
    idv = f"{arxiv_id}v{version}"
    abs_url = f"https://arxiv.org/abs/{idv}"
    eprint_url = f"https://arxiv.org/e-print/{idv}"
    return idv, abs_url, eprint_url


def year_ok(date_str: str, year_from: int | None, year_to: int | None) -> bool:
    if not date_str or len(date_str) < 4 or not date_str[:4].isdigit():
        return True
    y = int(date_str[:4])
    if year_from is not None and y < year_from:
        return False
    if year_to is not None and y > year_to:
        return False
    return True


def upsert_row(conn: sqlite3.Connection, rec: Record, version: int, latest: bool, include: bool) -> None:
    idv, abs_url, eprint_url = make_urls(rec.arxiv_id, version)
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, version, id_with_version, created, updated, oai_datestamp,
            title, abstract, authors_json, categories_json, primary_category,
            set_specs_json, doi, license, is_withdrawn,
            abs_url, eprint_url, latest, include, source, harvested_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, 'oai', datetime('now')
        )
        ON CONFLICT(arxiv_id, version) DO UPDATE SET
            id_with_version=excluded.id_with_version,
            created=excluded.created,
            updated=excluded.updated,
            oai_datestamp=excluded.oai_datestamp,
            title=excluded.title,
            abstract=excluded.abstract,
            authors_json=excluded.authors_json,
            categories_json=excluded.categories_json,
            primary_category=excluded.primary_category,
            set_specs_json=excluded.set_specs_json,
            doi=excluded.doi,
            license=excluded.license,
            is_withdrawn=excluded.is_withdrawn,
            abs_url=excluded.abs_url,
            eprint_url=excluded.eprint_url,
            latest=excluded.latest,
            include=excluded.include,
            source='oai',
            harvested_at=datetime('now')
        """,
        (
            rec.arxiv_id,
            version,
            idv,
            rec.created,
            rec.updated,
            rec.oai_datestamp,
            rec.title,
            rec.abstract,
            json.dumps(rec.authors, ensure_ascii=False),
            json.dumps(rec.categories, ensure_ascii=False),
            rec.primary_category,
            json.dumps(rec.set_specs, ensure_ascii=False),
            rec.doi,
            rec.license,
            1 if rec.withdrawn else 0,
            abs_url,
            eprint_url,
            1 if latest else 0,
            1 if include else 0,
        ),
    )


def set_latest_flags(conn: sqlite3.Connection, arxiv_id: str, latest_version: int) -> None:
    conn.execute("UPDATE papers SET latest = 0 WHERE arxiv_id = ?", (arxiv_id,))
    conn.execute(
        "UPDATE papers SET latest = 1 WHERE arxiv_id = ? AND version = ?",
        (arxiv_id, latest_version),
    )


def build_initial_url(base_url: str, oai_set: str, metadata_prefix: str, from_utc: str | None, until_utc: str | None) -> str:
    q = {
        "verb": "ListRecords",
        "metadataPrefix": metadata_prefix,
        "set": oai_set,
    }
    if from_utc:
        q["from"] = from_utc
    if until_utc:
        q["until"] = until_utc
    return base_url + "?" + urllib.parse.urlencode(q)


def extract_resumption_token(xml_root: ET.Element) -> str:
    tok = xml_root.findtext("oai:ListRecords/oai:resumptionToken", "", NS)
    return (tok or "").strip()


def build_resumption_url(base_url: str, token: str) -> str:
    return base_url + "?" + urllib.parse.urlencode({"verb": "ListRecords", "resumptionToken": token})


def run_harvest(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path} (run init_manifest_db.py first)")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")

    conn.execute(
        """
        INSERT INTO harvest_runs (
            base_url, oai_set, metadata_prefix, from_utc, until_utc,
            latest_only, include_crosslists, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            args.base_url,
            args.oai_set,
            args.metadata_prefix,
            args.from_utc,
            args.until_utc,
            1 if args.latest_only else 0,
            1 if args.include_crosslists else 0,
            args.notes,
        ),
    )
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()

    # Resume support
    token = ""
    if args.resume:
        row = conn.execute("SELECT state_value FROM harvest_state WHERE state_key='oai_resumption_token'").fetchone()
        if row:
            token = row[0]

    url = build_resumption_url(args.base_url, token) if token else build_initial_url(
        args.base_url, args.oai_set, args.metadata_prefix, args.from_utc, args.until_utc
    )

    seen = 0
    written = 0
    skipped = 0
    updated = 0
    page = 0

    try:
        while url:
            page += 1
            if page > 1 and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            xml_bytes = req_xml(
                url,
                args.user_agent,
                args.timeout,
                args.retries,
                args.retry_sleep_seconds,
            )
            root = ET.fromstring(xml_bytes)

            records = root.findall("oai:ListRecords/oai:record", NS)
            for rec_elem in records:
                rec = parse_record(rec_elem)
                if rec is None:
                    continue

                seen += 1

                if args.primary_prefix and rec.primary_category and not rec.primary_category.startswith(args.primary_prefix):
                    skipped += 1
                    continue

                if not year_ok(rec.created, args.year_from, args.year_to):
                    skipped += 1
                    continue

                versions = [max(rec.versions)] if args.latest_only else rec.versions
                latest_v = max(rec.versions)

                for v in versions:
                    include = True
                    if not args.include_crosslists and rec.primary_category and not rec.primary_category.startswith("math."):
                        include = False

                    before = conn.total_changes
                    upsert_row(conn, rec, v, latest=(v == latest_v), include=include)
                    after = conn.total_changes
                    if after > before:
                        # sqlite total_changes doesn't tell insert vs update; track as written.
                        written += 1

                set_latest_flags(conn, rec.arxiv_id, latest_v)

            token = extract_resumption_token(root)
            conn.execute(
                """
                INSERT INTO harvest_state(state_key, state_value, updated_at)
                VALUES('oai_resumption_token', ?, datetime('now'))
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value=excluded.state_value,
                    updated_at=datetime('now')
                """,
                (token,),
            )
            conn.commit()

            print(
                f"[harvest] page={page} records={len(records)} seen={seen} "
                f"written={written} skipped={skipped} token={'yes' if token else 'no'}"
            )

            url = build_resumption_url(args.base_url, token) if token else ""

        # Clear token on successful completion.
        conn.execute(
            """
            INSERT INTO harvest_state(state_key, state_value, updated_at)
            VALUES('oai_resumption_token', '', datetime('now'))
            ON CONFLICT(state_key) DO UPDATE SET
                state_value='', updated_at=datetime('now')
            """
        )

        conn.execute(
            """
            UPDATE harvest_runs
            SET finished_at=datetime('now'),
                records_seen=?, rows_written=?, rows_updated=?, rows_skipped=?
            WHERE run_id=?
            """,
            (seen, written, updated, skipped, run_id),
        )
        conn.commit()
    except Exception as exc:
        conn.execute(
            """
            UPDATE harvest_runs
            SET finished_at=datetime('now'),
                records_seen=?, rows_written=?, rows_updated=?, rows_skipped=?,
                notes=COALESCE(notes, '') || '\nERROR: ' || ?
            WHERE run_id=?
            """,
            (seen, written, updated, skipped, str(exc), run_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()

    print(f"[harvest] done run_id={run_id} seen={seen} written={written} skipped={skipped}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build arXiv manifest DB from OAI-PMH")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--base-url", default=BASE_URL_DEFAULT)
    ap.add_argument("--oai-set", default="math", help="OAI setSpec (default: math)")
    ap.add_argument("--metadata-prefix", default="arXiv", help="OAI metadataPrefix (default: arXiv)")
    ap.add_argument("--from-utc", default=None, help="Optional OAI from=YYYY-MM-DD")
    ap.add_argument("--until-utc", default=None, help="Optional OAI until=YYYY-MM-DD")
    ap.add_argument("--year-from", type=int, default=1992)
    ap.add_argument("--year-to", type=int, default=2026)
    ap.add_argument("--primary-prefix", default="math.", help="Keep records with primary category prefix")
    ap.add_argument("--latest-only", action="store_true", default=True)
    ap.add_argument("--all-versions", action="store_true", help="Override latest-only and keep all versions")
    ap.add_argument("--include-crosslists", action="store_true", default=False)
    ap.add_argument("--sleep-seconds", type=float, default=3.1)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=6, help="Retry transient request failures")
    ap.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=10.0,
        help="Base sleep between request retries; multiplied by attempt number",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true", help="Do not resume from saved token")
    ap.add_argument("--notes", default="")
    ap.add_argument(
        "--user-agent",
        default="futon6-manifest-builder/1.0 (+https://github.com/tothedarktowercame/futon6)",
    )
    args = ap.parse_args()

    if args.all_versions:
        args.latest_only = False
    if args.no_resume:
        args.resume = False

    return run_harvest(args)


if __name__ == "__main__":
    raise SystemExit(main())
