#!/usr/bin/env python3
"""pattern_exposure_daily.py — per-day pattern EXPOSURE counts.

How often futon3a retrieval SURFACED each library pattern to an A->B turn,
per day, keyed like data/pattern-roads.json (pattern_roads.pattern_key).
This is the illumination channel for the Futon City roads. It is EXPOSURE,
not attestation: an offer by the retriever, not a use, investigation, or
outcome (archipelago brief §5/§6/§9).

Reading of the evidence follows futon0.report.pattern-density:
  * GET /api/alpha/evidence entries tagged context-retrieval, one entry per
    turn; the body's `results` vector carries the surfaced pattern ids;
  * day bucket = (subs evidence/at 0 10);
  * malformed bodies are non-events, never fabricated empty retrievals.

Pagination discipline (refresh_pattern_attestation.sh): the serving JVM
spends ~5 ms/entry on broad pages, so query ONE DAY per request,
sequentially, never in parallel. A day returning exactly `limit` rows may
be truncated: split it into hours. A failed day goes in "gaps", never zero.

Incremental: an existing output is extended, re-querying only days >= its
last_day (the last day may have been partial).
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import futon6_config as config
import pattern_roads  # pattern_key identity, shared with the road table

ROOT = config.ROOT
OUT = ROOT / "data" / "pattern-exposure-daily.json"

EVIDENCE_BASE = (
    os.environ.get("FUTON3C_EVIDENCE_BASE")
    or os.environ.get("FUTON3C_SERVER")
    or f"http://localhost:{os.environ.get('FUTON3C_PORT', '7070')}"
).rstrip("/")
# ~5 ms/entry server-side on broad pages; a day page is far smaller, but a
# cold JVM read gets the same generous budget the snapshot publishers use.
TIMEOUT_S = int(os.environ.get("EXPOSURE_TIMEOUT_S", "240"))
LIMIT = int(os.environ.get("EXPOSURE_LIMIT", "2000"))
EARLIEST = date(2026, 1, 1)
TZ_LABEL = "Europe/London"  # pattern-density's bucketing convention


# ------------------------------------------------------- evidence parsing

def _plain(x):
    """Recursively convert parsed EDN/JSON into str-keyed python data.
    edn_format returns ImmutableDict/ImmutableList and Keyword keys."""
    from collections.abc import Mapping
    if isinstance(x, Mapping):
        return {str(k).lstrip(":"): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)) or (
            not isinstance(x, (str, bytes)) and hasattr(x, "__iter__")
            and type(x).__module__.startswith("edn_format")):
        return [_plain(v) for v in x]
    return str(x) if type(x).__name__ in ("Keyword", "Symbol") else x


def evidence_body(entry):
    """The API has returned decoded maps, JSON strings and EDN strings over
    time. Accept all three; malformed bodies are non-events (None)."""
    body = entry.get("evidence/body")
    if isinstance(body, dict):
        return _plain(body)
    if not isinstance(body, str):
        return None
    try:
        return _plain(json.loads(body))
    except ValueError:
        pass
    try:
        import edn_format
        return _plain(edn_format.loads(body))
    except Exception:
        return None


def retrieval_result_ids(entry):
    """Pattern ids the retriever surfaced in this turn's results vector."""
    body = evidence_body(entry)
    if not isinstance(body, dict):
        return []
    results = body.get("results") or []
    ids = []
    for r in results:
        if isinstance(r, dict):
            pid = r.get("id")
            if pid:
                ids.append(str(pid))
    return ids


def entry_day(entry):
    at = entry.get("evidence/at") or ""
    return at[:10] if len(at) >= 10 else None


# ------------------------------------------------------- id -> road key

def flexiarg_key_sets(code_root):
    """(keys, byname) from futon*/library/**/*.flexiarg: the road-table key
    space. keys: full 'namespace/name'; byname: bare name -> key when the
    name is unique across collections."""
    keys = set()
    names = defaultdict(set)
    for p in code_root.glob("futon*/library/**/*.flexiarg"):
        s = str(p)
        if "/library/" not in s:
            continue
        key = s.split("/library/", 1)[1][: -len(".flexiarg")]
        keys.add(key)
        names[key.rsplit("/", 1)[-1]].add(key)
    byname = {n: next(iter(ks)) for n, ks in names.items() if len(ks) == 1}
    return keys, byname


def map_result_id(rid, keys, byname):
    """A result id ('collection/name' or a bare name) -> the roads' key,
    or None (goes in the unmatched tally)."""
    if rid in keys:
        return rid
    if "/" not in rid:
        return byname.get(rid)
    # namespaced id that is not a flexiarg path: try its bare name
    return byname.get(rid.rsplit("/", 1)[-1])


# ------------------------------------------------------------- aggregation

def empty_state():
    return {"turns": Counter(), "exposure": defaultdict(Counter),
            "unmatched": Counter(), "gaps": set(), "seen": set()}


def absorb_entries(state, entries, keys, byname):
    """Fold one page of entries into the running counters (deduped by
    evidence id)."""
    for e in entries:
        eid = e.get("evidence/id")
        if eid is not None:
            if eid in state["seen"]:
                continue
            state["seen"].add(eid)
        day = entry_day(e)
        if not day:
            continue
        state["turns"][day] += 1
        for rid in retrieval_result_ids(e):
            key = map_result_id(rid, keys, byname)
            if key:
                state["exposure"][key][day] += 1
            else:
                state["unmatched"][rid] += 1


# ------------------------------------------------------------- fetching

def fetch_window(since, before, limit=LIMIT, base=EVIDENCE_BASE,
                 timeout=TIMEOUT_S):
    """One GET; returns the entries list. Raises on transport/HTTP error."""
    q = urllib.parse.urlencode({
        "tag": "context-retrieval", "since": since, "before": before,
        "limit": limit})
    req = urllib.request.Request(f"{base}/api/alpha/evidence?{q}",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        page = json.load(r)
    return page.get("entries", [])


def fetch_day(day, fetch=fetch_window, limit=LIMIT):
    """All context-retrieval entries for one day, splitting into hours when
    a day page hits the limit (never accept a truncated page)."""
    since = f"{day.isoformat()}T00:00:00Z"
    before = f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z"
    entries = fetch(since, before, limit)
    if len(entries) < limit:
        return entries
    # Possibly truncated: re-fetch hour by hour.
    out = []
    for h in range(24):
        h0 = datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc)
        h1 = h0 + timedelta(hours=1)
        chunk = fetch(h0.isoformat().replace("+00:00", "Z"),
                      h1.isoformat().replace("+00:00", "Z"), limit)
        if len(chunk) >= limit:
            raise RuntimeError(
                f"{day} hour {h:02d} still hits limit={limit}; "
                "refusing a truncated page")
        out.extend(chunk)
    return out


def run(fetch, keys, byname, today=None, earliest=EARLIEST, state=None,
        start_day=None, sleep_s=0.0, max_days=None, deadline=None):
    """Walk back day by day from start_day (default: today), sequentially.
    Stops at the first zero-entry day after a non-empty run, or at
    `earliest`. A failed day is retried once, then recorded in gaps."""
    state = state or empty_state()
    today = today or date.today()
    day = start_day or today
    seen_nonempty = False
    days_queried = 0
    while day >= earliest:
        if max_days is not None and days_queried >= max_days:
            break
        if deadline is not None and time.monotonic() > deadline:
            print("deadline reached; stopping with partial backfill",
                  file=sys.stderr)
            break
        try:
            entries = fetch_day(day, fetch=fetch)
        except Exception:
            try:
                entries = fetch_day(day, fetch=fetch)
            except Exception as exc:
                print(f"gap {day}: {exc}", file=sys.stderr)
                state["gaps"].add(day.isoformat())
                entries = None
        if entries is not None:
            state["gaps"].discard(day.isoformat())
            absorb_entries(state, entries, keys, byname)
            days_queried += 1
            if entries:
                seen_nonempty = True
            elif seen_nonempty:
                break  # first empty day after a non-empty run: log start
        day -= timedelta(days=1)
        if sleep_s:
            time.sleep(sleep_s)
    state["days_queried"] = days_queried
    return state


# ------------------------------------------------------------------ output

def render(state, existing=None):
    turns = dict(state["turns"])
    exposure = {k: dict(v) for k, v in state["exposure"].items()}
    gaps = set(state["gaps"])
    if existing:
        last = existing.get("last_day")
        for d, n in (existing.get("turns") or {}).items():
            if last is None or d < last:
                turns.setdefault(d, n)
        for k, days in (existing.get("exposure") or {}).items():
            dst = exposure.setdefault(k, {})
            for d, n in days.items():
                if last is None or d < last:
                    dst.setdefault(d, n)
        for rid, n in (existing.get("unmatched") or {}).items():
            state["unmatched"].setdefault(rid, n)
        gaps |= set(existing.get("gaps") or [])
    days = sorted(turns)
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "tz": TZ_LABEL,
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "days_queried": state.get("days_queried", 0),
        "turns": dict(sorted(turns.items())),
        "exposure": {k: dict(sorted(v.items()))
                     for k, v in sorted(exposure.items())},
        "unmatched": dict(sorted(state["unmatched"].items())),
        "gaps": sorted(gaps),
    }


def main():
    existing = None
    earliest = EARLIEST
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text())
        except ValueError:
            existing = None
    if existing and existing.get("last_day"):
        # Re-query only days >= last_day: it may have been partial. The
        # walk is backward from today, so clamp its floor to last_day.
        earliest = date.fromisoformat(existing["last_day"])
    keys, byname = flexiarg_key_sets(config.code_root())
    deadline = time.monotonic() + 20 * 60  # bounded backfill budget
    state = run(fetch_window, keys, byname, earliest=earliest,
                deadline=deadline)
    table = render(state, existing)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=1) + "\n")
    total_turns = sum(table["turns"].values())
    totals = Counter({k: sum(v.values()) for k, v in table["exposure"].items()})
    print(f"wrote {OUT}: {table['first_day']}..{table['last_day']} "
          f"({len(table['turns'])} days with turns), {total_turns} turns, "
          f"{len(table['gaps'])} gaps, {len(table['unmatched'])} unmatched ids")
    for k, n in totals.most_common(10):
        print(f"  {n:7d}  {k}")


if __name__ == "__main__":
    main()
