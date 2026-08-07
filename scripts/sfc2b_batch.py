#!/usr/bin/env python3
"""SFC2b batch adapter — run LLM symbol grounding over a corpus, per paper.

`sfc_symbol_grounding.py` grounds ONE formula against ONE context. That is the
right unit for the grounding loop and the wrong unit for a run: nothing in the
pipeline turned a corpus into those pairs, so SFC2b stayed "built, stub-validated,
never run at corpus scale" — a card that reads as a gap when what it lacked was
an adapter.

The pairing has to come from the snippets, not from `def-formulae.txt`. That file
is a deduplicated list of formula strings with the paper association discarded,
which is fine for canonical-shape counting (S11) and useless here: SFC2b binds a
symbol to its *per-paper* domain, so it needs the paper's own prose as context.
`data/warp/def-snippets.json` keeps `{paper, surface, snippet}` together, so this
reads there and reuses def_formulae_extract's own regexes rather than growing a
second, drifting copy of "what counts as a defining formula".

    sfc2b_batch.py --ids holes/mark7-16.ids.txt --backend stub --out /tmp/sfc2b.json
    sfc2b_batch.py --ids ... --backend openai --model glm-4.5-air \
                   --run-dir data/runs/$RUN_ID --checkpoint /tmp/sfc2b.jsonl

Checkpointing follows cas_select: one JSON line per paper, fsynced, resume keyed
on paper id. A corpus-scale LLM pass that only writes at the end is a wager on
the last call succeeding (H24).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from def_formulae_extract import PATTERNS, RELATION, clean  # noqa: E402
from sfc_symbol_grounding import ground  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _r(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def pairs_by_paper(snippets_path: Path, want: set[str] | None,
                   max_per_paper: int, min_len: int) -> dict[str, list[dict]]:
    """(paper -> [{formula, context, concept}]) using def_formulae_extract's rules.

    Dedup is per paper, not global: the same formula appearing in two papers is
    two grounding questions, since the binding is to each paper's own prose.
    """
    d = json.loads(snippets_path.read_text())
    snippets = d.get("snippets", d)
    out: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = defaultdict(set)
    for concept, rows in snippets.items():
        for row in (rows if isinstance(rows, list) else [rows]):
            paper = row.get("paper")
            if not paper or (want and paper not in want):
                continue
            # `len(out[paper])` would auto-vivify a defaultdict entry, so every
            # paper merely *seen* became a paper *with formulae* — the run
            # reported 10 papers when 5 had any content, and 0-formula papers
            # then aggregated as legitimate 0-grounded rows.
            if len(out.get(paper, ())) >= max_per_paper:
                continue
            text = row.get("snippet") or ""
            for rx in PATTERNS:
                for m in rx.finditer(text):
                    f = clean(m.group(1))
                    if len(f) < min_len or not RELATION.search(f):
                        continue
                    if f in seen[paper]:
                        continue
                    seen[paper].add(f)
                    out.setdefault(paper, []).append(
                        {"formula": f, "context": text, "concept": concept})
                    if len(out[paper]) >= max_per_paper:
                        break
                if len(out.get(paper, ())) >= max_per_paper:
                    break
    return out


def ground_paper(paper: str, items: list[dict], backend: str, model: str) -> dict:
    """Ground every formula for one paper. A failure costs one formula, not the run."""
    rows, errors = [], []
    for it in items:
        try:
            res = ground(it["formula"], it["context"], backend, model)
        except Exception as e:  # noqa: BLE001
            errors.append({"formula": it["formula"], "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append({"formula": it["formula"], "concept": it["concept"],
                     "summary": res.get("summary", {}), "groundings": res.get("groundings", [])})
    agg = defaultdict(int)
    for r in rows:
        for k, v in r["summary"].items():
            agg[k] += v
    return {"paper_id": paper, "formulae": len(items), "results": rows,
            "summary": dict(agg), "errors": errors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets", default="data/warp/def-snippets.json")
    ap.add_argument("--ids", help="restrict to these paper ids (one per line)")
    ap.add_argument("--backend", choices=["stub", "openai"], default="stub")
    ap.add_argument("--model", default="mark4-70b")
    ap.add_argument("--max-per-paper", type=int, default=12)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--out", help="write the assembled payload here (default stdout)")
    ap.add_argument("--checkpoint", type=Path,
                    help="append each paper as JSON lines; resume skips those present")
    ap.add_argument("--run-dir", help="emit the S5 symbol-grounding metric here")
    ap.add_argument("--run-id", default="adhoc")
    ap.add_argument("--corpus-id", default="adhoc")
    a = ap.parse_args(argv)

    want = None
    if a.ids:
        want = {l.strip() for l in _r(a.ids).read_text().splitlines() if l.strip()}

    by_paper = pairs_by_paper(_r(a.snippets), want, a.max_per_paper, a.min_len)
    print(f"{len(by_paper)} paper(s), {sum(len(v) for v in by_paper.values())} formula(e)",
          file=sys.stderr)

    done: dict[str, dict] = {}
    if a.checkpoint and a.checkpoint.exists():
        for line in a.checkpoint.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue                      # torn final line from a hard kill
            if isinstance(row, dict) and "paper_id" in row:
                done[row["paper_id"]] = row
        if done:
            print(f"resuming: {len(done)} paper(s) already in {a.checkpoint}", file=sys.stderr)

    todo = [p for p in sorted(by_paper) if p not in done]
    for i, paper in enumerate(todo, 1):
        row = ground_paper(paper, by_paper[paper], a.backend, a.model)
        done[paper] = row
        if a.checkpoint:
            with a.checkpoint.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        s = row["summary"]
        print(f"  [{i}/{len(todo)}] {paper}: {row['formulae']} formula(e), "
              f"{s.get('grounded', 0)} grounded, {s.get('unsupported', 0)} unsupported, "
              f"{len(row['errors'])} error(s)", file=sys.stderr, flush=True)

    total = defaultdict(int)
    for row in done.values():
        for k, v in row["summary"].items():
            total[k] += v
    payload = {"papers": len(done), "summary": dict(total), "results": done}

    if a.run_dir:
        # Same metric sfc_symbol_grounding emits per formula, aggregated per run.
        try:
            import metric_harness as mh
            tot = sum(total.get(k, 0) for k in ("grounded", "undefined_in_context", "unsupported"))
            if tot:
                mh.emit_record(a.run_dir, run_id=a.run_id, corpus_id=a.corpus_id,
                               paper_id="corpus", stage="S5",
                               metric="symbol-grounding/variable", axis="completeness",
                               value=round(total.get("grounded", 0) / tot, 4), computable=True)
        except Exception as e:  # noqa: BLE001
            print(f"  (S5 metric emit skipped: {e})", file=sys.stderr)

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if a.out:
        _r(a.out).write_text(text)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
