#!/usr/bin/env python3
"""Audit S4 expository scopes for the ways they can be plentiful and useless.

A run can accept hundreds of scope graphs - every one schema-valid, gated and
ledgered - and still carry nothing a reader could use. The first 12-paper
Superpod run (mark7master-20260921) did: 857 scopes, 81% the bare parent kind
`connection`, whose slot "the structure/theory connected to" accepts any noun
phrase, and half of the filled scopes failing one of the checks below. Nothing
in the pipeline reported it, because nothing looked at what the scopes say.

This is a measurement, not a gate: it never fails a run. It writes one record
per scope with the checks that scope failed, so a reader (or the scope renderer)
can see which scopes to distrust and why, and a summary of the run's shape.

Checks per filled scope, each mechanical and each a lower bound on the problem:
  echo        the fill repeats its slot's own definition ("the structure/theory
              of functors") - the model filled the hole with the question.
  unanchored  fewer than half of the fill's content words appear in the lines the
              scope cites - it is not "source-anchored text" from those lines.
  bare-noun   three words or fewer ("a diagram", "2-category") - a noun lifted
              off the line, not a statement of what the passage does with it.
A scope that fails none is `unflagged`: it passed these checks, which is not the
same as being right. Run shape: the share of each kind, and of scopes tagged
with a parent kind that has more specific children in the vocabulary.

Usage:
  scripts/expository_scope_audit.py --run-dir data/runs/<run-id> [--out audit.json]
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import expository_json
import run_manifest

STOP = set("the a an of in to for and or on with by is are as its it this that these those "
           "which be we from at into over under via our their".split())
ECHO_PREFIX = 20          # characters of a slot definition that count as repeating it
CHECKS = ("echo", "unanchored", "bare-noun")


def read_edn(paths: list[Path]) -> list[dict]:
    """EDN -> JSON through babashka, the reader the gates use (see expository_json)."""
    if not paths:
        return []
    program = ("(require '[cheshire.core :as json]) "
               "(doseq [p *command-line-args*] "
               "  (println (json/generate-string (clojure.edn/read-string (slurp p)))))")
    out = subprocess.run(["bb", "-e", program, *map(str, paths)],
                         capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def vocabulary_shape(path: Path = expository_json.VOCAB) -> tuple[dict[str, str], set[str]]:
    """({kind: slot definition}, {kinds that have children})."""
    program = ("(require '[cheshire.core :as json]) "
               "(let [v (clojure.edn/read-string (slurp (first *command-line-args*)))] "
               "  (println (json/generate-string "
               "    (for [s (:scopes v)] {:kind (subs (str (:kind s)) 1) "
               "                          :parent (some-> (:parent s) str (subs 1)) "
               "                          :type (get-in s [:hole :type])}))))")
    rows = json.loads(subprocess.run(["bb", "-e", program, str(path)],
                                     capture_output=True, text=True, check=True).stdout)
    return ({r["kind"]: r["type"] or "" for r in rows},
            {r["parent"] for r in rows if r["parent"] and r["parent"] != "root"})


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[^\W\d_]{3,}", text.lower()) if w not in STOP]


def flags(fill: str, cited: str, definitions) -> list[str]:
    found = []
    lowered = fill.lower()
    if any(d and d.lower()[:ECHO_PREFIX] in lowered for d in definitions):
        found.append("echo")
    words = content_words(fill)
    cited = cited.lower()
    if words and sum(w in cited for w in words) < len(words) / 2:
        found.append("unanchored")
    if len(fill.split()) <= 3:
        found.append("bare-noun")
    return found


def locate(fill: str, haystack: str, base: int = 0) -> list[int] | None:
    """Where the fill's own words sit in the text, as [start, end) from `base`.

    A scope cites lines, so anything it says about a passage can only be shown a
    whole line (or block) at a time. When the fill is quoted from the source, its
    exact extent can be recovered, and then a reader can be shown the words the
    scope is about. Whitespace differs (the fill is one line, the source wraps),
    so words are matched across any run of space; case and a trailing full stop
    are allowed to differ. Returns None when the fill is not in the text at all,
    which is the usual case: the model paraphrases.
    """
    words = [w for w in re.split(r"\s+", (fill or "").strip().rstrip(".")) if w]
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, haystack, re.I)
    return [base + m.start(), base + m.end()] if m else None


def audit(expo_dir: Path, candidates_dir: Path, vocab: Path = expository_json.VOCAB) -> dict:
    definitions, parents = vocabulary_shape(vocab)
    graphs = read_edn(sorted(expo_dir.glob("*.edn")))
    records, kinds, held = [], collections.Counter(), 0
    for g in graphs:
        paper, passage = g["paper/id"], g["passage/id"]
        _, region, _ = passage.split(":")
        candidate = candidates_dir / f"{paper}.{region}.candidate.json"
        window = json.loads(candidate.read_text()) if candidate.is_file() else None
        lines = window["source-window"].split("\n") if window else []
        first = window["window-lines"][0] if window else 0
        for scope in g.get("scopes", []):
            kind = scope["kind"]
            kinds[kind] += 1
            lo, hi = scope["source"]["lines"]
            cited = "\n".join(lines[max(0, lo - first):hi - first + 1])
            fill = next(iter((scope.get("slot-fill") or {}).values()), None)
            record = {"paper": paper, "passage": passage, "scope": scope["id"], "kind": kind,
                      "lines": [lo, hi], "fill": fill, "span": None,
                      "bare-parent": kind in parents}
            # A v2 scope carries the extent of its own fill; for older graphs a reader
            # that holds the paper text (the renderer) locates it.
            record["span"] = list(scope["fill-span"]) if scope.get("fill-span") else None
            record["units"] = list(scope["source"].get("units") or ())
            if fill is None:
                held += 1
                record["held"] = scope.get("held-reason") or scope.get("held") or True
                record["flags"] = []
            elif window is None:
                record["flags"] = ["no-source-window"]
            else:
                record["flags"] = flags(fill, cited, definitions.values())
            records.append(record)
    filled = [r for r in records if r["fill"] is not None]
    by_check = {c: sum(c in r["flags"] for r in filled) for c in CHECKS}
    unflagged = sum(not r["flags"] for r in filled)
    total = sum(kinds.values())
    return {"schema": "expository-scope-audit/v1",
            "summary": {"passages": len(graphs), "scopes": total, "filled": len(filled), "held": held,
                        "unflagged": unflagged, "by-check": by_check,
                        "top-kind-share": round(max(kinds.values()) / total, 3) if total else None,
                        "bare-parent-share": round(sum(r["bare-parent"] for r in records) / total, 3)
                        if total else None,
                        "kinds": dict(kinds.most_common())},
            "scopes": records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="default: <run-dir>/expository-scope-audit.json")
    args = ap.parse_args()
    doc = run_manifest.load(args.run_dir)
    art = lambda key: run_manifest.contained(args.run_dir, doc["artifacts"][key])
    result = audit(art("expo"), art("expo-candidates"))
    out = args.out or args.run_dir / "expository-scope-audit.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    s = result["summary"]
    filled = s["filled"] or 1
    print(f"{s['scopes']} scopes in {s['passages']} passages; {s['filled']} filled, {s['held']} held")
    print(f"  top kind {next(iter(s['kinds']), '-')} {s['top-kind-share']:.0%}; "
          f"bare parent kinds {s['bare-parent-share']:.0%}")
    print("  " + "  ".join(f"{c} {n} ({n / filled:.0%})" for c, n in s["by-check"].items())
          + f"  unflagged {s['unflagged']} ({s['unflagged'] / filled:.0%})")
    print(f"per-scope flags -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
