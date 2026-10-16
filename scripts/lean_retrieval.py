#!/usr/bin/env python3
"""Find proved Lean declarations by mathematical meaning.

A lemma index lists declarations by name and signature, so it only answers
someone who already knows what the lemma is called. This answers an informal
claim ("z -> z^5 is a 5-to-1 covering of the circle", "Euler characteristic
multiplies under a finite cover") with the proved declarations whose names,
docstrings and statements share its mathematics, across every Lean tree
indexed.

Index once, query many times:

  lean_retrieval.py index --root REPO --glob 'problems/*/lean/*.lean' \\
      --glob 'ConstructionTargets/*.lean' --out lean-index.json
  lean_retrieval.py query --index lean-index.json "Euler characteristic of a connected sum"
  lean_retrieval.py query --index lean-index.json --run-dir RUN --paper t98A01

The second query form takes the informal steps of a paper from a mining run
(its S3 proof graph) and answers each in turn; that paper's own declarations
are left out of its answers. Each hit gives the declaration, its file and
line, a score, and the terms that matched. Only declarations whose proofs
contain no `sorry` are indexed unless --include-sorry is given.

Matching is word overlap over a shared vocabulary (LaTeX commands, Unicode
symbols and Lean name parts reduced to the same words), weighted by TF-IDF.
It finds candidates; it proves nothing. See holes/M-informal-lean-alignment.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import align_informal_lean as align  # noqa: E402

SCHEMA = "futon6-lean-retrieval-index/v1"


def owner(rel: Path) -> str:
    """The problem a file belongs to (problems/<id>/...), else 'LIB:<module>'."""
    parts = rel.parts
    if "problems" in parts and parts.index("problems") + 1 < len(parts):
        return parts[parts.index("problems") + 1]
    return "LIB:" + rel.stem


def build_index(root: Path, globs: list[str], include_sorry: bool = False) -> dict:
    decls = []
    for pattern in globs:
        for f in sorted(glob.glob(str(root / pattern), recursive=True)):
            rel = Path(f).relative_to(root)
            for d in align.lean_decls(Path(f), owner(rel)):
                if d["kind"] not in ("theorem", "lemma"):
                    continue
                if not include_sorry and d.get("proved") is False:
                    continue
                d["file"] = str(rel)
                decls.append(d)
    return {"schema": SCHEMA, "globs": globs, "declarations": decls}


class Retriever:
    def __init__(self, index: dict):
        self.decls = index["declarations"]
        self.model = align.Tfidf([d["terms"] for d in self.decls])
        self.vecs = [self.model.vec(d["terms"]) for d in self.decls]

    def query_terms(self, terms: list[str], k: int = 5, exclude: str | None = None) -> list[dict]:
        qv = self.model.vec(terms)
        scored = []
        for d, v in zip(self.decls, self.vecs):
            if exclude and d["paper"] == exclude:
                continue
            s, shared = self.model.cos(qv, v)
            if s > 0:
                scored.append((s, shared, d))
        scored.sort(key=lambda x: -x[0])
        return [{"decl": d["name"], "owner": d["paper"], "file": d["file"], "line": d["line"],
                 "score": round(s, 3), "shared": shared, "doc": d["doc"][:160]}
                for s, shared, d in scored[:k]]

    def query(self, text: str, k: int = 5, exclude: str | None = None) -> list[dict]:
        return self.query_terms(align.tokens(text), k, exclude)


def show(label: str, hits: list[dict]) -> None:
    print(label)
    for h in hits:
        print(f"   {h['score']:.2f}  {h['decl']}  ({h['file']}:{h['line']})  [{', '.join(h['shared'][:4])}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index")
    ix.add_argument("--root", required=True, type=Path)
    ix.add_argument("--glob", action="append", required=True)
    ix.add_argument("--out", required=True, type=Path)
    ix.add_argument("--include-sorry", action="store_true")
    q = sub.add_parser("query")
    q.add_argument("--index", required=True, type=Path)
    q.add_argument("text", nargs="*")
    q.add_argument("--run-dir", type=Path)
    q.add_argument("--paper")
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.cmd == "index":
        index = build_index(args.root, args.glob, args.include_sorry)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(index, ensure_ascii=False) + "\n")
        print(f"indexed {len(index['declarations'])} proved declarations -> {args.out}")
        return 0

    r = Retriever(json.loads(args.index.read_text()))
    results = []
    if args.text:
        text = " ".join(args.text)
        results.append({"query": text, "hits": r.query(text, args.k)})
    if args.paper:
        if not args.run_dir:
            ap.error("--paper needs --run-dir")
        graph = align.s3_graph(args.run_dir, args.paper)
        if graph is None:
            print(f"no accepted S3 graph for {args.paper} in {args.run_dir}", file=sys.stderr)
            return 1
        for n in graph["nodes"]:
            results.append({"query": n["gloss"] or n["text"], "node": n["id"],
                            "hits": r.query_terms(n["terms"], args.k, exclude=args.paper)})
    if not results:
        ap.error("give query text, or --run-dir with --paper")
    if args.json:
        print(json.dumps(results, indent=1, ensure_ascii=False))
    else:
        for res in results:
            show(f"\n{res.get('node', 'query')}: {res['query'][:100]}", res["hits"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
