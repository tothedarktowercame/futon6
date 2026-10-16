#!/usr/bin/env python3
"""Align informal proof steps (S3 graphs) with the Lean declarations that prove them.

Prototype of the stage in holes/M-informal-lean-alignment.md. Classical: both
sides are reduced to word tokens (LaTeX commands and Unicode symbols to words,
Lean names split at case and underscore boundaries) and pairs are scored by
TF-IDF cosine. Per paper it writes

  aligned        node -> best declaration above --threshold, with the shared terms
  unaligned      informal nodes with no declaration, and declarations with no node
  holes          for each S3 :missing-warrant hole: the best otherwise-unaligned
                 helper declaration for the step, and whether one clears the bar
  retrieval      for each node, the best declarations in OTHER papers' Lean files
                 (and the shared library, when --library is given)

and a summary over all papers: coverage, hole agreement against the same test
on non-hole steps, and the retrieval hits.

  align_informal_lean.py --run-dir RUN --lean-root ROOT \\
      --lean-pattern 'problems/{paper}/lean/*.lean' --ids IDS --out DIR \\
      [--library 'ConstructionTargets/*.lean'] [--threshold 0.18]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import edn_format

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_accounting as accounting  # noqa: E402

UNICODE_WORDS = {
    "χ": "chi", "π": "pi", "σ": "sigma", "Σ": "sigma", "ζ": "zeta", "ω": "omega", "Ω": "omega",
    "α": "alpha", "β": "beta", "γ": "gamma", "Γ": "gamma", "δ": "delta", "Δ": "delta",
    "ε": "epsilon", "λ": "lambda", "μ": "mu", "φ": "phi", "ψ": "psi", "θ": "theta", "ρ": "rho",
    "τ": "tau", "ℤ": "integers", "ℝ": "reals", "ℂ": "complex", "ℕ": "naturals", "ℚ": "rationals",
    "≃": "homeomorphic", "≅": "isomorphic", "→": "to", "↦": "mapsto", "∫": "integral",
    "∑": "sum", "∂": "boundary", "∘": "compose", "×": "product", "#": "connected sum",
}
LATEX_WORDS = {"mathbb": "", "mathcal": "", "mathrm": "", "operatorname": "", "text": "",
               "cdot": "", "left": "", "right": "", "quad": "", "qquad": "", "frac": "",
               "infty": "infinity", "to": "to", "le": "", "leq": "", "ge": "", "geq": "",
               "in": "", "subseteq": "subset", "cong": "isomorphic", "simeq": "homeomorphic",
               "mathbb{Z}": "integers"}
BLACKBOARD = {"Z": "integers", "R": "reals", "C": "complex", "N": "naturals", "Q": "rationals"}
STOP = set("""a an the of to in on at by for with from and or not is are be as that this these
those it its we our let then so thus hence since if iff any every each all some there exists
such which where when one two use using via into also only both has have can must may will
proof lemma theorem def abbrev instance noncomputable private protected fun by exact show
type prop true false nat real complex int sorry have intro refine apply simp""".split())
# a few morphological equivalences the tokenizer cannot see
SYNONYMS = {"covering": "cover", "covers": "cover", "coverings": "cover", "sheeted": "sheet",
            "sheets": "sheet", "fold": "sheet", "nfold": "sheet", "homeomorphism": "homeomorphic",
            "homeomorph": "homeomorphic", "euler": "euler", "eulercharacteristic": "euler chi",
            "characteristic": "chi", "genera": "genus", "surfaces": "surface", "tori": "torus",
            "circles": "circle", "regular": "regular", "galois": "regular", "deck": "deck",
            "multiplicativity": "multiply", "multiplies": "multiply", "multiplicative": "multiply",
            "orientable": "orientable", "nonorientable": "nonorientable", "crosscaps": "crosscap",
            "projective": "projective", "rp2": "projective", "rp": "projective"}
DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|nonrec)\s+)*"
                  r"(theorem|lemma|def|abbrev|instance|structure|inductive)\s+([^\s:({\[]+)", re.M)
DOC = re.compile(r"/--(.*?)-/\s*$", re.S)


def split_name(name: str) -> list[str]:
    name = re.sub(r"(?i)^apm_?|_?[a-z]\d{2}[aj]\d{2}_?|^t\d{2}[aj]\d{2}", " ", name)
    name = name.replace(".", " ").replace("_", " ").replace("'", " ")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).split()


def stem(word: str) -> str:
    word = word.lower()
    if word in SYNONYMS:
        return SYNONYMS[word]
    for suffix in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    for ch, w in UNICODE_WORDS.items():
        text = text.replace(ch, f" {w} ")
    text = re.sub(r"\\mathbb\{?([A-Z])\}?", lambda m: " " + BLACKBOARD.get(m.group(1), m.group(1)) + " ", text)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: " " + LATEX_WORDS.get(m.group(1), m.group(1)) + " ", text)
    words = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9]*", text):
        for part in (split_name(raw) if re.search(r"[a-z][A-Z]", raw) else [raw]):
            for w in stem(part).split():
                if len(w) > 1 and w.lower() not in STOP:
                    words.append(w.lower())
    return words


def lean_decls(path: Path, paper: str) -> list[dict]:
    text = path.read_text(errors="replace")
    marks = list(DECL.finditer(text))
    out = []
    for n, m in enumerate(marks):
        end = marks[n + 1].start() if n + 1 < len(marks) else len(text)
        chunk = text[m.start():end]
        sig, _, body = chunk.partition(":=")
        before = text[max(0, m.start() - 3000):m.start()]
        # the docstring immediately before this declaration, not the first one in reach
        k = before.rfind("/--")
        doc = DOC.search(before[k:]) if k >= 0 else None
        out.append({"paper": paper, "file": str(path), "kind": m.group(1), "name": m.group(2),
                    "proved": not re.search(r"\bsorry\b", body) if body else None,
                    "line": text.count("\n", 0, m.start()) + 1,
                    "doc": " ".join(doc.group(1).split()) if doc else "",
                    "signature": " ".join(sig.split())[:600], "body": body})
    names = {d["name"] for d in out}
    for d in out:
        d["uses"] = sorted(x for x in names if x != d["name"] and re.search(r"(?<![\w.])" + re.escape(x) + r"(?![\w'])", d["body"]))
        d["terms"] = split_name(d["name"]) * 3 + tokens(d["doc"]) + tokens(d["signature"].split(":", 1)[-1])
        d["terms"] = [t for t in (stem(w).split() for w in d["terms"]) for t in t if len(t) > 1 and t not in STOP]
        del d["body"]
    return out


def s3_graph(run_dir: Path, paper: str) -> dict | None:
    graphs, _refused = accounting.accepted_finals(run_dir / "artifacts" / "graphs")
    best = None
    for _item, path in graphs:
        if path.name.startswith(paper + "__p") and not path.name.endswith(".rung2.edn"):
            best = path
            break
    if best is None:
        return None
    doc = edn_format.loads(best.read_text())
    K = edn_format.Keyword
    get = lambda m, k: m.get(K(k)) if hasattr(m, "get") else None
    nodes = [{"id": str(get(n, "id")).lstrip(":"), "kind": str(get(n, "kind")).lstrip(":"),
              "text": get(n, "text") or "", "gloss": get(n, "gloss") or ""} for n in get(doc, "nodes") or []]
    edges = []
    for e in get(doc, "edges") or []:
        w = get(e, "warrant") or {}
        edges.append({"id": str(get(e, "id")).lstrip(":"),
                      "premise": [str(p).lstrip(":") for p in get(e, "premise") or []],
                      "conclusion": str(get(e, "conclusion")).lstrip(":"),
                      "warrant": get(w, "text") or "", "missing": str(get(w, "kind")) == ":missing-warrant"})
    for n in nodes:
        n["terms"] = tokens(n["text"]) + tokens(n["gloss"])
    return {"graph": best.name, "nodes": nodes, "edges": edges}


class Tfidf:
    def __init__(self, docs: list[list[str]]):
        df = Counter(t for d in docs for t in set(d))
        self.n = len(docs)
        self.idf = {t: math.log((1 + self.n) / (1 + c)) + 1 for t, c in df.items()}

    def vec(self, terms: list[str]) -> dict:
        tf = Counter(terms)
        v = {t: (1 + math.log(c)) * self.idf.get(t, math.log(1 + self.n) + 1) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    @staticmethod
    def cos(a: dict, b: dict) -> tuple[float, list[str]]:
        shared = sorted(set(a) & set(b), key=lambda t: -(a[t] * b[t]))
        return sum(a[t] * b[t] for t in shared), shared[:6]


MAIN = re.compile(r"(?i)^apm_[a-z]\d{2}[aj]\d{2}$")


def capacity(d: dict) -> int:
    """How many informal nodes one declaration may answer for: a lemma proves one
    step; the main statement and definitions may stand behind several."""
    if MAIN.match(d["name"]) or d["kind"] not in ("theorem", "lemma"):
        return 3
    return 1


def align_paper(graph: dict, decls: list[dict], model: Tfidf, threshold: float) -> dict:
    """Score every node against every declaration, then assign best-first.

    Taking each node's single best declaration let broad ones soak up many steps
    (in t98A01 the definition t98A01IsRegularNfoldCover took 8 claims, and the
    lemma that proves the step S3 flagged was spent on connected-sum claims).
    A claim prefers a theorem or lemma to a definition; a lemma is given to one
    node only.
    """
    dvec = {d["name"]: model.vec(d["terms"]) for d in decls}
    by_name = {d["name"]: d for d in decls}
    edges_into = defaultdict(list)
    for e in graph["edges"]:
        edges_into[e["conclusion"]].append(e)
    pairs = []
    for n in graph["nodes"]:
        # a node is described by its own words and by the warrant of the step into it
        terms = n["terms"] + [t for e in edges_into[n["id"]] for t in tokens(e["warrant"])]
        nv = model.vec(terms)
        for d in decls:
            s, shared = model.cos(nv, dvec[d["name"]])
            if n["kind"] == "claim" and d["kind"] not in ("theorem", "lemma"):
                s *= 0.6
            pairs.append((s, n["id"], d["name"], shared))
    pairs.sort(key=lambda x: -x[0])
    node_of, load = {}, Counter()
    for s, nid, dname, shared in pairs:
        if s < threshold:
            break
        if nid in node_of or load[dname] >= capacity(by_name[dname]):
            continue
        node_of[nid] = (dname, round(s, 3), shared)
        load[dname] += 1
    rows = []
    for n in graph["nodes"]:
        hit = node_of.get(n["id"])
        rows.append({"node": n["id"], "decl": hit[0] if hit else None, "score": hit[1] if hit else 0.0,
                     "shared": hit[2] if hit else [], "node_text": n["gloss"] or n["text"][:120]})
    used = {r["decl"] for r in rows if r["decl"]}

    # A lemma proves a STEP -- hypotheses to conclusion -- not a lone claim:
    # t98A01_orientable_genus_arithmetic takes g = h and χ(g) = 5χ(h) and gives
    # genus 1, which is S3's step e4 (n3, n4, n5 => n6). So lemmas are matched to
    # steps, each described by its conclusion, its premises and its warrant.
    by_id = {n["id"]: n for n in graph["nodes"]}
    lemmas = [d for d in decls if d["kind"] in ("theorem", "lemma") and not MAIN.match(d["name"])]
    step_pairs = []
    for e in graph["edges"]:
        concl = by_id.get(e["conclusion"])
        if concl is None:
            continue
        terms = concl["terms"] * 2 + [t for p in e["premise"] if p in by_id for t in by_id[p]["terms"]]
        terms += tokens(e["warrant"]) * 2
        sv = model.vec(terms)
        for d in lemmas:
            s, shared = model.cos(sv, dvec[d["name"]])
            step_pairs.append((s, e["id"], d["name"], shared))
    step_pairs.sort(key=lambda x: -x[0])
    lemma_of, taken = {}, set()
    for s, eid, dname, shared in step_pairs:
        if s < threshold:
            break
        if eid in lemma_of or dname in taken:
            continue
        lemma_of[eid] = (dname, round(s, 3), shared)
        taken.add(dname)

    holes, others = [], []
    for e in graph["edges"]:
        concl = by_id.get(e["conclusion"])
        if concl is None:
            continue
        hit = lemma_of.get(e["id"])
        rec = {"edge": e["id"], "premises": e["premise"], "conclusion": e["conclusion"],
               "step": concl["gloss"] or concl["text"][:120], "warrant": e["warrant"][:120],
               "lemma": hit[0] if hit else None, "score": hit[1] if hit else 0.0,
               "shared": hit[2] if hit else [], "matched": bool(hit)}
        (holes if e["missing"] else others).append(rec)
    return {"aligned": rows,
            "unaligned_decls": [d["name"] for d in decls if d["name"] not in used],
            "helpers": [d["name"] for d in decls if d["kind"] in ("theorem", "lemma") and not MAIN.match(d["name"])],
            "holes": holes, "non_hole_steps": others}


def retrieval(graph: dict, paper: str, catalogue: list[dict], model: Tfidf, k: int = 5) -> list[dict]:
    cvec = [(d, model.vec(d["terms"])) for d in catalogue if d["paper"] != paper]
    out = []
    for n in graph["nodes"]:
        nv = model.vec(n["terms"])
        top = sorted(((model.cos(nv, v), d) for d, v in cvec), key=lambda x: -x[0][0])[:k]
        out.append({"node": n["id"], "node_text": n["gloss"] or n["text"][:120],
                    "hits": [{"decl": d["name"], "paper": d["paper"], "score": round(s, 3), "shared": sh}
                             for (s, sh), d in top]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--lean-root", required=True, type=Path)
    ap.add_argument("--lean-pattern", default="problems/{paper}/lean/*.lean")
    ap.add_argument("--library", help="glob under --lean-root for shared library files")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.18)
    args = ap.parse_args()
    papers = [l.strip() for l in open(args.ids) if l.strip() and not l.startswith("#")]

    decls = {p: [d for f in sorted(glob.glob(str(args.lean_root / args.lean_pattern.format(paper=p))))
                 for d in lean_decls(Path(f), p)] for p in papers}
    library = []
    if args.library:
        for f in sorted(glob.glob(str(args.lean_root / args.library))):
            library += lean_decls(Path(f), "LIB:" + Path(f).stem)
    graphs = {p: s3_graph(args.run_dir, p) for p in papers}
    docs = [d["terms"] for ds in decls.values() for d in ds] + [d["terms"] for d in library]
    docs += [n["terms"] for g in graphs.values() if g for n in g["nodes"]]
    model = Tfidf(docs)
    catalogue = [d for ds in decls.values() for d in ds] + library

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"threshold": args.threshold, "papers": {}}
    for p in papers:
        g = graphs[p]
        if g is None or not decls[p]:
            summary["papers"][p] = {"skipped": "no S3 graph" if g is None else "no Lean declarations"}
            continue
        res = align_paper(g, decls[p], model, args.threshold)
        res["retrieval"] = retrieval(g, p, catalogue, model)
        res.update({"paper": p, "graph": g["graph"], "declarations": len(decls[p])})
        (args.out / f"{p}.alignment.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        nodes = len(res["aligned"])
        summary["papers"][p] = {
            "nodes": nodes, "aligned": sum(1 for r in res["aligned"] if r["decl"]),
            "declarations": len(decls[p]), "helpers": len(res["helpers"]),
            "holes": len(res["holes"]), "holes_matched": sum(h["matched"] for h in res["holes"]),
            "steps": len(res["non_hole_steps"]), "steps_matched": sum(h["matched"] for h in res["non_hole_steps"])}
    rows = [v for v in summary["papers"].values() if "nodes" in v]
    tot = lambda k: sum(r[k] for r in rows)
    summary["totals"] = {k: tot(k) for k in ("nodes", "aligned", "holes", "holes_matched", "steps", "steps_matched")}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    t = summary["totals"]
    print(f"{len(rows)} papers aligned; nodes aligned {t['aligned']}/{t['nodes']}; "
          f"holes matched by a helper {t['holes_matched']}/{t['holes']}; "
          f"non-hole steps matched {t['steps_matched']}/{t['steps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
