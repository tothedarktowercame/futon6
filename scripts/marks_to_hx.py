#!/usr/bin/env python3
"""Import first-cut CT anatomy marks into substrate-2a hx/ grounding stores.

This is the re-runnable version of the shape-lock prototype for the grounding
projection (#3).  It emits one EDN hx/ store per paper plus a deterministic QA
report.  The metric intentionally matches SUBSTRATE-2A-BASELINE.md:

    grounded% = filled / (filled + open)

where filled mirrors the prototype counter: symbol-grounded + resolved
let-binder + bind/typed, and open = bare symbol + unresolved let-binder.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path(
    "/home/joe/code/futon6/data/showcases/ct-anatomy/golden"
)
DEFAULT_OUT = Path("/home/joe/code/futon6/data/substrate-2a")
BASELINE_PAPERS = ("math__0703763", "0704.0502", "2406.09832")


@dataclass(frozen=True)
class Edge:
    edge_id: str
    kind: str
    ends: tuple[tuple[str, str], ...]
    status: str
    source: dict[str, int]
    attrs: dict[str, str]


@dataclass
class PaperImport:
    paper: str
    file: Path
    edges: list[Edge]
    nodes: dict[str, str]
    filled: int
    open: int
    symbol_grounded: int
    symbol_open: int
    let_filled: int
    let_open: int
    bind_typed: int

    @property
    def denominator(self) -> int:
        return self.filled + self.open

    @property
    def grounded_pct(self) -> float:
        return (100.0 * self.filled / self.denominator) if self.denominator else 0.0


def field(mark: dict[str, Any], key: str) -> str | None:
    for k, v in mark.get("fields", []):
        if k == key:
            return str(v) if v is not None else None
    return None


def paper_id_from_path(path: Path) -> str:
    name = path.name
    if name.startswith("fable-"):
        name = name[len("fable-") :]
    if name.endswith("-dp-emacs.json"):
        name = name[: -len("-dp-emacs.json")]
    return name


def clean(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def span_text(text: str, mark: dict[str, Any]) -> str:
    start, end = mark.get("start"), mark.get("end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text):
        return clean(text[start:end])
    return ""


def source_span(mark: dict[str, Any]) -> dict[str, int]:
    start, end = mark.get("start"), mark.get("end")
    out: dict[str, int] = {}
    if isinstance(start, int):
        out["start"] = start
    if isinstance(end, int):
        out["end"] = end
    return out


def slug(value: str, fallback: str) -> str:
    base = clean(value)
    if base.startswith("$") and base.endswith("$") and len(base) >= 2:
        base = base[1:-1].strip()
    base = base.replace("\\", "")
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-")
    if not base:
        base = fallback
    if len(base) > 72:
        base = base[:60].rstrip("-")
    if not re.match(r"^[A-Za-z_?]", base):
        base = f"n-{base}"
    return base or fallback


def node_id(value: str, used: dict[str, str], fallback: str) -> str:
    value = clean(value)
    base = slug(value, fallback)
    current = used.get(base)
    if current is None or current == value:
        used[base] = value
        return base
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}-{suffix}"
    used[candidate] = value
    return candidate


def edn_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def source_edn(src: dict[str, int]) -> str:
    parts = []
    if "start" in src:
        parts.append(f":start {src['start']}")
    if "end" in src:
        parts.append(f":end {src['end']}")
    return "{" + " ".join(parts) + "}"


def attrs_edn(attrs: dict[str, str]) -> str:
    parts = [f":{k} {edn_string(v)}" for k, v in sorted(attrs.items())]
    return "{" + " ".join(parts) + "}"


def render_edge(edge: Edge) -> str:
    ends = " ".join(
        f"{{:role :{role}, :node :{node}}}" for role, node in edge.ends
    )
    return (
        f"  {{:id :{edge.edge_id}, :kind :{edge.kind}, "
        f":ends [{ends}], :status :{edge.status}, "
        f":source {source_edn(edge.source)}, :attrs {attrs_edn(edge.attrs)}}}"
    )


def render_store(imported: PaperImport, cut_id: dict[str, str | int]) -> str:
    node_lines = [
        f"  {{:id :{node}, :label {edn_string(label)}}}"
        for node, label in sorted(imported.nodes.items())
    ]
    edge_lines = [render_edge(edge) for edge in imported.edges]
    return "\n".join(
        [
            "{:store/id \"substrate-2a/grounding\"",
            " :format/version 1",
            " :projection :grounding",
            f" :paper/id {edn_string(imported.paper)}",
            f" :source/file {edn_string(str(imported.file))}",
            f" :source/cut {{:source-dir {edn_string(str(cut_id['source_dir']))} "
            f":file-count {cut_id['file_count']} :selection {edn_string(str(cut_id['selection']))} "
            f":selection-count {cut_id['selection_count']} :fingerprint {edn_string(str(cut_id['fingerprint']))}}}",
            f" :qa/metrics {{:filled {imported.filled} :open {imported.open} "
            f":grounded-pct {imported.grounded_pct:.1f} "
            f":symbol-grounded {imported.symbol_grounded} :symbol-open {imported.symbol_open} "
            f":let-filled {imported.let_filled} :let-open {imported.let_open} "
            f":bind-typed {imported.bind_typed}}}",
            " :nodes [",
            *node_lines,
            " ]",
            " :hyperedges [",
            *edge_lines,
            " ]}",
            "",
        ]
    )


def import_paper(path: Path) -> PaperImport:
    data = json.loads(path.read_text())
    text = data.get("text", "")
    paper = paper_id_from_path(path)
    used_nodes: dict[str, str] = {}
    nodes: dict[str, str] = {}
    edges: list[Edge] = []
    filled = open_ = symbol_grounded = symbol_open = let_filled = let_open = bind_typed = 0

    def intern(value: str, fallback: str) -> str:
        nid = node_id(value, used_nodes, fallback)
        nodes[nid] = clean(value)
        return nid

    def add_edge(kind: str, ends: Iterable[tuple[str, str]], status: str,
                 mark: dict[str, Any], attrs: dict[str, str]) -> None:
        edge_no = len(edges) + 1
        edges.append(
            Edge(
                edge_id=f"e-{edge_no:06d}",
                kind=kind,
                ends=tuple(ends),
                status=status,
                source=source_span(mark),
                attrs={k: clean(v) for k, v in attrs.items() if clean(v)},
            )
        )

    for mark in data.get("marks", []):
        kind = mark.get("kind")
        if kind == "symbol-grounded":
            sym, bound = clean(field(mark, "symbol")), clean(field(mark, "bound"))
            if sym and bound:
                add_edge(
                    "bind",
                    [("subject", intern(sym, "symbol")), ("concept", intern(bound, "concept"))],
                    "filled",
                    mark,
                    {"mark-kind": kind, "subject": sym, "concept": bound},
                )
                filled += 1
                symbol_grounded += 1
        elif kind == "symbol":
            sym = span_text(text, mark) or clean(field(mark, "symbol")) or "?"
            add_edge(
                "bind",
                [("subject", intern(sym, "symbol")), ("concept", "?")],
                "open",
                mark,
                {"mark-kind": kind, "subject": sym},
            )
            open_ += 1
            symbol_open += 1
        elif kind == "let-binder":
            binds, as_, canon = clean(field(mark, "binds")), clean(field(mark, "as")), clean(field(mark, "canon"))
            if not binds:
                continue
            filler = "" if (canon and "unresolved" in canon) else (canon or as_)
            if filler:
                add_edge(
                    "bind",
                    [("subject", intern(binds, "let")), ("concept", intern(filler, "concept"))],
                    "filled",
                    mark,
                    {"mark-kind": kind, "subject": binds, "concept": filler},
                )
                filled += 1
                let_filled += 1
            else:
                add_edge(
                    "bind",
                    [("subject", intern(binds, "let")), ("concept", "?")],
                    "open",
                    mark,
                    {"mark-kind": kind, "subject": binds},
                )
                open_ += 1
                let_open += 1
        elif kind == "bind/typed":
            sym, typ = clean(field(mark, "symbol")), clean(field(mark, "type"))
            if sym and typ:
                add_edge(
                    "bind-typed",
                    [("subject", intern(sym, "symbol")), ("type", intern(typ, "type"))],
                    "filled",
                    mark,
                    {"mark-kind": kind, "subject": sym, "type": typ},
                )
                filled += 1
                bind_typed += 1

    return PaperImport(
        paper=paper,
        file=path,
        edges=edges,
        nodes=nodes,
        filled=filled,
        open=open_,
        symbol_grounded=symbol_grounded,
        symbol_open=symbol_open,
        let_filled=let_filled,
        let_open=let_open,
        bind_typed=bind_typed,
    )


def all_input_files(source_dir: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(source_dir / "*-dp-emacs.json")))]


def select_files(source_dir: Path, sample_size: int, papers: list[str] | None) -> tuple[str, list[Path]]:
    all_files = all_input_files(source_dir)
    by_id = {paper_id_from_path(path): path for path in all_files}
    if papers:
        missing = [paper for paper in papers if paper not in by_id]
        if missing:
            raise SystemExit(f"missing requested paper(s): {', '.join(missing)}")
        return "explicit:" + ",".join(papers), [by_id[paper] for paper in papers]
    return f"sorted-first:{sample_size}", all_files[:sample_size]


def fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in files:
        stat = path.stat()
        h.update(path.name.encode("utf-8"))
        h.update(str(stat.st_size).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def render_manifest(cut_id: dict[str, str | int], imports: list[PaperImport]) -> str:
    papers = " ".join(edn_string(p.paper) for p in imports)
    return "\n".join(
        [
            "{:store/id \"substrate-2a/grounding-sample\"",
            " :format/version 1",
            f" :source-dir {edn_string(str(cut_id['source_dir']))}",
            f" :source-file-count {cut_id['file_count']}",
            f" :selection {edn_string(str(cut_id['selection']))}",
            f" :selection-count {cut_id['selection_count']}",
            f" :fingerprint {edn_string(str(cut_id['fingerprint']))}",
            f" :papers [{papers}]}}",
            "",
        ]
    )


def render_report(cut_id: dict[str, str | int], imports: list[PaperImport],
                  baseline: list[PaperImport]) -> str:
    total_filled = sum(p.filled for p in imports)
    total_open = sum(p.open for p in imports)
    total_edges = sum(len(p.edges) for p in imports)
    denom = total_filled + total_open
    pooled = (100.0 * total_filled / denom) if denom else 0.0

    lines = [
        "# substrate-2a grounding import QA",
        "",
        "Generated by `scripts/marks_to_hx.py`.",
        "",
        "## Cut Identity",
        "",
        f"- source dir: `{cut_id['source_dir']}`",
        f"- source file count: `{cut_id['file_count']}`",
        f"- selection: `{cut_id['selection']}`",
        f"- selection count: `{cut_id['selection_count']}`",
        f"- selection fingerprint: `{cut_id['fingerprint']}`",
        "- metric: `grounded% = filled / (filled + open)`",
        "- filled: `symbol-grounded` plus resolved `let-binder` plus `bind/typed` (prototype-compatible counter)",
        "- open: bare `symbol` plus unresolved `let-binder`",
        "- note: `bind/typed` is emitted as `:bind-typed`, while `symbol-grounded` and `let-binder` fills are emitted as `:bind`",
        "",
        "## Sample Metrics",
        "",
        "| paper | hx edges | filled | open | grounded% | symbol-grounded | symbol-open | let-filled | let-open | bind-typed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in imports:
        lines.append(
            f"| {p.paper} | {len(p.edges)} | {p.filled} | {p.open} | {p.grounded_pct:.1f} | "
            f"{p.symbol_grounded} | {p.symbol_open} | {p.let_filled} | {p.let_open} | {p.bind_typed} |"
        )
    lines.extend(
        [
            f"| **pooled** | **{total_edges}** | **{total_filled}** | **{total_open}** | **{pooled:.1f}** |  |  |  |  |  |",
            "",
            "## Baseline Tripwire",
            "",
            "The prototype's shape-lock papers are re-imported with the same metric.",
            "",
            "| paper | filled | open | grounded% | expected prototype grounded% |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    expected = {"math__0703763": 90.4, "0704.0502": 72.0, "2406.09832": 42.4}
    for p in baseline:
        lines.append(
            f"| {p.paper} | {p.filled} | {p.open} | {p.grounded_pct:.1f} | {expected[p.paper]:.1f} |"
        )
    b_filled = sum(p.filled for p in baseline)
    b_open = sum(p.open for p in baseline)
    b_pct = (100.0 * b_filled / (b_filled + b_open)) if (b_filled + b_open) else 0.0
    lines.extend(
        [
            f"| **pooled tripwire** | **{b_filled}** | **{b_open}** | **{b_pct:.1f}** |  |",
            "",
            "## ScopeQuery Dogfood",
            "",
            "The existing runtime can load any generated paper EDN with a one-line `GRAPH` change. Example:",
            "",
            "```python",
            "GRAPH = Path(\"/home/joe/code/futon6/data/substrate-2a/hx/0704.0502.edn\")",
            "```",
            "",
            "Then the existing Q1 scope `{:kind :bind (:subject :A) (:concept ?c)}` answers from real filled `:bind` edges when that subject occurs in the selected paper.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(out_dir: Path, imports: list[PaperImport], baseline: list[PaperImport],
                  cut_id: dict[str, str | int]) -> None:
    hx_dir = out_dir / "hx"
    hx_dir.mkdir(parents=True, exist_ok=True)
    for old in hx_dir.glob("*.edn"):
        old.unlink()
    for imported in imports:
        (hx_dir / f"{imported.paper}.edn").write_text(render_store(imported, cut_id))
    (out_dir / "manifest.edn").write_text(render_manifest(cut_id, imports))
    (out_dir / "QA-METRICS.md").write_text(render_report(cut_id, imports, baseline))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument(
        "--papers",
        help="Comma-separated paper ids to import instead of the deterministic sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    papers = [p.strip() for p in args.papers.split(",") if p.strip()] if args.papers else None
    selection, files = select_files(args.source_dir, args.sample_size, papers)
    all_files = all_input_files(args.source_dir)
    imports = [import_paper(path) for path in files]
    _, baseline_files = select_files(args.source_dir, args.sample_size, list(BASELINE_PAPERS))
    baseline_imports = [import_paper(path) for path in baseline_files]
    cut_id: dict[str, str | int] = {
        "source_dir": str(args.source_dir),
        "file_count": len(all_files),
        "selection": selection,
        "selection_count": len(files),
        "fingerprint": fingerprint(files),
    }
    if args.out_dir.exists():
        args.out_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.out_dir.mkdir(parents=True)
    write_outputs(args.out_dir, imports, baseline_imports, cut_id)

    total_filled = sum(p.filled for p in imports)
    total_open = sum(p.open for p in imports)
    denom = total_filled + total_open
    pct = (100.0 * total_filled / denom) if denom else 0.0
    print(
        f"imported {len(imports)} paper(s) into {args.out_dir} "
        f"filled={total_filled} open={total_open} grounded%={pct:.1f}"
    )


if __name__ == "__main__":
    main()
