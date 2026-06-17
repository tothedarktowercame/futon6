#!/usr/bin/env python3
"""Compose the per-paper anatomy -> proof-check demo.

This is an assembly script. It calls the existing concept coverage logic,
definition-structure transducer, IATC graph renderer utilities, and rung-2
semantic harness rather than reimplementing those checks.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_iatc_goldens as IATC  # noqa: E402
import dp_anatomy_html as R  # noqa: E402
import sfc_concept_coverage as SFC  # noqa: E402

GOLD = ROOT / "data" / "showcases" / "ct-anatomy" / "golden"
RUN = ROOT / "data" / "iatc-argument-graphs" / "loop-run-70b"
OUT_DIR = ROOT / "data" / "showcases" / "ct-anatomy" / "proofcheck-demo"
OUT = OUT_DIR / "index.html"
DEFAULT_PAPERS = ["0706.1286", "0708.2067", "0709.0248", "0708.2185"]
ATTEMPT_GRAPH = RUN / ".attempts" / "0708.2185.attempt2.edn"
FALLBACK_FORMULA = (
    r"\overline{L}=\{x\in L\mid \forall a,b:A\to B\,. "
    r"a|_x=b|_x\Rightarrow a\cdot x\cong b\cdot x\}"
)

FORMULA_RE = re.compile(r"\$([^$]{5,260})\$|\\\((.{5,260}?)\\\)", re.DOTALL)
FORMULA_HINT = re.compile(r"(\\to|\\in|\\mid|\\forall|\\Rightarrow|\\cong|=|\\approx|\\times)")


@dataclass(frozen=True)
class StructurePick:
    concept: str
    formula: str
    output: str
    source: str
    capability_fallback: bool = False


def graph_path_for(pid: str) -> Path:
    if pid == "0708.2185":
        return ATTEMPT_GRAPH
    return RUN / f"{pid}.edn"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def normalize(concept: str) -> str:
    return SFC.normalize_concept(concept)


def lenient_edn(text: str) -> str:
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("'", "")
    return "".join(parts)


def load_graph(path: Path) -> dict[str, Any]:
    import edn_format

    graph = R._edn_to_py(edn_format.loads(lenient_edn(path.read_text(encoding="utf8"))))
    for i, node in enumerate(graph.get("nodes", [])):
        if isinstance(node, dict) and "id" not in node:
            node["id"] = node.get("node-id", i)
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        for key in ("premise", "conclusion", "given", "depends-on"):
            value = edge.get(key)
            if isinstance(value, list):
                edge[key] = [v.get("id", v.get("node-id", str(v))) if isinstance(v, dict) else v
                             for v in value]
            elif isinstance(value, dict):
                edge[key] = value.get("id", value.get("node-id", str(value)))
    return graph


def concept_sources() -> dict[str, set[str]]:
    return SFC.definition_sets(
        load_json(SFC.DEFAULT_SNIPPETS),
        load_json(SFC.DEFAULT_DEFINED),
        load_json(SFC.DEFAULT_ENCYCLOPEDIA),
    )


def paper_concepts(pid: str) -> list[str]:
    usage = load_json(SFC.DEFAULT_USAGE)
    return [normalize(c) for c in usage["paper_concepts"].get(pid, [])]


def concept_panel(pid: str, sources: dict[str, set[str]]) -> str:
    concepts = paper_concepts(pid)
    defined = [c for c in concepts if sources.get(c)]
    rows = []
    for concept in concepts[:80]:
        src = sorted(sources.get(concept, []))
        rows.append(
            "<li>"
            f"<span class='concept-name'>{html.escape(concept)}</span>"
            f"<span class='badge {'ok' if src else 'sorry'}'>{'defined' if src else 'undefined'}</span>"
            f"<span class='src'>{html.escape(', '.join(src) if src else 'no substrate hit')}</span>"
            "</li>"
        )
    extra = ""
    if len(concepts) > 80:
        extra = f"<p class='muted'>Showing 80 of {len(concepts)} imported concepts.</p>"
    return f"""
<section class="panel concepts">
  <h3>1. Imported Concepts + Coverage</h3>
  <div class="metric"><b>{len(defined)}/{len(concepts)}</b><span>defined for this paper</span></div>
  <ul class="concept-list">{''.join(rows)}</ul>
  {extra}
</section>
"""


def snippet_formulas(snippet: str) -> list[str]:
    formulas = []
    for match in FORMULA_RE.finditer(snippet):
        formula = (match.group(1) or match.group(2) or "").strip()
        formula = re.sub(r"\s+", " ", formula)
        if (
            FORMULA_HINT.search(formula)
            and not formula.startswith(("\\begin", ".", ","))
            and "%" not in formula
            and not formula.endswith(("{", "\\"))
        ):
            formulas.append(formula)
    return formulas


def run_structure(formula: str) -> str | None:
    proc = subprocess.run(
        ["bb", "scripts/sfc_def_structure.bb", "-"],
        cwd=ROOT,
        input=formula,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    output = proc.stdout.strip()
    if ":structure" not in output:
        return None
    if ":structure absent" in output or ":structure fragments" in output:
        return None
    return output


def pick_structure(pid: str, snippets: dict[str, Any]) -> StructurePick:
    for concept in paper_concepts(pid)[:80]:
        rows = snippets.get(concept) or []
        for row in rows[:3]:
            for formula in snippet_formulas(row.get("snippet", ""))[:3]:
                output = run_structure(formula)
                if output and ":ungrounded" in output:
                    return StructurePick(
                        concept=concept,
                        formula=formula,
                        output=output,
                        source=f"def-snippets:{row.get('paper', 'unknown')}",
                    )
    output = run_structure(FALLBACK_FORMULA) or ""
    return StructurePick(
        concept="L-closure exemplar",
        formula=FALLBACK_FORMULA,
        output=output,
        source="capability fallback",
        capability_fallback=True,
    )


def structure_panel(pid: str, snippets: dict[str, Any]) -> str:
    pick = pick_structure(pid, snippets)
    label = "capability fallback" if pick.capability_fallback else "paper concept"
    return f"""
<section class="panel structure {'fallback' if pick.capability_fallback else ''}">
  <h3>2. Worked <code>:structure</code></h3>
  <p class="muted">Selected {label}: <b>{html.escape(pick.concept)}</b> · {html.escape(pick.source)}</p>
  <pre class="formula">{html.escape(pick.formula)}</pre>
  <pre class="edn">{html.escape(pick.output)}</pre>
</section>
"""


def semcheck_report(graph: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "semcheck.edn"
        cmd = ["bb", "scripts/iatc_semcheck.bb", "--out", str(out)]
        if ".attempts" in graph.parts:
            cmd.append("--include-attempts")
        cmd.append(str(graph.relative_to(ROOT)))
        subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=30, check=False)
        if not out.exists():
            raise RuntimeError(f"semcheck did not write {out}")
        import edn_format

        return R._edn_to_py(edn_format.loads(out.read_text(encoding="utf8")))


def check_label(check: dict[str, Any]) -> str:
    return {
        "anchor-faithfulness": "R2a anchor-faithfulness",
        "closure": "R2b closure",
        "warrant-resolution": "R2c warrant-resolution",
    }.get(str(check.get("check")), str(check.get("check")))


def reason_text(reason: Any) -> str:
    if isinstance(reason, dict):
        missing = reason.get("missing")
        suffix = f" missing={missing}" if missing else ""
        return f"{reason.get('id', '')} {reason.get('source', '')}: {reason.get('reason', reason)}{suffix}"
    return str(reason)


def verdict_panel(report: dict[str, Any]) -> str:
    graph = (report.get("graphs") or [{}])[0]
    checks = graph.get("checks") or []
    cards = []
    residuals = []
    for check in checks:
        reasons = check.get("reasons") or []
        status = str(check.get("status", "pass"))
        rate = check.get("rate")
        cls = "fail" if status == "fail" else "warn" if reasons else "pass"
        reason_items = "".join(f"<li>{html.escape(reason_text(r))}</li>" for r in reasons[:8])
        if reasons:
            residuals.extend((check_label(check), reason_text(r)) for r in reasons)
        cards.append(
            f"<div class='check {cls}'><b>{html.escape(check_label(check))}</b>"
            f"<span>{html.escape(status.upper())} · rate {rate if rate is not None else 'n/a'}</span>"
            f"<ul>{reason_items}</ul></div>"
        )
    residual_html = "".join(
        f"<li><b>{html.escape(label)}</b>: {html.escape(text)}</li>"
        for label, text in residuals[:12]
    ) or "<li>No residual sorries reported by rung-2.</li>"
    return f"""
<div class="verdict">
  <h4>Rung-2 Verdict</h4>
  <div class="checks">{''.join(cards)}</div>
  <div class="sorries"><b>Residual sorries</b><ul>{residual_html}</ul></div>
</div>
"""


def argument_panel(pid: str, graph_path: Path) -> str:
    source = load_json(GOLD / f"fable-{pid}-dp-emacs.json")
    text = source["text"]
    graph = load_graph(graph_path)
    line_start, line_end, char_start, char_end, window_text = IATC.passage_window(graph, text)
    marks = IATC.rebase_marks(IATC.iatc_to_marks(graph, text), char_start, char_end)
    marked = R.render_marked_source(window_text, marks) if marks else "<p>No IATC marks.</p>"
    report = semcheck_report(graph_path)
    graph_html = render_graph_summary(graph)
    return f"""
<section class="panel argument">
  <h3>3. Argument + Rung-2 Verdict</h3>
  <p class="muted">Graph: <code>{html.escape(str(graph_path.relative_to(ROOT)))}</code> · source lines {line_start}-{line_end}</p>
  {verdict_panel(report)}
  <div class="argument-grid">
    <div><h4>Anchored Source Window</h4><div class="paper">{marked}</div></div>
    <div><h4>IATC Argument Graph</h4>{graph_html}</div>
  </div>
</section>
"""


def render_graph_summary(graph: dict[str, Any]) -> str:
    nodes = {node.get("id"): node for node in graph.get("nodes", [])}

    def node_text(node_id: Any) -> str:
        node = nodes.get(node_id)
        return str(node.get("text", node_id)) if node else str(node_id)

    def ids(value: Any) -> list[Any]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    rows = []
    for edge in graph.get("edges", []):
        premise = ids(edge.get("premise")) + ids(edge.get("given")) + ids(edge.get("depends-on"))
        conclusion = ids(edge.get("conclusion"))
        warrant = edge.get("warrant")
        hole = ""
        if isinstance(warrant, dict) and warrant.get("kind") == "missing-warrant":
            hole = f"<div class='ax-hole'>missing warrant: {html.escape(str(warrant.get('text', warrant.get('wanted', ''))))}</div>"
        rows.append(
            "<div class='ax-edge'>"
            f"<span class='ax-prem'>{html.escape(' · '.join(node_text(x) for x in premise) or '(prior context)')}</span>"
            f"<span class='ax-rel'>{html.escape(str(edge.get('relation', 'infer')))}</span>"
            f"<span class='ax-concl'>{html.escape(' · '.join(node_text(x) for x in conclusion) or '(aside)')}</span>"
            f"{hole}</div>"
        )
    holes = "".join(
        f"<div class='ax-hole'>hole {html.escape(str(h.get('kind', '')))}: wants {html.escape(str(h.get('wanted', '')))}</div>"
        for h in graph.get("holes", [])
    )
    return f"<div class='ax'>{''.join(rows)}{holes}</div>"


def paper_section(pid: str, sources: dict[str, set[str]], snippets: dict[str, Any]) -> str:
    graph = graph_path_for(pid)
    label = {
        "0706.1286": "clean baseline",
        "0708.2067": "orphan-node residual sorries",
        "0709.0248": "anchor-faithfulness proposition flag",
        "0708.2185": "substance-fail self-loop attempt",
    }.get(pid, "demo paper")
    return f"""
<section class="paper-section" id="paper-{html.escape(pid)}">
  <h2>{html.escape(pid)} <span>{html.escape(label)}</span></h2>
  {concept_panel(pid, sources)}
  {structure_panel(pid, snippets)}
  {argument_panel(pid, graph)}
</section>
"""


def build_html(papers: list[str]) -> str:
    sources = concept_sources()
    snippets = load_json(SFC.DEFAULT_SNIPPETS)["snippets"]
    body = "\n".join(paper_section(pid, sources, snippets) for pid in papers)
    nav = " ".join(f"<a href='#paper-{html.escape(pid)}'>{html.escape(pid)}</a>" for pid in papers)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Proof-check demo — anatomy to rung-2</title>
<style>
{R.STYLE}
body{{background:#f7f4ed;color:#1d1a16}}
main{{max-width:1280px}}
.hero{{background:#2f2342;color:#fff;padding:24px 28px;margin:0 -28px 24px;border-bottom:4px solid #d9a441}}
.hero h1{{margin:0 0 8px;font:700 28px/1.15 ui-sans-serif,system-ui,sans-serif}}
.hero p,.hero nav{{font:13px/1.5 ui-sans-serif,system-ui,sans-serif;color:#efe7f6}}
.hero a{{color:#ffd88a;margin-right:12px}}
.paper-section{{background:#fffdf8;border:1px solid #dfd5c5;border-radius:8px;margin:24px 0;padding:18px}}
.paper-section h2{{font:700 22px/1.2 ui-sans-serif,system-ui,sans-serif;margin:0 0 16px;border-bottom:2px solid #e8dfcf;padding-bottom:8px}}
.paper-section h2 span{{font-weight:500;font-size:13px;color:#6a5f4f;margin-left:8px}}
.panel{{border:1px solid #e6dccd;border-radius:8px;background:#fff;margin:14px 0;padding:14px}}
.panel h3{{font:700 15px/1.2 ui-sans-serif,system-ui,sans-serif;margin:0 0 10px;color:#2f2342}}
.panel h4{{font:700 13px/1.2 ui-sans-serif,system-ui,sans-serif;margin:10px 0 8px;color:#443a31}}
.muted,.src{{color:#6a5f4f;font:12px/1.45 ui-sans-serif,system-ui,sans-serif}}
.metric{{display:inline-flex;gap:10px;align-items:baseline;border:1px solid #e9e0d0;border-radius:6px;padding:8px 12px;background:#fff8e8;margin-bottom:10px}}
.metric b{{font:700 24px/1 ui-sans-serif,system-ui,sans-serif;color:#0f766e}}
.metric span{{font:12px ui-sans-serif,system-ui,sans-serif;color:#6a5f4f}}
.concept-list{{columns:2;list-style:none;margin:8px 0;padding:0}}
.concept-list li{{break-inside:avoid;display:flex;gap:7px;align-items:center;margin:0 0 5px;font:12px/1.4 ui-sans-serif,system-ui,sans-serif}}
.concept-name{{min-width:210px;color:#2b2722}}
.badge{{border-radius:999px;padding:1px 7px;font-size:11px;color:#fff}}
.badge.ok{{background:#0f766e}}.badge.sorry{{background:#b45309}}
pre{{white-space:pre-wrap;overflow:auto;border:1px solid #e5dccd;border-radius:6px;padding:10px;background:#fbf8f0;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
.edn{{max-height:260px}}.formula{{background:#fffdf8}}
.verdict{{border-left:4px solid #7c6bd6;background:#faf7ff;padding:10px 12px;margin-bottom:12px}}
.checks{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}}
.check{{border-radius:6px;padding:8px;border:1px solid #ddd;background:#fff;font:12px/1.4 ui-sans-serif,system-ui,sans-serif}}
.check b{{display:block}}.check.pass{{border-color:#9bd1bd}}.check.warn{{border-color:#f2bc72;background:#fff9ed}}.check.fail{{border-color:#ef9a9a;background:#fff1f1}}
.check ul,.sorries ul{{margin:6px 0 0;padding-left:18px}}
.sorries{{margin-top:10px;font:12px/1.45 ui-sans-serif,system-ui,sans-serif;color:#6b2d13;background:#fff3e8;border-radius:6px;padding:8px}}
.argument-grid{{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr);gap:14px;align-items:start}}
.argument .paper{{max-height:560px;overflow:auto}}
.ax{{background:#fffdf8}}
@media (max-width:900px){{.argument-grid,.checks{{grid-template-columns:1fr}}.concept-list{{columns:1}}}}
</style>
<main>
<section class="hero">
  <h1>Per-paper anatomy → proof-check demo</h1>
  <p>Composed from green CPU tools: concept coverage, definition structure transduction, IATC graph marks, and rung-2 semantic checks.</p>
  <nav>{nav}</nav>
</section>
{body}
</main>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="+", default=DEFAULT_PAPERS)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(line.rstrip() for line in build_html(args.papers).splitlines()) + "\n"
    args.out.write_text(rendered, encoding="utf8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
