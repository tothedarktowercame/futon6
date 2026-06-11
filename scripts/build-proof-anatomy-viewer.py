#!/usr/bin/env python3
"""Build a self-contained proof anatomy scope-audit viewer.

Outputs:
  - data/showcases/proof-anatomy/index.html
  - data/showcases/proof-anatomy/problemN.html

The data source is live `proof_scope_audit.run_audit()`; this builder does not
read stale audit JSON from data/.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import proof_scope_audit
import proof_tex_audit

WRITEUP_DIR = Path("/home/joe/code/storage/futon6/data/first-proof")
OUT_DIR = ROOT / "data" / "showcases" / "proof-anatomy"

TYPE_COLORS = {
    # math-proofread-style.sty v0.9 / First Proof monograph palette.
    "greek": "#c92a82",       # Mulberry
    "operator": "#8b008b",    # Purple
    "binop": "#8b008b",
    "bridge": "#2e8b57",      # SeaGreen
    "named-op": "#cc5500",    # BurntOrange
    "dual": "#40e0d0",        # Turquoise
    "relation": "#7851a9",    # RoyalPurple
    "comparison": "#004225",
    "large-op": "#8a2be2",    # BlueViolet
    "large-operator": "#8a2be2",
    "arrow": "#008080",       # TealBlue
    "delimiter": "#ff00ff",   # Magenta
    "function": "#da70d6",    # Orchid
    "number": "#b22222",      # Red
    "math-text": "#704214",   # Sepia
    "math_text": "#704214",
    "math-italic": "#6b8e23", # OliveGreen
    "math_italic": "#6b8e23",
    "named-operator": "#cc5500",
    "variable": "#0a1830",    # MPSyntaxBlueBlack
}

BINDER_CLASSES = {
    "bind/let": "binder-let",
    "bind/define": "binder-define",
    "bind/typed": "binder-define",
    "quant/universal": "binder-quant",
    "assume/explicit": "binder-assume",
    "constrain/relation": "binder-constrain",
    "constrain/such-that": "binder-constrain",
    "constrain/where": "binder-constrain",
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def slug_for_writeup(writeup: str) -> str:
    match = re.search(r"problem(\d+)-writeup\.md$", writeup)
    if match:
        return f"problem{match.group(1)}"
    match = re.search(r"problem(\d+)-solution-full\.tex$", writeup)
    if match:
        return f"problem{match.group(1)}-full"
    return Path(writeup).stem.replace("-writeup", "")


def problem_no(writeup: str) -> int | None:
    match = re.search(r"problem(\d+)", writeup)
    return int(match.group(1)) if match else None


def span_from_scope(scope: dict, text_len: int) -> tuple[int, int] | None:
    content = scope.get("hx/content", {})
    start = content.get("position")
    end = content.get("end")
    if not isinstance(start, int):
        return None
    if not isinstance(end, int) or end <= start:
        end = start + len(str(content.get("match", "")))
    start = max(0, min(text_len, start))
    end = max(start, min(text_len, end))
    return (start, end) if end > start else None


def span_from_expr(expr: dict, text: str) -> tuple[int, int] | None:
    start = expr.get("position")
    if not isinstance(start, int) or start < 0 or start >= len(text):
        return None
    raw = str(expr.get("expr", ""))
    end = min(len(text), start + max(1, len(raw)))
    return (start, end) if end > start else None


def binder_class(hx_type: str) -> str:
    if hx_type in BINDER_CLASSES:
        return BINDER_CLASSES[hx_type]
    if hx_type.startswith("assume/"):
        return "binder-assume"
    if hx_type.startswith("constrain/"):
        return "binder-constrain"
    if hx_type.startswith("quant/"):
        return "binder-quant"
    if hx_type.startswith("bind/"):
        return "binder-let"
    return "binder-other"


def css_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(value).lower()).strip("-") or "unknown"


def annotate_text(text: str, scopes: list[dict], expressions: list[dict]) -> str:
    """Render escaped text with depth-aware scope and expression overlays."""
    scope_items = []
    for scope in scopes:
        span = span_from_scope(scope, len(text))
        if span:
            start, end = span
            scope_items.append({"start": start, "end": end, "scope": scope})

    expr_items = []
    for expr in expressions:
        span = span_from_expr(expr, text)
        if span:
            start, end = span
            expr_items.append({"start": start, "end": end, "expr": expr})

    cuts = {0, len(text)}
    for item in scope_items + expr_items:
        cuts.add(item["start"])
        cuts.add(item["end"])
    points = sorted(cuts)

    pieces = []
    for start, end in zip(points, points[1:]):
        if end <= start:
            continue
        raw = esc(text[start:end])
        active_scopes = [
            item for item in scope_items
            if item["start"] <= start < item["end"]
        ]
        active_scopes.sort(key=lambda item: (item["end"] - item["start"], item["start"]), reverse=True)
        starting_scopes = [
            item for item in active_scopes
            if item["start"] == start
        ]
        active_exprs = [
            item for item in expr_items
            if item["start"] <= start < item["end"]
        ]
        active_exprs.sort(key=lambda item: (item["end"] - item["start"], item["start"]))

        inner = raw
        if active_exprs:
            expr = active_exprs[0]["expr"]
            etype = css_name(expr.get("type", "unknown"))
            grade = css_name(expr.get("grade", "unknown"))
            title = f"{expr.get('expr', '')} · {expr.get('type', '?')} · {expr.get('grade', '?')}"
            inner = (
                f'<span class="expr expr-type-{etype} expr-grade-{grade}" '
                f'title="{esc(title)}">{inner}</span>'
            )

        if active_scopes:
            depth = min(5, len(active_scopes))
            deepest = active_scopes[-1]["scope"]
            hxtype = str(deepest.get("hx/type", "unknown"))
            labels = "".join(
                f'<span class="scope-label {binder_class(str(item["scope"].get("hx/type", "")))}">'
                f'{esc(item["scope"].get("hx/type", "?"))}</span>'
                for item in starting_scopes
            )
            title = " / ".join(str(item["scope"].get("hx/type", "?")) for item in active_scopes)
            inner = (
                f'<span class="scope depth-{depth} {binder_class(hxtype)}" title="{esc(title)}">'
                f"{labels}{inner}</span>"
            )
        pieces.append(inner)
    return "".join(pieces)


def stylesheet(expr_types: set[str]) -> str:
    type_rules = []
    for etype in sorted(expr_types):
        color = TYPE_COLORS.get(etype, TYPE_COLORS["variable"])
        type_rules.append(f".expr-type-{css_name(etype)} {{ --etype-color: {color}; }}")
    return f"""
    :root {{
      --bg: #f4efe8; --paper: #fffdf8; --ink: #1d1a16; --muted: #6b6258;
      --line: #d7cec2; --accent: #0f766e; --danger: #b22222;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; background: linear-gradient(180deg,#f2eadf 0%,var(--bg) 100%); color: var(--ink); font-family: Georgia, "Iowan Old Style", serif; }}
    .wrap {{ max-width: 1280px; margin: 0 auto; }}
    .topnav {{ margin-bottom: 18px; font: 14px/1.4 system-ui, sans-serif; }}
    a {{ color: var(--accent); text-decoration: none; }}
    .hero, .panel, .writeup-box {{ background: var(--paper); border: 1px solid var(--line); border-radius: 16px; padding: 16px; box-shadow: 0 12px 28px rgba(60,42,18,.06); }}
    .hero {{ margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }}
    .stats {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    .stat {{ background: #f7f0e7; border: 1px solid #eadccf; border-radius: 12px; padding: 10px; }}
    .stat .k {{ font: 700 .72rem/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }}
    .stat .v {{ margin-top: 4px; font: 700 1.2rem/1.15 system-ui, sans-serif; }}
    .sub, .tiny, .meta {{ color: var(--muted); }}
    .tiny {{ font: 12px/1.4 system-ui, sans-serif; }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    table {{ width: 100%; border-collapse: collapse; font: 13px/1.35 system-ui, sans-serif; }}
    th, td {{ border-bottom: 1px solid #e7ded2; padding: 8px 7px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
    .writeup-box {{ white-space: pre-wrap; line-height: 1.56; overflow-wrap: anywhere; }}
    .scope {{ border-radius: 3px; padding: 0 1px; }}
    .scope.depth-1 {{ background: linear-gradient(90deg,#fbd38d,#fee2e2); }}
    .scope.depth-2 {{ background: linear-gradient(90deg,#fbcfe8,#fce7f3); }}
    .scope.depth-3 {{ background: linear-gradient(90deg,#ddd6fe,#ede9fe); }}
    .scope.depth-4 {{ background: linear-gradient(90deg,#c7d2fe,#e0e7ff); }}
    .scope.depth-5 {{ background: linear-gradient(90deg,#a5b4fc,#cbd5e1); outline: 1px dashed rgba(71,85,105,.45); outline-offset: -2px; }}
    .scope-label {{ display: inline-block; margin-right: 5px; padding: 0 5px; border-radius: 999px; font: 700 10px/1.35 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .04em; color: white; }}
    .binder-let {{ background-color: #0f766e; }}
    .binder-define {{ background-color: #2563eb; }}
    .binder-quant {{ background-color: #7c3aed; }}
    .binder-assume {{ background-color: #9a3412; }}
    .binder-constrain {{ background-color: #334155; }}
    .binder-other {{ background-color: #6b7280; }}
    .expr {{ color: var(--etype-color, #0a1830); border-radius: 2px; padding: 0 1px; }}
    .expr-grade-strict {{ background: color-mix(in srgb, var(--etype-color, #0a1830) 18%, transparent); }}
    .expr-grade-weak {{ border-bottom: 2px solid var(--etype-color, #0a1830); }}
    .expr-grade-floating {{ border-bottom: 2px dashed var(--danger); background: rgba(178,34,34,.08); }}
    {''.join(type_rules)}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 12px; font: 12px/1.4 system-ui, sans-serif; color: var(--muted); }}
    .legend .swatch {{ display: inline-block; padding: 1px 7px; border-radius: 4px; margin-right: 4px; }}
    .pill {{ display: inline-block; border: 1px solid currentColor; border-radius: 999px; padding: 2px 8px; font: 700 12px/1 system-ui, sans-serif; }}
    ul.compact {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 980px) {{ .grid, .stats {{ grid-template-columns: 1fr; }} }}
    """


def legend_html(expr_types: set[str], grades: set[str], binder_types: set[str]) -> str:
    type_bits = "".join(
        f'<span><span class="swatch expr-type-{css_name(t)}" style="color:{TYPE_COLORS.get(t, TYPE_COLORS["variable"])};border-bottom:2px solid currentColor;">{esc(t)}</span> expression type</span>'
        for t in sorted(expr_types)
    )
    grade_bits = "".join(
        {
            "strict": '<span><span class="swatch expr-grade-strict" style="color:#7851a9;">strict</span> env-contained</span>',
            "weak": '<span><span class="swatch expr-grade-weak" style="color:#008080;">weak</span> paragraph/scope-adjacent</span>',
            "floating": '<span><span class="swatch expr-grade-floating">floating</span> outside binder support</span>',
        }.get(g, f"<span>{esc(g)}</span>")
        for g in sorted(grades)
    )
    binder_bits = "".join(
        f'<span><span class="scope-label {binder_class(t)}">{esc(t)}</span></span>'
        for t in sorted(binder_types)
    )
    depth_bits = "".join(f'<span><span class="swatch scope depth-{d}">d{d}</span> depth {d}</span>' for d in range(1, 6))
    return f'<div class="legend">{depth_bits}{binder_bits}{type_bits}{grade_bits}</div>'


def page_html(result: dict, text: str, all_expr_types: set[str], all_grades: set[str], all_binders: set[str], generated: str) -> str:
    slug = slug_for_writeup(result["writeup"])
    overlay = annotate_text(text, result["scopes"], result["expressions"])
    free = ", ".join(result["free-symbols"][:32])
    if len(result["free-symbols"]) > 32:
        free += f", +{len(result['free-symbols']) - 32} more"
    vacuous = result["vacuous-scopes"][:12]
    expr_counts = Counter(e["type"] for e in result["expressions"])
    grade_counts = Counter(e["grade"] for e in result["expressions"])
    gold_panel = ""
    if result.get("register") == "full-tex":
        examples = result.get("gold-disagreements", [])[:12]
        gold_panel = f"""
<section class="panel"><h2>Gold Diff</h2>
<p><span class="pill">{result.get('gold-agreement-rate', 0.0):.1f}% agreement</span>
 over {result.get('gold-annotated-count', 0)} macro-annotated expressions.</p>
<table><thead><tr><th>Expression</th><th>Gold</th><th>Classifier</th></tr></thead><tbody>
{''.join(f'<tr><td><code>{esc(e.get("expr", ""))}</code></td><td>{esc(", ".join(e.get("gold-types", [])))}</td><td>{esc(e.get("classified-type", ""))}</td></tr>' for e in examples)}
</tbody></table></section>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(result['writeup'])} — Proof Anatomy</title><style>{stylesheet(all_expr_types)}</style></head>
<body><div class="wrap">
<div class="topnav"><a href="index.html">Back to proof anatomy index</a></div>
<section class="hero">
<h1>{esc(result['writeup'])}</h1>
<p class="meta">Generated {esc(generated)} from live <code>proof_scope_audit.run_audit()</code>.</p>
<div class="stats">
<div class="stat"><div class="k">Expressions</div><div class="v">{result['expr-count']}</div></div>
<div class="stat"><div class="k">Scopes</div><div class="v">{result['scope-count']}</div></div>
<div class="stat"><div class="k">Floating</div><div class="v">{result['floating-expr-pct']:.1f}%</div></div>
<div class="stat"><div class="k">Free symbols</div><div class="v">{len(result['free-symbols'])}</div></div>
<div class="stat"><div class="k">Vacuous scopes</div><div class="v">{result['vacuous-count']}</div></div>
</div>
</section>
<section class="panel"><h2>Markup Legend</h2>{legend_html(all_expr_types, all_grades, all_binders)}</section>
<div class="grid">
<section class="panel"><h2>Expression Mix</h2><table><tbody>{''.join(f'<tr><td>{esc(k)}</td><td>{v}</td></tr>' for k, v in sorted(expr_counts.items()))}</tbody></table></section>
<section class="panel"><h2>Grade Mix</h2><table><tbody>{''.join(f'<tr><td>{esc(k)}</td><td>{v}</td></tr>' for k, v in sorted(grade_counts.items()))}</tbody></table></section>
</div>
<div class="grid">
<section class="panel"><h2>Free Symbols</h2><p>{esc(free or 'none')}</p></section>
<section class="panel"><h2>Vacuous Scopes</h2><ul class="compact">{''.join(f'<li><code>{esc(v.get("type"))}</code> {esc(v.get("match", ""))}</li>' for v in vacuous) or '<li>none</li>'}</ul></section>
</div>
{gold_panel}
<section class="writeup-box" id="{esc(slug)}">{overlay}</section>
</div></body></html>"""


def index_html(summary_results: list[dict], full_results: list[dict], all_expr_types: set[str], all_grades: set[str], all_binders: set[str], generated: str) -> str:
    summary = proof_scope_audit.summarize(summary_results)
    full_summary = proof_tex_audit.summarize(full_results)
    full_by_problem = {problem_no(r["writeup"]): r for r in full_results}
    rows = []
    for r in summary_results:
        slug = slug_for_writeup(r["writeup"])
        full = full_by_problem.get(problem_no(r["writeup"]))
        if full:
            full_slug = slug_for_writeup(full["writeup"])
            full_cells = (
                f"<td><a href=\"{full_slug}.html\">{full['expr-count']}</a></td>"
                f"<td>{full['scope-count']}</td>"
                f"<td>{full['floating-expr-pct']:.1f}%</td>"
                f"<td>{full.get('gold-agreement-rate', 0.0):.1f}%</td>"
            )
        else:
            full_cells = "<td></td><td></td><td></td><td></td>"
        # The full proof gets a NAMED link beside the writeup — hiding it
        # as a bare count made the index read "writeups only" (Joe).
        title_cell = (
            f"<a href=\"{slug}.html\">{esc(r['writeup'])}</a>"
            + (f" &middot; <a href=\"{slug_for_writeup(full['writeup'])}.html\">"
               f"<b>full proof</b></a>" if full else "")
        )
        rows.append(
            f"<tr><td>{title_cell}</td>"
            f"<td>{r['expr-count']}</td><td>{r['scope-count']}</td>"
            f"<td>{r['floating-expr-pct']:.1f}%</td><td>{len(r['free-symbols'])}</td>"
            f"<td>{r['vacuous-count']}</td>{full_cells}</tr>"
        )
    baseline = summary["nlab-baseline"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Proof Anatomy Viewer</title><style>{stylesheet(all_expr_types)}</style></head>
<body><div class="wrap">
<section class="hero"><h1>First Proof Anatomy Viewer</h1>
<p class="meta">Generated {esc(generated)} from live <code>proof_scope_audit.run_audit()</code>; output is local and self-contained.</p>
<div class="stats">
<div class="stat"><div class="k">Writeups</div><div class="v">{summary['writeups']}</div></div>
<div class="stat"><div class="k">Expressions</div><div class="v">{summary['expr-total']}</div></div>
<div class="stat"><div class="k">Scopes</div><div class="v">{summary['scope-total']}</div></div>
<div class="stat"><div class="k">Floating</div><div class="v">{summary['floating-expr-pct']:.1f}%</div></div>
<div class="stat"><div class="k">Vacuous</div><div class="v">{summary['vacuous-scope-count']}</div></div>
</div></section>
<section class="panel"><h2>Markup Legend</h2>{legend_html(all_expr_types, all_grades, all_binders)}</section>
<section class="panel"><h2>Corpus Table</h2><table><thead><tr><th>Writeup</th><th>Summary Expr</th><th>Summary Scopes</th><th>Summary Floating</th><th>Free Symbols</th><th>Vacuous</th><th>Full Expr</th><th>Full Scopes</th><th>Full Floating</th><th>Gold Agree</th></tr></thead><tbody>
{''.join(rows)}
<tr><td><strong>Total</strong></td><td>{summary['expr-total']}</td><td>{summary['scope-total']}</td><td>{summary['floating-expr-pct']:.1f}%</td><td></td><td>{summary['vacuous-scope-count']}</td><td>{full_summary['expr-total']}</td><td>{full_summary['scope-total']}</td><td>{full_summary['floating-expr-pct']:.1f}%</td><td>{full_summary['gold-agreement-rate']:.1f}%</td></tr>
<tr><td><strong>nLab baseline</strong></td><td colspan="3">{baseline['floating-expr-pct']}% floating expressions</td><td colspan="6">{baseline['vacuous-envs']['vacuous']}/{baseline['vacuous-envs']['envs']} vacuous envs</td></tr>
</tbody></table></section>
</div></body></html>"""


def build(out_dir: Path = OUT_DIR, writeup_dir: Path = WRITEUP_DIR) -> dict:
    results = proof_scope_audit.run_audit(writeup_dir)
    full_results = proof_tex_audit.run_audit()
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = results + full_results
    all_expr_types = {e["type"] for r in all_results for e in r["expressions"]}
    all_expr_types.update(
        t
        for r in full_results
        for e in r["expressions"]
        for t in e.get("gold-types", [])
    )
    all_grades = {e["grade"] for r in all_results for e in r["expressions"]}
    all_binders = {str(s.get("hx/type", "unknown")) for r in all_results for s in r["scopes"]}

    for result in results:
        path = writeup_dir / result["writeup"]
        text = path.read_text(encoding="utf-8", errors="ignore")
        slug = slug_for_writeup(result["writeup"])
        (out_dir / f"{slug}.html").write_text(
            page_html(result, text, all_expr_types, all_grades, all_binders, generated),
            encoding="utf-8",
        )
    for result in full_results:
        path = proof_tex_audit.FULL_TEX_DIR / result["writeup"]
        text = path.read_text(encoding="utf-8", errors="ignore")
        slug = slug_for_writeup(result["writeup"])
        (out_dir / f"{slug}.html").write_text(
            page_html(result, text, all_expr_types, all_grades, all_binders, generated),
            encoding="utf-8",
        )
    (out_dir / "index.html").write_text(
        index_html(results, full_results, all_expr_types, all_grades, all_binders, generated),
        encoding="utf-8",
    )
    return {"out-dir": str(out_dir), "pages": len(results) + len(full_results) + 1, "results": results, "full-results": full_results}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--writeup-dir", type=Path, default=WRITEUP_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report = build(args.out_dir, args.writeup_dir)
    print(f"wrote {report['pages']} pages to {report['out-dir']}")


if __name__ == "__main__":
    main()
