#!/usr/bin/env python3
"""Build scope delta + two-up LaTeX/s-expression demo for math.SE thread 633512.

Outputs:
  data/first-proof/nnexus-glasses-demo-scope-delta-two-up.html
  data/first-proof/nnexus-glasses-demo-scope-delta-two-up-validation.json

The original demo files are left untouched:
  data/first-proof/nnexus-glasses-demo.html
  data/first-proof/nnexus-glasses-demo-scope-delta.html
"""

from __future__ import annotations

import html
import importlib
import json
import re
import tarfile
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data/first-proof/thread-633512-raw.json"
WIRING_PATH = ROOT / "data/first-proof/thread-633512-wiring.json"
OUT_PATH = ROOT / "data/first-proof/nnexus-glasses-demo-scope-delta-two-up.html"
REPORT_PATH = ROOT / "data/first-proof/nnexus-glasses-demo-scope-delta-two-up-validation.json"
EPRINT_TAR = ROOT / "data/arxiv-math-ct-eprints/0705.0462.tar.gz"
EPRINT_MEMBER = "resource_modalities.tex"

GREEK_OR_SYMBOL_COMMANDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta",
    "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau",
    "upsilon", "phi", "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
}


def strip_html_to_text(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = " ".join(s.split())
    return s


def norm_match(s: str) -> str:
    return " ".join((s or "").split())


def scope_key(scope: dict) -> tuple:
    c = scope.get("hx/content", {})
    return (
        scope.get("hx/type", ""),
        int(c.get("position", -1)),
        norm_match(c.get("match", "")),
    )


def get_legacy_scopes(wiring: dict, answer_id: str) -> list[dict]:
    for node in wiring.get("nodes", []):
        if node.get("id") != answer_id:
            continue
        return [d for d in node.get("discourse", []) if d.get("hx/role") == "component"]
    return []


def find_paper_snippet(detect_scopes) -> dict:
    if not EPRINT_TAR.exists():
        return {"error": f"missing {EPRINT_TAR}"}

    try:
        with tarfile.open(EPRINT_TAR, "r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name == EPRINT_MEMBER), None)
            if member is None:
                return {"error": f"missing {EPRINT_MEMBER} in {EPRINT_TAR.name}"}
            fh = tf.extractfile(member)
            if fh is None:
                return {"error": f"cannot extract {EPRINT_MEMBER}"}
            text = fh.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return {"error": str(exc)}

    env_pat = re.compile(r"\\begin\{(theorem|lemma|proposition|corollary|proof|definition|defn)\}", re.I)
    binder_pat = re.compile(r"\\(forall|exists|sum|prod|int|coprod|bigcup|bigcap)\b")
    em = env_pat.search(text)
    bm = binder_pat.search(text)
    if not em or not bm:
        return {"error": "no env+binder snippet in selected member"}

    start = max(0, min(em.start(), bm.start()) - 220)
    end = min(len(text), max(em.end(), bm.end()) + 260)
    snippet = " ".join(text[start:end].split())

    scopes = detect_scopes("arxiv:0705.0462", snippet)
    scope_types = Counter(s.get("hx/type", "?") for s in scopes)

    return {
        "arxiv_id": "0705.0462",
        "title": "Resource modalities in game semantics",
        "url": "https://arxiv.org/abs/0705.0462",
        "snippet": snippet[:700],
        "detected_env": em.group(1),
        "detected_binder": bm.group(1),
        "scope_type_top": scope_types.most_common(10),
    }


def extract_math_expressions(text: str) -> list[dict]:
    """Extract $...$ and $$...$$ with positions from plain text."""
    out = []
    display_ranges = []

    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        tex = m.group(1).strip()
        if not tex:
            continue
        display_ranges.append((m.start(), m.end()))
        out.append({
            "latex": tex,
            "display": True,
            "position": m.start(),
            "open_len": 2,
        })

    for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", text):
        if any(s <= m.start() < e for s, e in display_ranges):
            continue
        tex = m.group(1).strip()
        if not tex:
            continue
        out.append({
            "latex": tex,
            "display": False,
            "position": m.start(),
            "open_len": 1,
        })

    out.sort(key=lambda r: r["position"])
    return out


def _overlaps(start: int, end: int, spans: list[dict]) -> bool:
    return any(not (end <= s["start"] or start >= s["end"]) for s in spans)


def _overlaps_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(not (end <= a or start >= b) for a, b in ranges)


def _find_nonsemantic_exclusion_ranges(tex: str) -> list[tuple[int, int]]:
    """Ranges in TeX that should not count as variable tokens.

    Example: in ``\begin{array}{c}``, the ``c`` is a layout column-spec, not a
    mathematical variable.
    """
    ranges: list[tuple[int, int]] = []

    # Common matrix/table environments with an alignment/column spec argument.
    for m in re.finditer(r"\\begin\{(?:array|tabular|tabularx)\}\{([^}]*)\}", tex):
        spec = m.group(1)
        # Ignore short layout specs; skip arbitrary long/complex arguments.
        if len(spec) <= 48:
            ranges.append((m.start(1), m.end(1)))

    return ranges


def find_variable_spans(tex: str) -> list[dict]:
    r"""Heuristic variable tokens in a LaTeX expression.

    Captures:
      - \mathcal{X}
      - Greek/symbol command variables (\Gamma, \beta, ...)
      - Single-letter symbols (x, V, e, ...)
    """
    spans = []
    excluded_ranges = _find_nonsemantic_exclusion_ranges(tex)

    for m in re.finditer(r"\\mathcal\{([A-Za-z])\}", tex):
        spans.append({
            "token": f"\\mathcal{{{m.group(1)}}}",
            "start": m.start(),
            "end": m.end(),
        })

    for m in re.finditer(r"\\([A-Za-z]+)", tex):
        cmd = m.group(1)
        if cmd not in GREEK_OR_SYMBOL_COMMANDS:
            continue
        if _overlaps_ranges(m.start(), m.end(), excluded_ranges):
            continue
        if _overlaps(m.start(), m.end(), spans):
            continue
        spans.append({
            "token": f"\\{cmd}",
            "start": m.start(),
            "end": m.end(),
        })

    for m in re.finditer(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])", tex):
        s, e = m.start(1), m.end(1)
        if _overlaps_ranges(s, e, excluded_ranges):
            continue
        if _overlaps(s, e, spans):
            continue
        spans.append({
            "token": m.group(1),
            "start": s,
            "end": e,
        })

    spans.sort(key=lambda d: (d["start"], d["end"]))
    return spans


def build_scope_ranges(scopes: list[dict]) -> list[tuple[int, int]]:
    ranges = []
    for s in scopes:
        c = s.get("hx/content", {})
        pos = c.get("position")
        if not isinstance(pos, int):
            continue
        end = c.get("end")
        if not isinstance(end, int):
            match = c.get("match", "")
            if not match:
                continue
            end = pos + len(match)
        if end > pos:
            ranges.append((pos, end))
    ranges.sort()
    return ranges


def is_pos_in_any_scope(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in ranges)


def build() -> None:
    if not RAW_PATH.exists() or not WIRING_PATH.exists():
        raise FileNotFoundError("required thread fixture files are missing")

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    nw = importlib.import_module("nlab-wiring")
    sexp_mod = importlib.import_module("futon6.latex_sexp")

    raw = json.loads(RAW_PATH.read_text())
    wiring = json.loads(WIRING_PATH.read_text())

    q_text = strip_html_to_text(raw["question"]["body_html"])
    answer = raw["answers"][0]
    answer_id = f"a-{answer['id']}"
    a_text = strip_html_to_text(answer["body_html"])

    legacy_scopes = get_legacy_scopes(wiring, answer_id)
    new_scopes = nw.detect_scopes(answer_id, a_text)

    legacy_keys = {scope_key(s) for s in legacy_scopes}
    new_rows = []
    for s in new_scopes:
        row = dict(s)
        row["delta_status"] = "common" if scope_key(s) in legacy_keys else "added"
        new_rows.append(row)

    legacy_type_counts = Counter(s.get("hx/type", "?") for s in legacy_scopes)
    new_type_counts = Counter(s.get("hx/type", "?") for s in new_rows)

    added = [s for s in new_rows if s["delta_status"] == "added"]
    added_sorted = sorted(
        added,
        key=lambda s: (s.get("hx/content", {}).get("position", 10**9), s.get("hx/type", "")),
    )

    # Build two-up expression mapping with scope-coverage annotations.
    scope_ranges = build_scope_ranges(new_rows)
    expr_rows = []
    total_vars = 0
    scoped_vars = 0
    expr_with_unscoped = 0
    unscoped_token_counts = Counter()
    unscoped_expr_rows = []

    for i, expr in enumerate(extract_math_expressions(a_text), start=1):
        latex = expr["latex"]
        sexp = sexp_mod.parse(latex)
        var_spans = find_variable_spans(latex)

        enriched = []
        for span in var_spans:
            global_pos = expr["position"] + expr["open_len"] + span["start"]
            scoped = is_pos_in_any_scope(global_pos, scope_ranges)
            enriched.append({
                "token": span["token"],
                "start": span["start"],
                "end": span["end"],
                "global_pos": global_pos,
                "scoped": scoped,
            })

        n_vars = len(enriched)
        n_scoped = sum(1 for v in enriched if v["scoped"])
        n_unscoped = n_vars - n_scoped

        total_vars += n_vars
        scoped_vars += n_scoped
        if n_unscoped > 0:
            expr_with_unscoped += 1
            unscoped_token_counts.update(v["token"] for v in enriched if not v["scoped"])
            unscoped_expr_rows.append({
                "index": i,
                "position": expr["position"],
                "latex": latex,
                "unscoped_count": n_unscoped,
                "unscoped_tokens": sorted({v["token"] for v in enriched if not v["scoped"]}),
            })

        expr_rows.append({
            "index": i,
            "position": expr["position"],
            "display": expr["display"],
            "latex": latex,
            "sexp": sexp,
            "parse_fallback": bool(sexp.startswith('"') and sexp.endswith('"')),
            "vars": enriched,
            "var_count": n_vars,
            "scoped_count": n_scoped,
            "unscoped_count": n_unscoped,
            "unscoped_tokens": sorted({v["token"] for v in enriched if not v["scoped"]}),
        })

    paper = find_paper_snippet(nw.detect_scopes)
    coverage = {
        "expr_count": len(expr_rows),
        "expr_with_unscoped": expr_with_unscoped,
        "total_vars": total_vars,
        "scoped_vars": scoped_vars,
        "unscoped_vars": total_vars - scoped_vars,
        "pct_scoped": round((100.0 * scoped_vars / total_vars), 1) if total_vars else 100.0,
        "all_scoped": (scoped_vars == total_vars),
    }

    payload = {
        "thread": {
            "id": raw["question"]["id"],
            "url": raw["question"]["url"],
            "title": raw["question"]["title"],
            "question_text": q_text,
            "answer_id": answer_id,
            "answer_text": a_text,
        },
        "legacy_scopes": legacy_scopes,
        "new_scopes": new_rows,
        "added_scopes": added_sorted,
        "expr_rows": expr_rows,
        "scope_coverage": coverage,
        "stats": {
            "legacy_count": len(legacy_scopes),
            "new_count": len(new_rows),
            "added_count": len(added_sorted),
            "legacy_type_counts": legacy_type_counts,
            "new_type_counts": new_type_counts,
        },
        "paper_scope_case": paper,
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": raw["question"]["id"],
        "answer_id": answer_id,
        "html_output": str(OUT_PATH.relative_to(ROOT)),
        "scope_coverage": coverage,
        "unscoped_token_counts": dict(unscoped_token_counts.most_common()),
        "expressions_with_unscoped": unscoped_expr_rows,
        "notes": [
            "Goal check is strict: every variable token occurrence must fall inside at least one detected scope span.",
            "Variable extraction ignores non-semantic layout spans such as array/tabular column specs.",
        ],
    }

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>NNexus Glasses Scope Delta + Two-Up Demo</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; background: #f7f7f4; color: #1f2937; }}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px 16px 56px; }}
h1 {{ margin: 0 0 4px; font-size: 1.45rem; }}
.sub {{ color: #6b7280; font-size: 0.9rem; margin-bottom: 18px; }}
.note {{ background: #e8eefc; border: 1px solid #bfdbfe; padding: 12px 14px; border-radius: 8px; font-family: system-ui, sans-serif; font-size: 0.86rem; line-height: 1.45; margin-bottom: 16px; }}
.stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 16px; }}
.stat {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 10px 12px; }}
.stat .k {{ font: 700 0.72rem/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .06em; color: #64748b; }}
.stat .v {{ font: 700 1.25rem/1.15 system-ui, sans-serif; color: #111827; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.card {{ background: #fff; border: 1px solid #d1d5db; border-radius: 10px; padding: 14px 14px 16px; }}
.card h2 {{ margin: 0 0 8px; font-size: 1rem; font-family: system-ui, sans-serif; }}
.pill {{ display: inline-block; font: 700 0.65rem/1 system-ui, sans-serif; text-transform: uppercase; letter-spacing: .05em; border-radius: 999px; padding: 4px 8px; margin-right: 6px; }}
.pill.legacy {{ background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }}
.pill.common {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
.pill.added {{ background: #ffedd5; color: #9a3412; border: 1px solid #fdba74; }}
.pill.warn {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
.textbox {{ border: 1px solid #e5e7eb; border-radius: 8px; background: #fcfcfb; padding: 12px; line-height: 1.5; max-height: 420px; overflow: auto; white-space: pre-wrap; }}
.scope-legacy {{ background: rgba(59,130,246,.13); border-bottom: 2px solid #3b82f6; border-radius: 3px; padding: 0 2px; }}
.scope-common {{ background: rgba(16,185,129,.14); border-bottom: 2px solid #10b981; border-radius: 3px; padding: 0 2px; }}
.scope-added {{ background: rgba(249,115,22,.14); border-bottom: 2px solid #f97316; border-radius: 3px; padding: 0 2px; }}
.table {{ width: 100%; border-collapse: collapse; font-family: system-ui, sans-serif; font-size: 0.82rem; }}
.table th, .table td {{ border-bottom: 1px solid #e5e7eb; padding: 7px 6px; text-align: left; vertical-align: top; }}
.table th {{ color: #475569; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .04em; }}
.table code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
.section {{ margin-top: 16px; }}
.mini {{ font-family: system-ui, sans-serif; font-size: 0.82rem; color: #475569; }}
pre {{ margin: 8px 0 0; background: #0f172a; color: #e2e8f0; border-radius: 8px; padding: 10px 12px; overflow: auto; font-size: 0.78rem; }}
.latex-box {{ margin-top: 6px; background: #0f172a; color: #e2e8f0; border-radius: 8px; padding: 10px 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.78rem; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }}
.sexp-box {{ margin-top: 6px; background: #111827; color: #e5e7eb; border-radius: 8px; padding: 10px 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.78rem; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }}
.var-scoped {{ background: rgba(34,197,94,.28); border-bottom: 2px solid #16a34a; border-radius: 3px; padding: 0 1px; color: #dcfce7; }}
.var-unscoped {{ background: rgba(239,68,68,.28); border-bottom: 2px solid #ef4444; border-radius: 3px; padding: 0 1px; color: #fee2e2; }}
.row-meta {{ margin-top: 4px; font-size: 0.74rem; color: #64748b; font-family: system-ui, sans-serif; }}
.twoup td {{ width: 50%; }}
@media (max-width: 980px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .stats {{ grid-template-columns: 1fr 1fr; }}
  .twoup td {{ width: auto; display: block; }}
}}
</style>
</head>
<body>
<div class=\"wrap\">
  <h1>NNexus Glasses Scope Delta + Two-Up Demo</h1>
  <p class=\"sub\">New file: <code>data/first-proof/nnexus-glasses-demo-scope-delta-two-up.html</code>. Existing demos unchanged.</p>

  <div class=\"note\">
    This page compares scope treatment on the same thread (<a href=\"{html.escape(raw['question']['url'])}\" target=\"_blank\">math.SE #{raw['question']['id']}</a>) and adds a two-up expression view:
    <strong>(A)</strong> marked-up LaTeX variable tokens by scope coverage, <strong>(B)</strong> parsed s-expression mapping.
  </div>

  <div class=\"stats\" id=\"stats\"></div>

  <div class=\"grid\">
    <div class=\"card\">
      <h2>Legacy Scope Overlay <span class=\"pill legacy\">Legacy</span></h2>
      <div class=\"mini\">Scopes in original wiring snapshot.</div>
      <div class=\"textbox\" id=\"legacy-text\"></div>
    </div>

    <div class=\"card\">
      <h2>New Scope Overlay <span class=\"pill common\">Common</span><span class=\"pill added\">Added</span></h2>
      <div class=\"mini\">Updated detector includes metatheory mapping, symbolic binders, and theorem-like environments.</div>
      <div class=\"textbox\" id=\"new-text\"></div>
    </div>
  </div>

  <div class=\"section card\">
    <h2>Two-Up Expressions: Marked-up LaTeX vs s-expression</h2>
    <div class=\"mini\" id=\"coverage-line\"></div>
    <table class=\"table twoup\" id=\"expr-table\">
      <thead><tr><th>A. Marked-up LaTeX</th><th>B. s-expression mapping</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class=\"section card\">
    <h2>Added Scopes In This Thread</h2>
    <table class=\"table\" id=\"added-table\">
      <thead><tr><th>Type</th><th>Match</th><th>Pos</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class=\"section card\">
    <h2>Type Count Delta</h2>
    <table class=\"table\" id=\"type-table\">
      <thead><tr><th>Type</th><th>Legacy</th><th>New</th><th>Delta</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class=\"section card\">
    <h2>Paper TeX Scope Case (math.CT)</h2>
    <div class=\"mini\" id=\"paper-meta\"></div>
    <pre id=\"paper-snippet\"></pre>
  </div>
</div>

<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};

function esc(s) {{
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}}

function renderHighlighted(text, scopes, clsFn) {{
  const arr = [...scopes].sort((a,b) => {{
    const pa = Number(a?.['hx/content']?.position ?? 1e12);
    const pb = Number(b?.['hx/content']?.position ?? 1e12);
    if (pa !== pb) return pa - pb;
    const la = String(a?.['hx/content']?.match ?? '').length;
    const lb = String(b?.['hx/content']?.match ?? '').length;
    return lb - la;
  }});

  let out = '';
  let cur = 0;
  for (const s of arr) {{
    const c = s['hx/content'] || {{}};
    const pos = Number(c.position ?? -1);
    const m = String(c.match ?? '');
    if (pos < 0 || !m) continue;
    if (pos < cur) continue;
    const end = Math.min(text.length, pos + m.length);
    if (pos > text.length || end <= pos) continue;
    out += esc(text.slice(cur, pos));
    const cls = clsFn(s);
    const tip = `${{s['hx/type'] || '?'}} @${{pos}}`;
    out += `<span class=\"${{cls}}\" title=\"${{esc(tip)}}\">${{esc(text.slice(pos, end))}}</span>`;
    cur = end;
  }}
  out += esc(text.slice(cur));
  return out;
}}

function annotateLatex(latex, vars) {{
  const arr = [...(vars || [])].sort((a,b) => Number(a.start) - Number(b.start));
  let cur = 0;
  let out = '';
  for (const v of arr) {{
    const s = Number(v.start);
    const e = Number(v.end);
    if (s < cur || e <= s) continue;
    out += esc(latex.slice(cur, s));
    const cls = v.scoped ? 'var-scoped' : 'var-unscoped';
    const tip = `${{v.token}} @${{v.global_pos}} ${{v.scoped ? 'scoped' : 'unscoped'}}`;
    out += `<span class=\"${{cls}}\" title=\"${{esc(tip)}}\">${{esc(latex.slice(s, e))}}</span>`;
    cur = e;
  }}
  out += esc(latex.slice(cur));
  return out;
}}

function buildStats() {{
  const s = DATA.stats;
  const c = DATA.scope_coverage || {{}};
  const cards = [
    ['Legacy scopes', s.legacy_count],
    ['New scopes', s.new_count],
    ['Added scopes', s.added_count],
    ['Variable tokens', c.total_vars ?? 0],
    ['Scoped tokens', c.scoped_vars ?? 0],
    ['Unscoped tokens', c.unscoped_vars ?? 0],
    ['Scoped %', `${{c.pct_scoped ?? 0}}%`],
    ['Exprs w/ unscoped', c.expr_with_unscoped ?? 0],
  ];
  document.getElementById('stats').innerHTML = cards.map(([k,v]) =>
    `<div class=\"stat\"><div class=\"k\">${{esc(k)}}</div><div class=\"v\">${{esc(v)}}</div></div>`
  ).join('');
}}

function buildTexts() {{
  const text = DATA.thread.answer_text;
  const legacy = DATA.legacy_scopes;
  const news = DATA.new_scopes;
  document.getElementById('legacy-text').innerHTML = renderHighlighted(text, legacy, () => 'scope-legacy');
  document.getElementById('new-text').innerHTML = renderHighlighted(text, news, (s) =>
    s.delta_status === 'added' ? 'scope-added' : 'scope-common'
  );
}}

function buildExprTable() {{
  const c = DATA.scope_coverage || {{}};
  const rows = DATA.expr_rows || [];
  const line = document.getElementById('coverage-line');
  const goalState = c.all_scoped ? 'goal met' : 'goal not met';
  line.innerHTML =
    `Goal: every variable token is in >=1 scope. ` +
    `<strong>${{esc(goalState)}}</strong>. ` +
    `Current: <strong>${{Number(c.scoped_vars || 0)}}/${{Number(c.total_vars || 0)}} (${{esc(c.pct_scoped || 0)}}%)</strong> scoped, ` +
    `<strong>${{Number(c.unscoped_vars || 0)}}</strong> unscoped.`;

  const body = document.querySelector('#expr-table tbody');
  if (!rows.length) {{
    body.innerHTML = '<tr><td colspan="2">No math expressions found.</td></tr>';
    return;
  }}

  body.innerHTML = rows.map((r) => {{
    const status = Number(r.unscoped_count || 0) === 0
      ? '<span class="pill common">fully scoped</span>'
      : '<span class="pill warn">needs scope</span>';
    const unscoped = (r.unscoped_tokens || []).length
      ? `unscoped: <code>${{esc((r.unscoped_tokens || []).join(', '))}}</code>`
      : 'unscoped: none';
    const parseFlag = r.parse_fallback
      ? ' <span class="pill added">fallback parse</span>'
      : '';
    return `
      <tr>
        <td>
          <div>${{status}} <span class="mini">expr #${{r.index}} @${{r.position}}</span></div>
          <div class="latex-box">${{annotateLatex(String(r.latex || ''), r.vars || [])}}</div>
          <div class="row-meta">vars: ${{r.var_count}} | scoped: ${{r.scoped_count}} | ${{unscoped}}</div>
        </td>
        <td>
          <div class="mini">s-expression${{parseFlag}}</div>
          <div class="sexp-box">${{esc(String(r.sexp || ''))}}</div>
        </td>
      </tr>`;
  }}).join('');
}}

function buildAddedTable() {{
  const body = document.querySelector('#added-table tbody');
  const rows = DATA.added_scopes || [];
  if (!rows.length) {{
    body.innerHTML = '<tr><td colspan="3">No added scopes in this thread.</td></tr>';
    return;
  }}
  body.innerHTML = rows.map((s) => {{
    const c = s['hx/content'] || {{}};
    return `<tr><td><code>${{esc(s['hx/type'] || '?')}}</code></td><td>${{esc(c.match || '')}}</td><td>${{Number(c.position ?? -1)}}</td></tr>`;
  }}).join('');
}}

function buildTypeDelta() {{
  const legacy = DATA.stats.legacy_type_counts || {{}};
  const now = DATA.stats.new_type_counts || {{}};
  const keys = Array.from(new Set([...Object.keys(legacy), ...Object.keys(now)])).sort();
  const body = document.querySelector('#type-table tbody');
  body.innerHTML = keys.map((k) => {{
    const a = Number(legacy[k] || 0);
    const b = Number(now[k] || 0);
    const d = b - a;
    const ds = d > 0 ? `+${{d}}` : `${{d}}`;
    return `<tr><td><code>${{esc(k)}}</code></td><td>${{a}}</td><td>${{b}}</td><td>${{ds}}</td></tr>`;
  }}).join('');
}}

function buildPaperCase() {{
  const p = DATA.paper_scope_case || {{}};
  const meta = document.getElementById('paper-meta');
  const pre = document.getElementById('paper-snippet');
  if (p.error) {{
    meta.textContent = p.error;
    pre.textContent = '';
    return;
  }}
  const scopeTop = (p.scope_type_top || []).map(([t,n]) => `${{t}}:${{n}}`).join(', ');
  meta.innerHTML =
    `arXiv <code>${{esc(p.arxiv_id || '')}}</code> - ` +
    `<a href=\"${{esc(p.url || '#')}}\" target=\"_blank\">${{esc(p.title || '')}}</a> - ` +
    `env=<code>${{esc(p.detected_env || '')}}</code>, binder=<code>${{esc(p.detected_binder || '')}}</code><br>` +
    `Detected scope types: ${{esc(scopeTop)}}`;
  pre.textContent = p.snippet || '';
}}

buildStats();
buildTexts();
buildExprTable();
buildAddedTable();
buildTypeDelta();
buildPaperCase();
</script>
</body>
</html>
"""

    OUT_PATH.write_text(html_doc, encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[demo] wrote {OUT_PATH}")
    print(f"[demo] wrote {REPORT_PATH}")


if __name__ == "__main__":
    build()
