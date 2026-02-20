#!/usr/bin/env python3
"""Build a side-by-side scope delta demo for math.SE thread 633512.

Outputs:
  data/first-proof/nnexus-glasses-demo-scope-delta.html

The original demo file is left untouched.
"""

from __future__ import annotations

import html
import importlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data/first-proof/thread-633512-raw.json"
WIRING_PATH = ROOT / "data/first-proof/thread-633512-wiring.json"
OUT_PATH = ROOT / "data/first-proof/nnexus-glasses-demo-scope-delta.html"
EPRINT_TAR = ROOT / "data/arxiv-math-ct-eprints/0705.0462.tar.gz"
EPRINT_MEMBER = "resource_modalities.tex"


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


def build() -> None:
    if not RAW_PATH.exists() or not WIRING_PATH.exists():
        raise FileNotFoundError("required thread fixture files are missing")

    sys.path.insert(0, str(ROOT / "scripts"))
    nw = importlib.import_module("nlab-wiring")

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

    paper = find_paper_snippet(nw.detect_scopes)

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
        "stats": {
            "legacy_count": len(legacy_scopes),
            "new_count": len(new_rows),
            "added_count": len(added_sorted),
            "legacy_type_counts": legacy_type_counts,
            "new_type_counts": new_type_counts,
        },
        "paper_scope_case": paper,
    }

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>NNexus Glasses Scope Delta Demo</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; background: #f7f7f4; color: #1f2937; }}
.wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px 56px; }}
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
@media (max-width: 980px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .stats {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
</head>
<body>
<div class=\"wrap\">
  <h1>NNexus Glasses Scope Delta Demo</h1>
  <p class=\"sub\">New file, original demo unchanged: <code>data/first-proof/nnexus-glasses-demo.html</code></p>

  <div class=\"note\">
    This page compares scope treatment on the same thread (<a href=\"{html.escape(raw['question']['url'])}\" target=\"_blank\">math.SE #{raw['question']['id']}</a>):
    legacy scopes from the original wiring snapshot vs new scopes from the updated detector (metatheory typing + extra scope forms).
  </div>

  <div class=\"stats\" id=\"stats\"></div>

  <div class=\"grid\">
    <div class=\"card\">
      <h2>Legacy Scope Overlay <span class=\"pill legacy\">Legacy</span></h2>
      <div class=\"mini\">Scopes in original wiring: <code>bind/let</code>, <code>quant/universal</code>, <code>constrain/where</code></div>
      <div class=\"textbox\" id=\"legacy-text\"></div>
    </div>

    <div class=\"card\">
      <h2>New Scope Overlay <span class=\"pill common\">Common</span><span class=\"pill added\">Added</span></h2>
      <div class=\"mini\">New detector adds scope forms like <code>constrain/such-that</code> and symbolic binders/environments where present.</div>
      <div class=\"textbox\" id=\"new-text\"></div>
    </div>
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
    out += `<span class="${{cls}}" title="${{esc(tip)}}">${{esc(text.slice(pos, end))}}</span>`;
    cur = end;
  }}
  out += esc(text.slice(cur));
  return out;
}}

function buildStats() {{
  const s = DATA.stats;
  const cards = [
    ['Legacy scopes', s.legacy_count],
    ['New scopes', s.new_count],
    ['Added scopes', s.added_count],
    ['Delta', s.new_count - s.legacy_count],
  ];
  document.getElementById('stats').innerHTML = cards.map(([k,v]) =>
    `<div class="stat"><div class="k">${{esc(k)}}</div><div class="v">${{v}}</div></div>`
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
    `<a href="${{esc(p.url || '#')}}" target="_blank">${{esc(p.title || '')}}</a> - ` +
    `env=<code>${{esc(p.detected_env || '')}}</code>, binder=<code>${{esc(p.detected_binder || '')}}</code><br>` +
    `Detected scope types: ${{esc(scopeTop)}}`;
  pre.textContent = p.snippet || '';
}}

buildStats();
buildTexts();
buildAddedTable();
buildTypeDelta();
buildPaperCase();
</script>
</body>
</html>
"""

    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"[demo] wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
