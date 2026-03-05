#!/usr/bin/env python3
"""Build a contextual HTML reviewer for PlanetMath distinctor pilot HITs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HITS_IN = ROOT / "data/ct-validation/planetmath-distinctor-pilot-hits.jsonl"
DEFAULT_SUMMARY_IN = ROOT / "data/ct-validation/planetmath-distinctor-pilot.json"
DEFAULT_OUT = ROOT / "data/ct-validation/planetmath-distinctor-pilot-hits.html"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def sanitize_rows(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows, start=1):
        pair = r.get("pair") or []
        pair_s = ", ".join(pair) if isinstance(pair, list) else str(pair)
        hit_id = r.get("hit_id") or f"hit-{i}"
        out.append({
            "idx": i,
            "hit_id": hit_id,
            "domain": r.get("domain", ""),
            "status": r.get("status", ""),
            "entry_id": r.get("entry_id", ""),
            "title": r.get("title", ""),
            "pair": pair,
            "pair_s": pair_s,
            "scope_id": r.get("scope_id", ""),
            "scope_type": r.get("scope_type", ""),
            "scope_start": r.get("scope_start"),
            "scope_end": r.get("scope_end"),
            "scope_match": r.get("scope_match", ""),
            "scope_symbols": r.get("scope_symbols") or [],
            "scope_context": r.get("scope_context", ""),
            "support_latex": r.get("support_latex", ""),
            "support_expr_start": r.get("support_expr_start"),
            "support_expr_end": r.get("support_expr_end"),
            "support_context": r.get("support_context", ""),
            "has_nontrivial_context": bool(r.get("has_nontrivial_context", False)),
            "mit_decision": r.get("mit_decision", "unclear"),
            "mit_label": r.get("mit_label", "unclear"),
            "mit_can_equal": r.get("mit_can_equal"),
            "mit_confidence": r.get("mit_confidence", 0.0),
            "mit_rationale": r.get("mit_rationale") or [],
        })
    return out


def build_fallback_summary(rows: list[dict]) -> dict:
    by_domain = Counter(r["domain"] for r in rows)
    by_status = Counter(r["status"] for r in rows)
    by_scope = Counter(r["scope_type"] for r in rows)
    by_mit = Counter(r.get("mit_label", "unclear") for r in rows)
    return {
        "domains": sorted(k for k in by_domain if k),
        "sampled_entries": None,
        "entry_binder_coverage": None,
        "candidate_pair_events": len(rows),
        "unresolved_pair_events": by_status.get("unresolved", 0),
        "explicit_equal_pair_events": by_status.get("explicit-equal", 0),
        "explicit_distinct_pair_events": by_status.get("explicit-distinct", 0),
        "scope_type_counts_top": by_scope.most_common(12),
        "mit_counts": dict(by_mit),
    }


def load_summary(path: Path, rows: list[dict]) -> dict:
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            agg = obj.get("aggregate", {})
            return {
                "domains": agg.get("domains", []),
                "sampled_entries": agg.get("sampled_entries"),
                "entry_binder_coverage": agg.get("entry_binder_coverage"),
                "candidate_pair_events": agg.get("candidate_pair_events"),
                "unresolved_pair_events": agg.get("unresolved_pair_events"),
                "explicit_equal_pair_events": agg.get("explicit_equal_pair_events"),
                "explicit_distinct_pair_events": agg.get("explicit_distinct_pair_events"),
                "mit_counts": agg.get("mit_counts", {}),
            }
        except Exception:
            pass
    return build_fallback_summary(rows)


def build_html(rows: list[dict], summary: dict, hits_in: Path) -> str:
    rows_json = json.dumps(rows, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)
    source = str(hits_in)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PlanetMath Distinctor HIT Reviewer</title>
  <style>
    :root {
      --bg: #f6f6f1;
      --paper: #ffffff;
      --line: #d7dccf;
      --ink: #1a2117;
      --muted: #4c5a46;
      --accent: #2d7058;
      --warn: #8a6200;
      --bad: #8c2f2f;
      --ok: #236a3c;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: radial-gradient(circle at 80% 0%, #ebf2e6 0%, var(--bg) 45%);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 0.9rem 1.1rem;
      background: rgba(255,255,255,0.88);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(2px);
    }
    h1 {
      margin: 0 0 0.2rem;
      font-size: 1.25rem;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 0.88rem;
    }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 46%) 1fr;
      gap: 0.85rem;
      padding: 0.9rem;
    }
    .card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.72rem;
    }
    .left {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 0.7rem;
      min-height: calc(100vh - 95px);
    }
    .filters {
      position: sticky;
      top: 78px;
      z-index: 4;
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.4rem;
    }
    label {
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 0.35rem;
      margin-bottom: 0.12rem;
    }
    select, input, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0.42rem 0.5rem;
      font-size: 0.88rem;
      color: var(--ink);
      background: #fff;
    }
    textarea {
      min-height: 88px;
      font-family: var(--mono);
      font-size: 0.8rem;
      line-height: 1.3;
      resize: vertical;
    }
    button {
      cursor: pointer;
      background: #f8faf6;
    }
    button:hover {
      background: #f0f6ec;
    }
    .stats {
      margin-top: 0.45rem;
      border-top: 1px dashed var(--line);
      padding-top: 0.45rem;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.35;
    }
    .tag {
      display: inline-block;
      padding: 0.07rem 0.36rem;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 0.74rem;
      margin-right: 0.3rem;
    }
    .tag.unresolved { color: var(--warn); background: #fff8e7; border-color: #dfcfaa; }
    .tag.explicit-equal { color: var(--ok); background: #eef8f1; border-color: #badcc7; }
    .tag.explicit-distinct { color: var(--bad); background: #fff1f1; border-color: #e6c6c6; }
    .decision {
      display: inline-block;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 0.08rem 0.4rem;
      font-size: 0.72rem;
      color: #556;
      background: #f6f8f9;
    }
    .mit {
      display: inline-block;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 0.08rem 0.4rem;
      font-size: 0.72rem;
    }
    .mit.benign-cooccurrence {
      color: var(--ok);
      background: #eef8f1;
      border-color: #badcc7;
    }
    .mit.likely-distinctor {
      color: var(--bad);
      background: #fff1f1;
      border-color: #e6c6c6;
    }
    .mit.unclear {
      color: #66521a;
      background: #fff8e7;
      border-color: #dfcfaa;
    }
    .mono {
      font-family: var(--mono);
      font-size: 0.78rem;
      line-height: 1.3;
    }
    .rows {
      overflow: auto;
      border: 1px solid #e8ece3;
      border-radius: 8px;
      max-height: calc(100vh - 275px);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    thead th {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 0.34rem 0.28rem;
      background: #f9fbf8;
      color: var(--muted);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    tbody td {
      border-bottom: 1px solid #edf1e8;
      padding: 0.32rem 0.28rem;
      vertical-align: top;
    }
    tbody tr {
      cursor: pointer;
    }
    tbody tr:hover {
      background: #f5faf3;
    }
    tbody tr.active {
      background: #e8f4e9;
    }
    .detail h2 {
      margin: 0 0 0.45rem;
      font-size: 1.03rem;
    }
    .detail {
      border: 2px solid #c5d8c8;
      box-shadow: 0 0 0 3px #edf5ee inset;
    }
    .task-box {
      background: linear-gradient(180deg, #eef7ef 0%, #f8fcf7 100%);
      border: 1px solid #bcd7bf;
      border-left: 6px solid var(--accent);
      border-radius: 8px;
      padding: 0.55rem 0.65rem;
      margin-bottom: 0.6rem;
    }
    .task-title {
      margin: 0 0 0.28rem;
      font-size: 0.95rem;
      font-weight: 700;
      color: #184733;
    }
    .task-text {
      margin: 0.15rem 0;
      font-size: 0.84rem;
      color: #2f4334;
      line-height: 1.35;
    }
    .task-list {
      margin: 0.25rem 0 0;
      padding-left: 1rem;
      font-size: 0.83rem;
      color: #2f4334;
      line-height: 1.35;
    }
    .task-list li {
      margin: 0.15rem 0;
    }
    .task-question {
      margin: 0.3rem 0 0;
      font-size: 0.86rem;
      font-weight: 700;
      color: #163f2e;
    }
    .decision-guide {
      margin-top: 0.35rem;
      border-top: 1px dashed #bcd7bf;
      padding-top: 0.35rem;
      font-size: 0.8rem;
      color: #2e4232;
      line-height: 1.35;
    }
    .action-box {
      background: #f8fbf5;
      border: 1px solid #cfdbca;
      border-radius: 8px;
      padding: 0.5rem 0.55rem;
      margin-top: 0.35rem;
      margin-bottom: 0.55rem;
    }
    .detail-row {
      margin-bottom: 0.42rem;
      font-size: 0.9rem;
    }
    .detail pre {
      margin: 0.2rem 0 0.55rem;
      white-space: pre-wrap;
      font-family: var(--mono);
      font-size: 0.78rem;
      line-height: 1.35;
      background: #fbfcfa;
      border: 1px solid #e5eadf;
      border-radius: 8px;
      padding: 0.58rem;
    }
    .hl {
      background: #fff0a8;
      border: 1px solid #e1cd6b;
      border-radius: 4px;
      padding: 0 0.15rem;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }
    .pair {
      border-radius: 4px;
      padding: 0 0.12rem;
      border: 1px solid transparent;
      font-weight: 600;
    }
    .pair-a {
      background: #dff4e4;
      border-color: #8ac49a;
      color: #1d5a2b;
    }
    .pair-b {
      background: #e1ecff;
      border-color: #8fb2f2;
      color: #1f4f9b;
    }
    .hint {
      color: var(--muted);
      font-size: 0.84rem;
    }
    .btn-row {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.4rem;
      margin-top: 0.35rem;
    }
    .btn-row3 {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.4rem;
      margin-top: 0.35rem;
    }
    @media (max-width: 1050px) {
      main {
        grid-template-columns: 1fr;
      }
      .left {
        min-height: 0;
      }
      .rows {
        max-height: 45vh;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>PlanetMath Distinctor HIT Reviewer</h1>
    <p class="sub">Source: __SOURCE_PATH__</p>
  </header>
  <main>
    <section class="left">
      <section class="card filters">
        <div class="grid2">
          <div>
            <label for="domainSel">Domain</label>
            <select id="domainSel"></select>
          </div>
          <div>
            <label for="statusSel">Model Status</label>
            <select id="statusSel"></select>
          </div>
          <div>
            <label for="scopeSel">Scope Type</label>
            <select id="scopeSel"></select>
          </div>
          <div>
            <label for="decisionFilterSel">My Decision</label>
            <select id="decisionFilterSel"></select>
          </div>
        </div>
        <label for="q">Search</label>
        <input id="q" placeholder="entry, pair, scope, snippet..." />
        <div class="btn-row3">
          <button id="nextUnreviewedBtn" type="button">Next Unreviewed</button>
          <button id="exportJsonBtn" type="button">Export JSON</button>
          <button id="exportMdBtn" type="button">Export Findings MD</button>
        </div>
        <div class="stats" id="stats"></div>
      </section>

      <section class="card rows">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Pair</th>
              <th>Entry</th>
              <th>Status</th>
              <th>MIT</th>
              <th>Review</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </section>
    </section>

    <section class="card detail">
      <h2>Selected HIT</h2>
      <div class="task-box">
        <p class="task-title">Review Task: decide if this pair needs a distinctor</p>
        <p class="task-text">Goal: for each selected HIT, decide whether the variable pair should carry an explicit distinctness condition.</p>
        <ol class="task-list">
          <li>Read the highlighted scope context and support expression in context.</li>
          <li>Choose one decision label based on mathematical intent.</li>
          <li>Add a short note that justifies your label, then click <strong>Save Review</strong>.</li>
        </ol>
        <p class="task-question" id="taskQuestion">Decision question: Can A = B in this scope?</p>
        <div class="decision-guide">
          <div><strong>likely-distinctor</strong>: answer is <strong>no</strong>, they should be distinct.</div>
          <div><strong>benign-cooccurrence</strong>: answer is <strong>yes</strong> (or unconstrained).</div>
          <div><strong>scope-detection-issue</strong>: scope extraction is wrong/insufficient.</div>
          <div><strong>unclear</strong>: needs deeper reading or domain judgment.</div>
        </div>
      </div>
      <div id="emptyHint" class="hint">Select a row to inspect context and record findings.</div>
      <div id="detailPanel" hidden>
        <div class="detail-row" id="metaLine"></div>
        <div class="detail-row mono" id="idLine"></div>
        <div class="detail-row mono" id="mitLine"></div>
        <div class="detail-row">
          <span class="hint">Scope declaration context</span>
          <pre id="scopeContext"></pre>
        </div>
        <div class="detail-row">
          <span class="hint">Support math expression</span>
          <pre id="supportLatex"></pre>
        </div>
        <div class="detail-row">
          <span class="hint">Support expression context</span>
          <pre id="supportContext"></pre>
        </div>

        <div class="action-box">
          <div class="detail-row">
            <label for="decisionSel">Decision</label>
            <select id="decisionSel">
              <option value="">(unreviewed)</option>
              <option value="likely-distinctor">Likely Distinctor Needed</option>
              <option value="benign-cooccurrence">Benign Co-occurrence</option>
              <option value="scope-detection-issue">Scope Detection Issue</option>
              <option value="unclear">Unclear</option>
            </select>
          </div>
          <div class="btn-row">
            <button type="button" id="applyMitBtn">Apply MIT Suggestion</button>
            <button type="button" id="saveBtn">Save Review</button>
          </div>
          <div class="btn-row">
            <button type="button" id="clearBtn">Clear Review</button>
            <button type="button" id="nextUnreviewedBtnDetail">Next Unreviewed</button>
          </div>
          <div class="detail-row">
            <label for="notes">Notes / findings text</label>
            <textarea id="notes" placeholder="What did you observe? Why this label?"></textarea>
          </div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const ROWS = __ROWS_JSON__;
    const SUMMARY = __SUMMARY_JSON__;
    const STORAGE_KEY = 'pm_distinctor_review_v2';

    const domainSel = document.getElementById('domainSel');
    const statusSel = document.getElementById('statusSel');
    const scopeSel = document.getElementById('scopeSel');
    const decisionFilterSel = document.getElementById('decisionFilterSel');
    const qInput = document.getElementById('q');
    const rowsEl = document.getElementById('rows');
    const statsEl = document.getElementById('stats');
    const nextUnreviewedBtn = document.getElementById('nextUnreviewedBtn');
    const exportJsonBtn = document.getElementById('exportJsonBtn');
    const exportMdBtn = document.getElementById('exportMdBtn');

    const emptyHint = document.getElementById('emptyHint');
    const detailPanel = document.getElementById('detailPanel');
    const metaLine = document.getElementById('metaLine');
    const idLine = document.getElementById('idLine');
    const mitLine = document.getElementById('mitLine');
    const taskQuestion = document.getElementById('taskQuestion');
    const scopeContext = document.getElementById('scopeContext');
    const supportLatex = document.getElementById('supportLatex');
    const supportContext = document.getElementById('supportContext');
    const decisionSel = document.getElementById('decisionSel');
    const notes = document.getElementById('notes');
    const applyMitBtn = document.getElementById('applyMitBtn');
    const saveBtn = document.getElementById('saveBtn');
    const clearBtn = document.getElementById('clearBtn');
    const nextUnreviewedBtnDetail = document.getElementById('nextUnreviewedBtnDetail');

    let annotations = loadAnnotations();
    let selectedHitId = null;
    let visibleRows = [];
    let rowById = new Map(ROWS.map(r => [r.hit_id, r]));
    let activeTr = null;

    function loadAnnotations() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return {};
        const obj = JSON.parse(raw);
        if (!obj || typeof obj !== 'object') return {};
        return obj;
      } catch {
        return {};
      }
    }

    function saveAnnotations() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
    }

    function uniq(values) {
      return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
    }

    function setOptions(select, values) {
      select.innerHTML = '';
      const all = document.createElement('option');
      all.value = '';
      all.textContent = '(all)';
      select.appendChild(all);
      for (const v of values) {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        select.appendChild(opt);
      }
    }

    function statusTag(status) {
      const cls = String(status || '');
      return `<span class="tag ${cls}">${escapeHtml(cls || 'n/a')}</span>`;
    }

    function decisionBadge(decision) {
      if (!decision) return '<span class="decision">unreviewed</span>';
      return `<span class="decision">${escapeHtml(decision)}</span>`;
    }

    function mitBadge(label, conf) {
      const cls = String(label || 'unclear');
      const pct = Number(conf || 0);
      const confText = Number.isFinite(pct) ? ` ${(pct * 100).toFixed(0)}%` : '';
      return `<span class="mit ${cls}">${escapeHtml(cls)}${confText}</span>`;
    }

    function mitLabelToDecision(label) {
      if (label === 'likely-distinctor') return 'likely-distinctor';
      if (label === 'benign-cooccurrence') return 'benign-cooccurrence';
      if (label === 'unclear') return 'unclear';
      return '';
    }

    function escapeHtml(s) {
      return String(s ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
    }

    function escapeRegExp(s) {
      return String(s).replace(/[-/\\\\^$*+?.()|[\\]{}]/g, '\\\\$&');
    }

    function markSymbol(raw, sym, marker) {
      if (!sym) return raw;
      const re = new RegExp(`(^|[^A-Za-z\\\\\\\\])(${escapeRegExp(sym)})(?![A-Za-z])`, 'g');
      return raw.replace(re, (m, pfx) => `${pfx}${marker}`);
    }

    function formatPair(pair) {
      const a = pair?.[0] || '?';
      const b = pair?.[1] || '?';
      return `<span class="pair pair-a">${escapeHtml(a)}</span> / <span class="pair pair-b">${escapeHtml(b)}</span>`;
    }

    function renderMarkedWithPair(raw, pair) {
      const a = pair?.[0] || '';
      const b = pair?.[1] || '';
      let s = String(raw ?? '');
      s = markSymbol(s, a, '@@PAIRA@@');
      s = markSymbol(s, b, '@@PAIRB@@');
      let esc = escapeHtml(s);
      esc = esc.replace(/&lt;&lt;([\\s\\S]*?)&gt;&gt;/g, '<span class="hl">$1</span>');
      esc = esc.replaceAll('@@PAIRA@@', `<span class="pair pair-a">${escapeHtml(a)}</span>`);
      esc = esc.replaceAll('@@PAIRB@@', `<span class="pair pair-b">${escapeHtml(b)}</span>`);
      return esc;
    }

    function annFor(hitId) {
      return annotations[hitId] || {};
    }

    function rowMatches(r) {
      if (domainSel.value && r.domain !== domainSel.value) return false;
      if (statusSel.value && r.status !== statusSel.value) return false;
      if (scopeSel.value && r.scope_type !== scopeSel.value) return false;
      if (decisionFilterSel.value) {
        const d = annFor(r.hit_id).decision || '';
        if (decisionFilterSel.value === '__unreviewed__') {
          if (d) return false;
        } else if (d !== decisionFilterSel.value) {
          return false;
        }
      }
      const q = qInput.value.trim().toLowerCase();
      if (!q) return true;
      const hay = [
        r.domain, r.entry_id, r.title, r.pair_s, r.scope_type, r.scope_match,
        r.scope_context, r.support_latex, r.support_context, r.mit_label,
        (r.mit_rationale || []).join(' ')
      ].join(' ').toLowerCase();
      return hay.includes(q);
    }

    function renderStats() {
      const reviewed = visibleRows.filter(r => !!annFor(r.hit_id).decision).length;
      const unresolved = visibleRows.filter(r => r.status === 'unresolved').length;
      const likely = visibleRows.filter(r => (annFor(r.hit_id).decision || '') === 'likely-distinctor').length;
      const mitLikely = visibleRows.filter(r => r.mit_label === 'likely-distinctor').length;
      const mitBenign = visibleRows.filter(r => r.mit_label === 'benign-cooccurrence').length;
      const mitUnclear = visibleRows.filter(r => r.mit_label === 'unclear').length;
      statsEl.innerHTML = [
        `<div>visible_hits=${visibleRows.length} reviewed=${reviewed} unreviewed=${visibleRows.length - reviewed}</div>`,
        `<div>visible_unresolved=${unresolved} likely_distinctor=${likely}</div>`,
        `<div>mit_visible: benign=${mitBenign} likely=${mitLikely} unclear=${mitUnclear}</div>`,
        `<div>sampled_entries=${SUMMARY.sampled_entries ?? 'n/a'} entry_binder_coverage=${SUMMARY.entry_binder_coverage ?? 'n/a'}</div>`,
        `<div>candidate_pair_events=${SUMMARY.candidate_pair_events ?? 'n/a'} explicit_distinct=${SUMMARY.explicit_distinct_pair_events ?? 'n/a'}</div>`,
        `<div>mit_global=${JSON.stringify(SUMMARY.mit_counts || {})}</div>`
      ].join('');
    }

    function renderRows() {
      visibleRows = ROWS.filter(rowMatches);
      rowsEl.innerHTML = '';
      activeTr = null;

      for (const r of visibleRows) {
        const tr = document.createElement('tr');
        const ann = annFor(r.hit_id);
        tr.innerHTML = `
          <td>${r.idx}</td>
          <td class="mono">${formatPair(r.pair)}</td>
          <td><div class="mono">${escapeHtml(r.entry_id)}</div><div>${escapeHtml(r.title)}</div></td>
          <td>${statusTag(r.status)}</td>
          <td>${mitBadge(r.mit_label, r.mit_confidence)}</td>
          <td>${decisionBadge(ann.decision || '')}</td>
        `;
        tr.addEventListener('click', () => selectRow(r.hit_id, tr));
        rowsEl.appendChild(tr);
      }
      renderStats();

      if (selectedHitId && visibleRows.some(r => r.hit_id === selectedHitId)) {
        const idx = visibleRows.findIndex(r => r.hit_id === selectedHitId);
        const tr = rowsEl.children[idx];
        if (tr) selectRow(selectedHitId, tr, true);
      } else {
        selectedHitId = null;
        emptyHint.hidden = false;
        detailPanel.hidden = true;
      }
    }

    function selectRow(hitId, tr, keepScroll = false) {
      const r = rowById.get(hitId);
      if (!r) return;
      selectedHitId = hitId;
      if (activeTr) activeTr.classList.remove('active');
      activeTr = tr;
      if (activeTr) activeTr.classList.add('active');

      const ann = annFor(hitId);
      emptyHint.hidden = true;
      detailPanel.hidden = false;
      metaLine.innerHTML =
        `${statusTag(r.status)} ${decisionBadge(ann.decision || '')} ` +
        `<span class="mono">pair=${formatPair(r.pair)}</span> ` +
        `<span class="mono">scope=${escapeHtml(r.scope_type)}</span> ` +
        `${mitBadge(r.mit_label, r.mit_confidence)}`;
      idLine.textContent =
        `domain=${r.domain} entry=${r.entry_id} scope_id=${r.scope_id} hit_id=${r.hit_id}`;
      mitLine.textContent =
        `MIT: decision=${r.mit_decision || 'unclear'} can_equal=${r.mit_can_equal} rationale=${(r.mit_rationale || []).join(',')}`;

      const a = r.pair?.[0] || 'A';
      const b = r.pair?.[1] || 'B';
      taskQuestion.innerHTML = `Decision question: Can ${formatPair([a, b])} be equal in this scope?`;

      scopeContext.innerHTML = renderMarkedWithPair(r.scope_context || '(no context captured)', r.pair);
      supportLatex.innerHTML = renderMarkedWithPair(r.support_latex || '(none)', r.pair);
      supportContext.innerHTML = renderMarkedWithPair(r.support_context || '(no context captured)', r.pair);
      decisionSel.value = ann.decision || '';
      notes.value = ann.notes || '';

      if (!keepScroll && activeTr) {
        activeTr.scrollIntoView({ block: 'nearest' });
      }
    }

    function saveCurrent() {
      if (!selectedHitId) return;
      annotations[selectedHitId] = {
        decision: decisionSel.value || '',
        notes: notes.value || '',
        reviewed_at: new Date().toISOString(),
      };
      if (!annotations[selectedHitId].decision && !annotations[selectedHitId].notes) {
        delete annotations[selectedHitId];
      }
      saveAnnotations();
      renderRows();
    }

    function clearCurrent() {
      if (!selectedHitId) return;
      delete annotations[selectedHitId];
      saveAnnotations();
      decisionSel.value = '';
      notes.value = '';
      renderRows();
    }

    function applyMitSuggestion() {
      if (!selectedHitId) return;
      const r = rowById.get(selectedHitId);
      if (!r) return;
      decisionSel.value = mitLabelToDecision(r.mit_label);
      if (!notes.value) {
        notes.value =
          `MIT suggestion: ${r.mit_label} (conf=${r.mit_confidence}, rationale=${(r.mit_rationale || []).join(',')})`;
      }
    }

    function nextUnreviewed() {
      if (!visibleRows.length) return;
      let start = 0;
      if (selectedHitId) {
        const idx = visibleRows.findIndex(r => r.hit_id === selectedHitId);
        start = idx >= 0 ? idx + 1 : 0;
      }
      for (let i = 0; i < visibleRows.length; i++) {
        const j = (start + i) % visibleRows.length;
        const row = visibleRows[j];
        const ann = annFor(row.hit_id);
        if (!ann.decision) {
          const tr = rowsEl.children[j];
          if (tr) selectRow(row.hit_id, tr);
          return;
        }
      }
    }

    function downloadText(filename, text, mime) {
      const blob = new Blob([text], { type: mime || 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }

    function exportJson() {
      const reviewed = [];
      for (const r of ROWS) {
        const ann = annFor(r.hit_id);
        if (!ann.decision && !ann.notes) continue;
        reviewed.push({
          hit_id: r.hit_id,
          decision: ann.decision || '',
          notes: ann.notes || '',
          reviewed_at: ann.reviewed_at || '',
          domain: r.domain,
          entry_id: r.entry_id,
          title: r.title,
          pair: r.pair,
          scope_type: r.scope_type,
          model_status: r.status,
          mit_label: r.mit_label,
          mit_confidence: r.mit_confidence,
          mit_rationale: r.mit_rationale,
          support_latex: r.support_latex,
        });
      }
      const payload = {
        generated_at: new Date().toISOString(),
        source: '__SOURCE_PATH__',
        reviewed_count: reviewed.length,
        total_hits: ROWS.length,
        summary: SUMMARY,
        reviewed,
      };
      downloadText('planetmath-distinctor-findings.json', JSON.stringify(payload, null, 2), 'application/json');
    }

    function exportMarkdown() {
      const reviewed = [];
      for (const r of ROWS) {
        const ann = annFor(r.hit_id);
        if (!ann.decision) continue;
        reviewed.push({ r, ann });
      }
      reviewed.sort((a, b) => {
        if (a.ann.decision !== b.ann.decision) return a.ann.decision.localeCompare(b.ann.decision);
        return a.r.domain.localeCompare(b.r.domain);
      });
      const lines = [];
      lines.push('# PlanetMath Distinctor Findings');
      lines.push('');
      lines.push(`- Generated: ${new Date().toISOString()}`);
      lines.push(`- Source HIT file: __SOURCE_PATH__`);
      lines.push(`- Reviewed: ${reviewed.length} / ${ROWS.length}`);
      lines.push('');
      let k = 1;
      for (const item of reviewed) {
        const r = item.r;
        const ann = item.ann;
        lines.push(`${k}. [${ann.decision}] ${r.domain} :: ${r.entry_id} :: pair=(${r.pair_s}) scope=${r.scope_type}`);
        if (ann.notes) lines.push(`   note: ${ann.notes.replaceAll('\\n', ' ')}`);
        lines.push(`   mit: ${r.mit_label} conf=${r.mit_confidence} rationale=${(r.mit_rationale || []).join(',')}`);
        if (r.support_latex) lines.push(`   support: ${r.support_latex}`);
        k += 1;
      }
      lines.push('');
      downloadText('planetmath-distinctor-findings.md', lines.join('\\n'), 'text/markdown');
    }

    function initDecisionFilterOptions() {
      decisionFilterSel.innerHTML = '';
      const opts = [
        ['', '(all)'],
        ['__unreviewed__', 'unreviewed'],
        ['likely-distinctor', 'likely-distinctor'],
        ['benign-cooccurrence', 'benign-cooccurrence'],
        ['scope-detection-issue', 'scope-detection-issue'],
        ['unclear', 'unclear'],
      ];
      for (const [value, label] of opts) {
        const o = document.createElement('option');
        o.value = value;
        o.textContent = label;
        decisionFilterSel.appendChild(o);
      }
    }

    function init() {
      setOptions(domainSel, uniq(ROWS.map(r => r.domain)));
      setOptions(statusSel, uniq(ROWS.map(r => r.status)));
      setOptions(scopeSel, uniq(ROWS.map(r => r.scope_type)));
      initDecisionFilterOptions();

      [domainSel, statusSel, scopeSel, decisionFilterSel].forEach(el => el.addEventListener('change', renderRows));
      qInput.addEventListener('input', renderRows);
      applyMitBtn.addEventListener('click', applyMitSuggestion);
      saveBtn.addEventListener('click', saveCurrent);
      clearBtn.addEventListener('click', clearCurrent);
      nextUnreviewedBtn.addEventListener('click', nextUnreviewed);
      nextUnreviewedBtnDetail.addEventListener('click', nextUnreviewed);
      exportJsonBtn.addEventListener('click', exportJson);
      exportMdBtn.addEventListener('click', exportMarkdown);
      renderRows();
    }

    init();
  </script>
</body>
</html>
"""
    html = template.replace("__ROWS_JSON__", rows_json)
    html = html.replace("__SUMMARY_JSON__", summary_json)
    html = html.replace("__SOURCE_PATH__", source.replace("\\", "\\\\"))
    return html


def main() -> int:
    parser = argparse.ArgumentParser(description="Build contextual HTML reviewer for distinctor pilot HITs")
    parser.add_argument("--hits-in", default=str(DEFAULT_HITS_IN), help="Input HIT JSONL path")
    parser.add_argument("--summary-in", default=str(DEFAULT_SUMMARY_IN), help="Input summary JSON path")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output HTML path")
    args = parser.parse_args()

    hits_in = Path(args.hits_in)
    summary_in = Path(args.summary_in)
    out = Path(args.out)

    rows = sanitize_rows(load_jsonl(hits_in))
    summary = load_summary(summary_in, rows)
    html_text = build_html(rows, summary, hits_in)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"[hits-html] wrote {out}")
    print(f"[hits-html] rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
