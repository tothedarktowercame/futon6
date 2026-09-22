// Three layers at once, printed as ink separations: S1 source marks on the cyan plate,
// S3 proof graphs on magenta, S4 expository scopes on yellow. Each word and formula of
// the rendered paper is placed at its offset in the run's source, and each plate inks it
// by what its layer says there; overlaps mix like ink (cyan + magenta = blue, cyan +
// yellow = green, magenta + yellow = red, all three = grey). The black plate is the text.
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('m7-data').textContent);
  const starts = data.starts, article = document.querySelector('article') || document.body;
  const el = (tag, cls, text, parent) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.append(e);
    return e;
  };
  const lineOf = off => { let lo = 0, hi = starts.length; while (lo < hi) { const m = (lo + hi) >> 1; if (starts[m] <= off) lo = m + 1; else hi = m; } return lo; };
  // sourcepos "file:line:col-file:line:col". A block's start column is exact at col-2
  // (0-based); a formula's start column falls just past its first token, so a formula is
  // placed by the S1 math mark that contains that point.
  const pos = e => {
    const m = /^(\d+):(\d+):(\d+)-(\d+):(\d+):(\d+)$/.exec(e.dataset.sourcepos || '');
    return m && +m[1] === data.file ? [starts[+m[2] - 1] + +m[3] - 2, starts[+m[5] - 1] + +m[6] - 2] : null;
  };

  // ---- the S1 marks, indexed by source line ------------------------------------------
  const KINDS = data.kinds;                       // [name, group, meaning]
  const MARKS = data.marks;                       // [start, end, kind, grounded, tip]
  const MATH = KINDS.findIndex(k => k[0] === 'math');
  const byLine = new Map();
  MARKS.forEach((m, i) => { for (let l = lineOf(m[0]); l <= lineOf(m[1] - 1); l++) { if (!byLine.has(l)) byLine.set(l, []); byLine.get(l).push(i); } });
  const marksAt = (a, b) => { const out = new Set(); for (let l = lineOf(a); l <= lineOf(b - 1); l++) (byLine.get(l) || []).forEach(i => { if (MARKS[i][0] < b && a < MARKS[i][1]) out.add(i); }); return [...out]; };

  // ---- place every word and formula at its source offset -------------------------------
  data.source = JSON.parse(document.getElementById('m7-source').textContent);   // the run's marks text
  const units = [];                               // {e, a, b}
  const SKIP = n => n.nodeType === 1 && (n.matches('.m7-banner, .m7-rail, .m7-note, .m7-read, script, style, .ltx_bibliography') );
  function mathRange(e) {
    const p = pos(e); if (!p) return null;
    const hits = marksAt(p[0], p[0] + 1).filter(i => MARKS[i][2] === MATH);
    if (!hits.length) return p[1] > p[0] ? p : null;
    const inner = hits.reduce((x, y) => (MARKS[y][1] - MARKS[y][0] < MARKS[x][1] - MARKS[x][0] ? y : x));
    return [MARKS[inner][0], MARKS[inner][1]];
  }
  // A word is found whole (not inside \\begin or another word) and nearby: short words
  // recur everywhere, so they get a short reach.
  const letter = ch => /[A-Za-z0-9]/.test(ch || '');
  function findWord(w, from) {
    const reach = from + (w.length <= 3 ? 60 : 400);
    for (let at = data.source.indexOf(w, from); at >= 0 && at <= reach; at = data.source.indexOf(w, at + 1)) {
      const before = data.source[at - 1];
      if (!letter(before) && before !== '\\' && !letter(data.source[at + w.length])) return skipsProse(from, at) ? -1 : at;
    }
    return -1;
  }
  // Reaching past three plain words means this is a later occurrence, not this one:
  // the word stays unplaced (no ink) rather than inked at the wrong place.
  const MARKUP = /\$[^$]*\$|\\\[[\s\S]*?\\\]|\\begin\{[^}]*\}[\s\S]*?\\end\{[^}]*\}|\\[A-Za-z]+(\{[^}]*\})?|%.*|[^A-Za-z]/g;
  const skipsProse = (a, b) => b - a > 12 &&
    data.source.slice(a, b).replace(MARKUP, ' ').split(/\s+/).filter(x => x.length > 2).length >= 3;
  // A formula with no position is the next S1 math span at the cursor, if reaching it
  // skips no prose.
  const mathStarts = MARKS.map((m, i) => i).filter(i => MARKS[i][2] === MATH).sort((x, y) => MARKS[x][0] - MARKS[y][0]);
  function nextMath(from) {
    let lo = 0, hi = mathStarts.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (MARKS[mathStarts[m]][0] < from) lo = m + 1; else hi = m; }
    const i = mathStarts[lo];
    return i != null && MARKS[i][0] - from < 200 && !skipsProse(from, MARKS[i][0]) ? [MARKS[i][0], MARKS[i][1]] : null;
  }
  let words = 0, placed = 0;
  function walk(node, cur) {
    for (const n of [...node.childNodes]) {
      if (SKIP(n)) continue;
      if (n.nodeType === 1 && n.tagName.toLowerCase() === 'math') {
        const r = mathRange(n) || nextMath(cur.at);
        if (r) { units.push({e: n, a: r[0], b: r[1]}); cur.at = Math.max(cur.at, r[1]); }
        continue;
      }
      if (n.nodeType === 1) { if (blockSet.has(n)) cur.at = blockStart(n); walk(n, cur); continue; }
      if (n.nodeType !== 3 || !/[A-Za-z0-9]/.test(n.nodeValue)) continue;
      const frag = document.createDocumentFragment(), text = n.nodeValue;
      let last = 0;
      for (const w of text.matchAll(TOKEN)) {
        words++;
        const at = findWord(w[0], cur.at);
        if (at < 0) continue;                               // not found near here: left unplaced
        placed++; cur.at = at + w[0].length;
        frag.append(text.slice(last, w.index));
        const s = el('span', 'm7-u', w[0], frag);
        units.push({e: s, a: at, b: at + w[0].length});
        last = w.index + w[0].length;
      }
      frag.append(text.slice(last));
      n.replaceWith(frag);
    }
  }
  // A block built by an author macro (\\df{...} -> \\begin{dfn}\\emph{#1}) is stamped where
  // the expansion ended, after its own text, and its formulas carry no position. So a
  // block starts where its own first words are found near its stamp.
  const TOKEN = /[A-Za-z0-9][A-Za-z0-9'’-]*/g;
  function leadWords(b, n) {
    const out = [], w = document.createTreeWalker(b, NodeFilter.SHOW_TEXT);
    for (let t; out.length < n && (t = w.nextNode());) {
      if (t.parentElement.closest('math, .ltx_tag')) continue;
      for (const m of t.nodeValue.matchAll(TOKEN)) { out.push(m[0]); if (out.length === n) break; }
    }
    return out;
  }
  const esc = x => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  function blockStart(b) {
    const at = pos(b)[0], lead = leadWords(b, 4);
    if (lead.length < 2) return at;
    const re = new RegExp('(?<![A-Za-z\\\\])' + lead.map(esc).join('(?![A-Za-z])[\\s\\S]{1,120}?(?<![A-Za-z\\\\])'), 'g');
    const lo = Math.max(0, at - 4000), win = data.source.slice(lo, at + 400);
    let best = null;
    for (const m of win.matchAll(re)) { const d = Math.abs(lo + m.index - at); if (!best || d < best.d) best = {d, i: lo + m.index}; }
    return best ? best.i : at;
  }
  const BLOCK = /^(P|H[1-6]|TD|TH|LI|DT|DD|FIGCAPTION)$/;
  const blockSet = new Set([...article.querySelectorAll('[data-sourcepos]')].filter(e => BLOCK.test(e.tagName) && pos(e) && !e.closest('math')));
  // Walk from the outermost blocks; a nested block resets the cursor to its own exact start.
  const blocks = [...blockSet].filter(e => { for (let a = e.parentElement; a; a = a.parentElement) if (blockSet.has(a)) return false; return true; });
  // ---- ink ----------------------------------------------------------------------------
  const INK = {c: 0.42, m: 0.42, y: 0.55};
  const color = (c, m, y) => `rgb(${Math.round(255 * (1 - INK.c * c))},${Math.round(255 * (1 - INK.m * m))},${Math.round(255 * (1 - INK.y * y))})`;
  const proofs = data.notes.filter(n => n.type === 'proof'), regions = data.notes.filter(n => n.type === 'region');
  const hit = (a, b, rs) => rs.some(([x, y]) => x < b && a < y);
  let ctl;
  const carvedAt = line => data.carved.find(r => r.lines[0] <= line && line <= r.lines[1]);
  function quotes(mock) {
    const out = [];
    proofs.forEach(p => p.nodes.forEach(n => {
      if (mock && n.mock.verdict === 'unanchored') return;
      (mock && n.mock.verdict === 're-anchored' && n.proposal ? [n.proposal.at] : n.at).forEach(r => out.push([r[0], r[1], p, n]));
    }));
    return out;
  }
  function plate(u, mock, qs) {
    const line = lineOf(u.a);
    let c = 0;
    if (ctl.c.checked) {
      marksAt(u.a, u.b).forEach(i => { if (KINDS[MARKS[i][2]][1] === ctl.cg.value) c = Math.max(c, MARKS[i][3] ? 1 : 0.5); });
    }
    let m = 0;
    if (ctl.m.checked) {
      if (qs.some(q => q[0] < u.b && u.a < q[1])) m = 1;
      else if (proofs.some(p => p.lo <= line && line <= p.hi)) m = 0.3;
    }
    let y = 0;
    if (ctl.y.checked) {
      if (regions.some(r => r.scopes.some(s => !(mock && s.mock.verdict === 'rejected') && s.lines[0] <= line && line <= s.lines[1]))) y = 1;
      else { const c = carvedAt(line); if (c) y = c.type === 'in-proof' ? 0.14 : 0.3; }
    }
    return [c, m, y];
  }
  const TERMCLS = {'tagged as defined': 'm7-t-def', 'tagged generically': 'm7-t-gen', 'untagged': 'm7-t-none'};
  const occs = data.terms.flatMap(t => t.occurrences.map(o => ({a: o[0], b: o[1], status: o[2], how: o[3], t})));
  function paint() {
    const mock = document.body.classList.contains('m7-mock'), qs = quotes(mock);
    units.forEach(u => {
      const [c, m, y] = plate(u, mock, qs);
      u.e.style.backgroundColor = c || m || y ? color(c, m, y) : '';
      Object.values(TERMCLS).forEach(k => u.e.classList.remove(k));
      if (ctl.t.checked) { const o = occs.find(o => o.a < u.b && u.a < o.b); if (o) u.e.classList.add(TERMCLS[o.status]); }
    });
  }

  // ---- banner: plates, legend, glossary, defined terms --------------------------------
  const banner = document.querySelector('.m7-banner');
  const box = el('div', 'm7-plates');
  banner.querySelector('h2').after(box);
  el('h3', null, 'Three layers at once', box);
  const P = el('p', null, null, box);
  P.append('Each layer is an ink plate, as in colour printing, so all three show together and overlaps mix. ' +
           'Hover (or tap) any word or formula to read every layer at that point.');
  const row = el('p', 'm7-plate-row', null, box);
  const check = (label, sw, on) => { const l = el('label', null, null, row); const c = el('input', null, null, l); c.type = 'checkbox'; c.checked = on;
    const s = el('span', 'm7-sw', null, l); s.style.background = sw; l.append(' ' + label); return c; };
  ctl = {};
  ctl.c = check('C — S1 source marks:', color(1, 0, 0), true);
  ctl.cg = el('select', null, null, row.lastChild);
  [['term', 'symbols and terms'], ['clause', 'clauses'], ['region', 'environments and displays']].forEach(([v, t]) => { const o = el('option', null, t, ctl.cg); o.value = v; });
  ctl.m = check('M — S3 proof graphs', color(0, 1, 0), true);
  ctl.y = check('Y — S4 expository scopes', color(0, 0, 1), true);
  ctl.t = check('defined terms', 'transparent', true);
  row.lastChild.querySelector('.m7-sw').className = 'm7-sw m7-sw-term';
  const legend = el('div', 'm7-legend-grid', null, box);
  const sw = (bg, text, cls) => { const d = el('div', null, null, legend); const s = el('span', 'm7-sw ' + (cls || ''), null, d); s.style.background = bg; d.append(' ' + text); };
  sw(color(1, 0, 0), 'S1 tied it to a meaning (grounded)');
  sw(color(0.5, 0, 0), 'S1 tagged it but found no meaning');
  sw(color(0, 1, 0), 'a proof-graph node quotes it');
  sw(color(0, 0.3, 0), 'inside a proof S3 read, not quoted');
  sw(color(0, 0, 1), 'an S4 scope cites this line');
  sw(color(0, 0, 0.3), 'a region S4 would read now, no scope in this run');
  sw(color(0, 0, 0.14), 'prose inside a proof, carved as an in-proof region');
  sw(color(1, 1, 0), 'C + M: S1 grounded what a node quotes');
  sw(color(0, 1, 1), 'M + Y: a node quotes a line a scope also reads');
  sw(color(1, 0, 1), 'C + Y: S1 grounded what a scope reads');
  sw('transparent', 'defined term, S1 tagged it as defined', 'm7-sw-term m7-t-def');
  sw('transparent', 'defined term, S1 tagged it only generically', 'm7-sw-term m7-t-gen');
  sw('transparent', 'defined term, S1 did not tag it', 'm7-sw-term m7-t-none');
  const align = el('p', 'm7-legend', null, box);
  ctl.cg.addEventListener('change', paint);
  [ctl.c, ctl.m, ctl.y, ctl.t].forEach(c => c.addEventListener('change', paint));
  document.querySelectorAll('.m7-banner input').forEach(c => c.addEventListener('change', paint));

  const K = data.summary.carving, cp = el('p', null, null, box);
  el('b', null, 'Where S4 reads: ', cp);
  cp.append(`this run carved ${K.run.regions} region(s), ${K.run.expository_lines} of ${K.run.body_lines} body lines (${K.run.pct}%). ` +
            `The current extractor, given S1's environments, carves ${K.now.regions} (` +
            Object.entries(K.now.types).map(([t, n]) => `${n} ${t}`).join(', ') + `), ${K.now.expository_lines} lines (${K.now.pct}%). ` +
            'Only the run\'s regions have scopes; the rest show what the next run would read.');
  const T = data.summary.terms;
  const tp = el('p', null, null, box);
  el('b', null, `${T.defined} terms the paper defines, ${T.occurrences} occurrences: `, tp);
  tp.append(`S1 tags ${T['tagged as defined']} as the paper's defined term, ${T['tagged generically']} only as a generic lexicon word or phrase, ` +
            `and ${T.untagged} not at all.`);
  const td = el('details', null, null, box);
  el('summary', null, 'the defined terms, one by one', td);
  const table = el('table', 'm7-terms', null, td);
  const hr = el('tr', null, null, table); ['term', 'defined', 'uses', 'as defined', 'generic', 'untagged', 'S1 grounded it to'].forEach(h => el('th', null, h, hr));
  data.terms.forEach(t => {
    const r = el('tr', null, null, table), n = s => t.occurrences.filter(o => o[2] === s).length;
    el('td', null, t.term.replace(/\$([^$]*)\$/g, '$1').replace(/\\mathcal\{?(\w)\}?|\\(\w+)/g, (_, a, b) => a || b), r).title = t.definition;
    el('td', null, 'L' + t.line, r); el('td', null, String(t.occurrences.length), r);
    el('td', 'm7-ok', String(n('tagged as defined')), r); el('td', 'm7-warn', String(n('tagged generically')), r); el('td', 'm7-bad', String(n('untagged')), r);
    el('td', 'm7-legend', [...new Set(t.occurrences.map(o => o[3]).filter(Boolean).flatMap(x => x.split(', ')))].join(', '), r);
  });
  const gd = el('details', null, null, box);
  el('summary', null, 'what every label on this page means', gd);
  Object.entries(data.glossary).forEach(([sec, defs]) => {
    el('h4', null, sec, gd);
    const dl = el('dl', 'm7-gloss', null, gd);
    Object.entries(defs).forEach(([k, v]) => { el('dt', null, k, dl); el('dd', null, v, dl); });
  });

  // ---- readout: every layer at the point under the pointer ------------------------------
  const read = el('div', 'm7-read m7-hidden', null, document.body);
  let pinned = null;
  const excerpt = (a, b) => data.source.slice(a, b).replace(/\s+/g, ' ');
  function show(u) {
    read.textContent = ''; read.classList.remove('m7-hidden');
    const line = lineOf(u.a);
    const h = el('div', 'm7-read-h', null, read);
    el('code', null, excerpt(u.a, u.b).slice(0, 120), h); h.append(`  · L${line}`);
    if (pinned) el('span', 'm7-legend', '  (pinned — click again to release)', h);
    const o = occs.find(o => o.a < u.b && u.a < o.b);
    if (o) { const d = el('div', 'm7-read-sec', null, read); el('b', null, 'Defined term: ', d);
      d.append(`${o.t.term} (defined L${o.t.line}) — ${o.status}${o.how ? ': ' + o.how : ''}`);
      el('div', 'm7-q', o.t.definition.slice(0, 300) + (o.t.definition.length > 300 ? ' …' : ''), d); }
    const c = el('div', 'm7-read-sec m7-read-c', null, read); el('b', null, 'C · S1 source marks', c);
    const ms = marksAt(u.a, u.b).sort((x, y) => (MARKS[x][1] - MARKS[x][0]) - (MARKS[y][1] - MARKS[y][0]));
    if (!ms.length) el('div', 'm7-legend', 'no mark here', c);
    ms.forEach(i => { const [a, b, k, g, tip] = MARKS[i], d = el('div', null, null, c);
      const kk = el('span', 'm7-pill ' + (g ? 'm7-ok' : 'm7-warn'), KINDS[k][0], d); kk.title = KINDS[k][2];
      d.append(' ' + (tip || excerpt(a, b).slice(0, 80)).slice(0, 140)); });
    const m = el('div', 'm7-read-sec m7-read-m', null, read); el('b', null, 'M · S3 proof graphs', m);
    const mock = document.body.classList.contains('m7-mock');
    const qs = quotes(mock).filter(q => q[0] < u.b && u.a < q[1]);
    const inProof = proofs.filter(p => p.lo <= line && line <= p.hi);
    if (!inProof.length) el('div', 'm7-legend', 'not in a proof S3 read', m);
    inProof.forEach(p => { const quoting = qs.filter(q => q[2] === p);
      el('div', 'm7-legend', `${p.id} (L${p.lo}–${p.hi}): ${quoting.length ? 'quoted by' : 'no node quotes this'}`, m);
      quoting.forEach(q => { const d = el('div', null, null, m); el('span', 'm7-pill m7-stated', q[3].id + ' ' + q[3].kind, d).title = data.glossary['S3 node kinds'][q[3].kind] || 'UNDEFINED';
        d.append(' ' + q[3].gloss); }); });
    const y = el('div', 'm7-read-sec m7-read-y', null, read); el('b', null, 'Y · S4 expository scopes', y);
    const sc = regions.flatMap(r => r.scopes.filter(s => s.lines[0] <= line && line <= s.lines[1]));
    const cv = carvedAt(line);
    if (cv) { const d = el('div', 'm7-legend', null, y); const p = el('span', 'm7-pill m7-stated', cv.type, d);
      p.title = data.glossary['This page'][cv.type] || 'UNDEFINED'; d.append(` ${cv.id} · L${cv.lines[0]}–${cv.lines[1]} · ${cv.section}`); }
    if (!sc.length) el('div', 'm7-legend', cv ? (regions.some(r => r.lo <= line && line <= r.hi) ? 'read in this run; no scope cites this line' : 'not read in this run: no scopes yet') : 'not in an expository region', y);
    sc.forEach(s => { const d = el('div', null, null, y); el('span', 'm7-pill ' + (s['bare-parent'] ? 'm7-warn' : 'm7-stated'), s.kind, d).title = data.glossary['S4 scope kinds'][s.kind] || 'UNDEFINED';
      d.append(' ' + (s.fill == null ? 'held' : s.fill) + (mock ? ` — mock: ${s.mock.verdict}` : '')); });
  }
  const unitOf = new Map();
  let hide;
  article.addEventListener('mouseover', ev => { if (pinned) return; const u = unitOf.get(ev.target.closest('.m7-u, math'));
    clearTimeout(hide); if (u) show(u); else hide = setTimeout(() => read.classList.add('m7-hidden'), 600); });
  article.addEventListener('click', ev => { const u = unitOf.get(ev.target.closest('.m7-u, math')); if (!u) return;
    pinned = pinned === u ? null : u; show(u); });

  blocks.forEach(b => walk(b, {at: blockStart(b)}));
  units.forEach(u => unitOf.set(u.e, u));
  window.m7units = units;                         // for audits from the console
  align.textContent = `Placed ${placed} of ${words} words and ${units.length - placed} formulas at their source offsets; ` +
                      `${words - placed} words could not be placed and carry no ink.`;
  paint();
})();
