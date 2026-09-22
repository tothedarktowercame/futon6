(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('m7-data').textContent);
  const S = data.summary;
  const el = (tag, cls, text, parent) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    if (parent) parent.append(e);
    return e;
  };
  // Every label is defined in data.glossary; one that is not is marked on the page, not hidden.
  const DEFINE = Object.assign({}, ...Object.values(data.glossary));
  const define = text => DEFINE[text.replace(/^mock:\s*/, '').replace(/^[\d/]+\s+/, '')];
  const pill = (text, cls, parent, title) => {
    const p = el('span', 'm7-pill ' + cls, text, parent), meaning = define(text);
    p.title = meaning ? meaning + (title ? ' — ' + title : '') : 'UNDEFINED LABEL' + (title ? ' — ' + title : '');
    if (!meaning) p.classList.add('m7-undef');
    return p;
  };
  const pct = (a, b) => b ? Math.round(100 * a / b) + '%' : '–';

  // Source blocks of THIS file, by the line they start on.
  const BLOCK = /^(DIV|P|TABLE|H[1-6]|UL|OL|LI|FIGURE)$/;
  const blocks = [...document.querySelectorAll('[data-sourcepos]')].flatMap(e => {
    const m = /^(\d+):(\d+):/.exec(e.dataset.sourcepos);
    return m && +m[1] === data.file && BLOCK.test(e.tagName) && !e.closest('math') ? [{e, line: +m[2]}] : [];
  });
  function covering(lo, hi) {
    const inside = blocks.filter(b => b.line >= lo && b.line <= hi).map(b => b.e);
    return inside.filter(e => !inside.some(o => o !== e && o.contains(e)));
  }

  const MOCK = {'kept': 'm7-ok', 're-anchored': 'm7-cite', 'unanchored': 'm7-warn', 'retyped': 'm7-cite', 'held': 'm7-warn', 'rejected': 'm7-bad'};
  function mockLine(m, parent) {
    const d = el('div', 'm7-legend m7-mockline', null, parent);
    pill('mock: ' + m.verdict, MOCK[m.verdict] || 'm7-warn', d); d.append(m.reason);
    return d;
  }
  const WARRANT = {'claim': ['stated', 'm7-stated'], 'citation': ['cited', 'm7-cite'], 'missing-warrant': ['missing', 'm7-hole']};
  const VERDICT = {'agrees': ['quote matches gloss', 'm7-ok'], 're-anchor': ['quote is another clause', 'm7-bad'],
                   'unclear': ['no clause matches', 'm7-warn'], 'uncheckable': ["formula: can't check", 'm7-stated']};

  // ---- banner: what this page is, and the run-level numbers --------------------------
  const article = document.querySelector('article') || document.body;
  const banner = el('div', 'm7-banner');
  el('h2', null, `Mark7 reading of ${S.paper}`, banner);
  el('p', null, `Run ${S.run}. Each note in the margin is what the pipeline made of the passage beside it: ` +
     `proof graphs (S3) and expository scopes (S4). Hover a note to light up its passage; hover a node to light up the lines it quotes.`, banner);
  const p1 = el('p', null, null, banner);
  el('b', null, `${S.proofs} proofs, ${S.nodes} nodes. `, p1);
  const checkable = S.nodes - S.uncheckable;
  p1.append(`Of the ${checkable} nodes whose gloss has enough words to check, the quoted mathematics matches the gloss for ` +
            `${S.agrees} (${pct(S.agrees, checkable)}); for ${S['re-anchor']} (${pct(S['re-anchor'], checkable)}) a different offered clause matches better. ` +
            `${S.uncheckable} mostly-formula glosses can't be checked this way. ` +
            `${S.sequential} nodes (${pct(S.sequential, S.nodes)}) cite clause sᵢ as node i — the list's order, not the text's.`);
  const p2 = el('p', null, null, banner);
  el('b', null, `${S.edges} inference steps: `, p2);
  p2.append(`${S.warrants.claim} stated, ${S.warrants.citation} cited, ${S.warrants['missing-warrant']} missing — the missing ones are the holes.`);
  if (S.regions) {
    const p3 = el('p', null, null, banner);
    el('b', null, `${S.scopes} expository scopes in ${S.regions} region(s): `, p3);
    p3.append(`${S['scopes-flagged']} fail a mechanical check, ${S['scopes-bare-parent']} carry only a generic parent kind.`);
  }
  const controls = el('p', null, null, banner);
  const toggle = (label, on) => { const l = el('label', null, null, controls); const c = el('input', null, null, l); c.type = 'checkbox'; c.checked = on; l.append(' ' + label); return c; };
  const showProofs = toggle('proofs', true), showScopes = toggle('scopes', true), onlyFlagged = toggle('only notes with problems', false);
  const showMock = toggle('proposed contract (mock)', false);
  const mockP = el('p', 'm7-mockline', null, banner);
  el('b', null, 'Proposed contract, mocked on this run: ', mockP);
  const MN = S.mock.nodes, MS = S.mock.scopes;
  mockP.append(`S3 nodes ${MN.kept} kept, ${MN['re-anchored']} re-anchored to the clause their gloss describes, ` +
               `${MN.unanchored} left with a gloss but no quote. ` +
               (S.scopes ? `S4 scopes ${MS.kept} kept, ${MS.retyped} given a specific kind, ${MS.held} held, ${MS.rejected} rejected.` : '') +
               ' Nothing here is re-run; it shows what the rules would do to what this run produced.');
  const unanchored = el('p', 'm7-legend', null, banner);
  const h1 = article.querySelector('.ltx_authors') || article.querySelector('h1');
  (h1 ? h1.after(banner) : article.prepend(banner));

  // ---- notes ------------------------------------------------------------------------
  const rail = el('div', 'm7-rail', null, document.body);
  const notes = [];
  function glossOf(proof, id) { const n = proof.nodes.find(x => x.id === id); return n ? n.gloss : id; }

  function litNode(n, on) {
    covering(n.lines[0], n.lines[1]).forEach(e => e.classList.toggle('m7-lit-node', on));
  }
  function proofNote(p) {
    const note = el('div', 'm7-note proof');
    const h = el('h4', null, `Proof${p.proved ? ' of a ' + p.proved : ''} `, note);
    el('span', 'm7-where', `· ${p.id} · L${p.lo}–${p.hi}`, h);
    const w = {claim: 0, citation: 0, 'missing-warrant': 0};
    p.edges.forEach(e => { if (e.warrant in w) w[e.warrant]++; });
    const bar = el('div', 'm7-bar', null, note), total = p.edges.length || 1;
    [['claim', '#9a9a8e'], ['citation', '#2d6a9f'], ['missing-warrant', '#7a4fa3']].forEach(([k, c]) => {
      const s = el('span', null, null, bar); s.style.width = (100 * w[k] / total) + '%'; s.style.background = c;
    });
    const line = el('p', 'm7-legend', null, note);
    line.append(`${p.edges.length} steps: `);
    pill(`${w.claim} stated`, 'm7-stated', line); pill(`${w.citation} cited`, 'm7-cite', line);
    pill(`${w['missing-warrant']} missing`, 'm7-hole', line, 'missing warrants are the holes S9 mines');
    const agree = p.nodes.filter(n => n.verdict === 'agrees').length;
    const moved = p.nodes.filter(n => n.verdict === 're-anchor').length;
    const seq = p.nodes.filter(n => n.sequential).length;
    const q = el('p', 'm7-legend', null, note);
    q.append(`${p.nodes.length} nodes: `);
    const unk = p.nodes.filter(n => n.verdict === 'uncheckable').length;
    pill(`${agree} quotes match`, agree === p.nodes.length - unk ? 'm7-ok' : 'm7-warn', q);
    if (unk) pill(`${unk} formula-only`, 'm7-stated', q, "too few words for this check to judge");
    if (moved) pill(`${moved} quote another clause`, 'm7-bad', q, 'a different offered clause matches the gloss better');
    if (seq > p.nodes.length / 2) pill(`${seq}/${p.nodes.length} cite sᵢ in order`, 'm7-bad', q, 'node i cites clause sᵢ: citations follow the list, not the text');
    note.dataset.problem = (moved || seq > p.nodes.length / 2 || agree < (p.nodes.length - unk) / 2) ? '1' : '';
    const tally = v => p.nodes.filter(n => n.mock.verdict === v).length;
    const mq = el('p', 'm7-legend m7-mockline', null, note);
    mq.append('mock: '); pill(`${tally('kept')} kept`, 'm7-ok', mq); pill(`${tally('re-anchored')} re-anchored`, 'm7-cite', mq);
    if (tally('unanchored')) pill(`${tally('unanchored')} unanchored`, 'm7-warn', mq);

    const steps = el('details', null, null, note);
    el('summary', null, 'the argument, step by step', steps);
    p.edges.forEach(e => {
      const s = el('div', 'm7-step', null, steps);
      el('div', null, '∴ ' + glossOf(p, e.conclusion), s);
      if (e.premises.length) el('div', 'm7-legend', 'from ' + e.premises.map(id => '“' + glossOf(p, id) + '”').join(' and '), s);
      const r = el('div', 'm7-legend', null, s);
      el('span', 'm7-rel', e.relation + ' ', r);
      const [label, cls] = WARRANT[e.warrant] || [e.warrant || 'no warrant', 'm7-warn'];
      pill(label, cls, r); if (e.why) r.append(e.why);
    });
    const nodes = el('details', null, null, note);
    el('summary', null, 'each node: gloss against the text it quotes', nodes);
    p.nodes.forEach(n => {
      const box = el('div', 'm7-node', null, nodes);
      el('div', 'm7-k', `${n.id} · ${n.kind} · L${n.lines[0]}${n.lines[1] !== n.lines[0] ? '–' + n.lines[1] : ''}`, box)
        .title = n.kind + ': ' + (data.glossary['S3 node kinds'][n.kind] || 'UNDEFINED');
      el('div', null, n.gloss, box);
      const [label, cls] = VERDICT[n.verdict];
      const v = el('div', 'm7-legend', null, box); pill(label, cls, v); v.append(`quotes ${n.cites.join(', ')}`);
      el('code', 'm7-q', n.quote.length > 260 ? n.quote.slice(0, 260) + ' …' : n.quote, box);
      mockLine(n.mock, box);
      if (n.proposal) {
        const pr = el('div', 'm7-prop', null, box);
        el('div', 'm7-legend m7-ok', `better match: ${n.proposal.span} (${n.proposal.kind}, L${n.proposal.line})`, pr);
        el('code', 'm7-q', n.proposal.text.length > 260 ? n.proposal.text.slice(0, 260) + ' …' : n.proposal.text, pr);
      }
      box.addEventListener('mouseenter', () => litNode(n, true));
      box.addEventListener('mouseleave', () => litNode(n, false));
    });
    return note;
  }
  const FLAG = {'echo': 'repeats its slot definition', 'unanchored': 'not in the lines it cites', 'bare-noun': 'a bare noun phrase'};
  function regionNote(r) {
    const note = el('div', 'm7-note region');
    const h = el('h4', null, 'Exposition ', note);
    el('span', 'm7-where', `· ${r.id} · L${r.lo}–${r.hi} · ${r.scopes.length} scopes`, h);
    const flagged = r.scopes.filter(s => s.flags.length).length, generic = r.scopes.filter(s => s['bare-parent']).length;
    const line = el('p', 'm7-legend', null, note);
    pill(`${r.scopes.length - flagged} pass the checks`, flagged ? 'm7-warn' : 'm7-ok', line);
    if (flagged) pill(`${flagged} flagged`, 'm7-bad', line);
    if (generic) pill(`${generic} generic kind`, 'm7-warn', line, 'a parent kind whose vocabulary has more specific children');
    note.dataset.problem = (flagged || generic > r.scopes.length / 2) ? '1' : '';
    const mt = v => r.scopes.filter(x => x.mock && x.mock.verdict === v).length;
    const ms = el('p', 'm7-legend m7-mockline', null, note);
    ms.append('mock: ');
    [['kept', 'm7-ok'], ['retyped', 'm7-cite'], ['held', 'm7-warn'], ['rejected', 'm7-bad']].forEach(([v, c]) => { if (mt(v)) pill(`${mt(v)} ${v}`, c, ms); });
    const list = el('details', null, null, note); list.open = r.scopes.length <= 4;
    el('summary', null, 'what the scopes say', list);
    r.scopes.forEach(s => {
      const box = el('div', 'm7-scope', null, list);
      const k = el('div', 'm7-legend', null, box);
      pill(s.kind, s['bare-parent'] ? 'm7-warn' : 'm7-stated', k, s['bare-parent'] ? 'generic parent kind' : '');
      k.append(`L${s.lines[0]}${s.lines[1] !== s.lines[0] ? '–' + s.lines[1] : ''}`);
      el('div', null, s.fill == null ? 'held: ' + (typeof s.held === 'string' ? s.held : 'no fill') : s.fill, box);
      if (s.flags.length) { const f = el('div', 'm7-legend', null, box); s.flags.forEach(x => pill(FLAG[x] || x, 'm7-bad', f)); }
      if (s.mock) mockLine(s.mock, box);
      box.addEventListener('mouseenter', () => covering(s.lines[0], s.lines[1]).forEach(e => e.classList.add('m7-lit-node')));
      box.addEventListener('mouseleave', () => covering(s.lines[0], s.lines[1]).forEach(e => e.classList.remove('m7-lit-node')));
    });
    return note;
  }

  const missing = [];
  data.notes.forEach(n => {
    const cover = covering(n.lo, n.hi);
    if (!cover.length) { missing.push(`${n.id} (L${n.lo}–${n.hi})`); return; }
    const note = n.type === 'proof' ? proofNote(n) : regionNote(n);
    cover.forEach(e => e.classList.add('m7-covered'));
    note.addEventListener('mouseenter', () => { note.classList.add('m7-active'); cover.forEach(e => e.classList.add('m7-lit')); });
    note.addEventListener('mouseleave', () => { note.classList.remove('m7-active'); cover.forEach(e => e.classList.remove('m7-lit')); });
    note.querySelectorAll('details').forEach(d => d.addEventListener('toggle', layout));
    rail.append(note);
    notes.push({n, note, anchor: cover[0]});
  });
  unanchored.textContent = missing.length ? `No rendered position for ${missing.length} note(s): ${missing.join(', ')} — shown nowhere rather than guessed.` : 'Every note is placed beside the passage it describes.';

  // ---- layout: beside the text column, never overlapping, in reading order ---------------
  function layout() {
    // The text column is the paragraph <p>, not its full-width .ltx_para wrapper.
    const paras = [...article.querySelectorAll('.ltx_para > .ltx_p')].slice(0, 60).map(p => p.getBoundingClientRect().right);
    paras.sort((a, b) => a - b);
    const right = (paras.length ? paras[Math.floor(paras.length / 2)] : 0) + window.scrollX;
    const left = right + 32, width = Math.min(460, document.documentElement.clientWidth - left - 24);
    const inline = width < 260;
    document.body.classList.toggle('m7-inline', inline);
    let floor = 0;
    notes.forEach(({n, note, anchor}) => {
      const shown = (n.type === 'proof' ? showProofs.checked : showScopes.checked) && (!onlyFlagged.checked || note.dataset.problem);
      note.classList.toggle('m7-hidden', !shown);
      if (inline) { if (note.parentNode !== anchor.parentNode) anchor.before(note); return; }
      if (note.parentNode !== rail) rail.append(note);
      if (!shown) return;
      const top = Math.max(anchor.getBoundingClientRect().top + window.scrollY, floor);
      note.style.left = left + 'px'; note.style.width = width + 'px'; note.style.top = top + 'px';
      floor = top + note.offsetHeight + 10;
    });
  }
  const mockMode = () => { document.body.classList.toggle('m7-mock', showMock.checked); layout(); };
  showMock.addEventListener('change', mockMode);
  [showProofs, showScopes, onlyFlagged].forEach(c => c.addEventListener('change', layout));
  window.addEventListener('resize', layout);
  layout(); setTimeout(layout, 400); window.addEventListener('load', layout);
})();
