'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const { debouncer, sequenceGate, presentationFor } = require('../lib/live');
const { apmWebviewHtml } = require('../lib/apmweb');

test('debounce coalesces rapid movement into one resolve', () => {
  const fired = [];
  let timers = [];
  const clock = {
    now: 0,
    setTimeout: (fn) => { timers.push(fn); return timers.length; },
    clearTimeout: (h) => { timers[h - 1] = null; },
  };
  const d = debouncer(200, clock);
  d.push(() => fired.push('a'));
  d.push(() => fired.push('b'));
  d.push(() => fired.push('c'));
  assert.deepEqual(fired, []);
  timers = timers.filter(Boolean);
  assert.equal(timers.length, 1, 'only one pending timer survives coalescing');
  timers[0]();
  assert.deepEqual(fired, ['c'], 'only the newest event resolves');
});

test('sequence gate refuses stale and out-of-order responses', () => {
  const gate = sequenceGate();
  const s1 = gate.next();
  const s2 = gate.next();
  assert.equal(gate.admit(s1), false, 'older response may not paint');
  assert.equal(gate.admit(s2), true, 'newest response paints');
  const s3 = gate.next();
  assert.equal(gate.admit(s2), false, 'a newer submit invalidates in-flight s2');
  assert.equal(gate.admit(s3), true);
  assert.equal(gate.admit(0), false);
  assert.equal(gate.admit(-1), false);
  assert.equal(gate.admit('3'), false, 'non-integer sequences refused');
});

test('resolved response paints innermost source range and every target', () => {
  const p = presentationFor({ ok: {
    'result/status': 'resolved',
    'semantic/digest': 'abc123',
    'result/scopes': [
      { 'assertion/id': 'A1', 'subject/id': 'a93J02/subsection-1', verification: 'verified',
        direction: 'source-to-target',
        source: { range: { start: 2672, end: 12215 } },
        target: { locator: { kind: 'dom-id', value: 'Sx1.SSx1' } } },
      { 'assertion/id': 'A2', 'subject/id': 'a93J02/informal-p1', verification: 'partial',
        direction: 'source-to-target',
        source: { range: { start: 2727, end: 3353 } },
        target: { locator: { kind: 'dom-id', value: 'Sx1.SSx1.p1' } } },
    ],
    'result/innermost': { 'assertion/id': 'A2', source: { range: { start: 2727, end: 3353 } } },
  } });
  assert.equal(p.status, 'resolved');
  assert.deepEqual(p.sourceRange, { start: 2727, end: 3353 });
  assert.equal(p.targets.length, 2);
  assert.equal(p.targets[1].innermost, true);
  assert.equal(p.targets[0].innermost, false);
  assert.equal(p.targets[1].domId, 'Sx1.SSx1.p1');
  assert.equal(p.digest, 'abc123');
});

test('ambiguous response keeps every assertion visible and selects none', () => {
  const p = presentationFor({ ok: {
    'result/status': 'ambiguous',
    'result/scopes': [
      { 'assertion/id': 'T1', 'subject/id': 'a93J02/twin-a', verification: 'ambiguous',
        direction: 'source-to-target',
        source: { range: { start: 2800, end: 2900 } },
        target: { locator: { value: 'Sx1.SSx1' } } },
      { 'assertion/id': 'T2', 'subject/id': 'a93J02/twin-b', verification: 'ambiguous',
        direction: 'source-to-target',
        source: { range: { start: 2800, end: 2900 } },
        target: { locator: { value: 'Sx1.SSx1' } } },
    ],
    'result/innermost': null,
  } });
  assert.equal(p.status, 'ambiguous');
  assert.equal(p.sourceRange, null, 'no innermost is established for an ambiguous set');
  assert.deepEqual(p.targets.map((t) => t.assertionId), ['T1', 'T2']);
  assert.ok(p.targets.every((t) => t.innermost === false));
});

test('typed refusal and missing scope clear all highlights', () => {
  const cleared = presentationFor({ error: {
    'error/category': 'missing',
    'error/code': 'arxana.scope/no-enclosing-scope',
    'error/detail': { 'position/codepoint': 10 } } });
  assert.equal(cleared.status, 'cleared');
  assert.equal(cleared.sourceRange, null);
  assert.deepEqual(cleared.targets, []);
  assert.equal(cleared.refusal.kind, 'arxana.scope/no-enclosing-scope');
  assert.equal(cleared.refusal.category, 'missing');

  const transport = presentationFor({ error: { 'error/code': undefined } });
  assert.equal(transport.status, 'cleared');
  assert.equal(transport.refusal.kind, 'client/unknown');

  const empty = presentationFor({ ok: { 'result/scopes': [] } });
  assert.equal(empty.status, 'cleared');
  assert.equal(empty.refusal.kind, 'client/empty-result');

  const garbage = presentationFor(null);
  assert.equal(garbage.status, 'cleared');
});

test('webview document renders the page, not raw source, with pinned styles', () => {
  const html = apmWebviewHtml(path.join(__dirname, '..', 'fixtures', 'apm'), 'vscode-webview-test:');
  assert.ok(html.includes('Content-Security-Policy'));
  assert.ok(html.includes('arxana/highlight'), 'webview script injected');
  assert.ok(html.includes('.ltx_page_main'), 'LaTeXML stylesheet inlined');
  assert.ok(!/rel="stylesheet"/.test(html), 'no external stylesheet loads remain');
  assert.ok(!html.includes('lean-proofread'), 'public demo highlight style not carried');
  // The rendered article element (parsed, not source-escaped) is present.
  assert.ok(html.includes('<article class="ltx_document">'));
  // Fixture copies digest-match the manifest (byte identity with the
  // service-side fixture).
  const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'fixtures', 'apm', 'manifest.json'), 'utf8'));
  const crypto = require('crypto');
  for (const f of ['apm-a93J02.tex', 'index.html']) {
    const sha = crypto.createHash('sha256')
      .update(fs.readFileSync(path.join(__dirname, '..', 'fixtures', 'apm', f))).digest('hex');
    assert.equal(sha, manifest.artifacts[f].sha256, `${f} bytes match the pinned manifest`);
  }
});

test('scopeAtPosition sends the pinned revision and codepoint', async () => {
  const { ArxanaClient } = require('../lib/client');
  const sent = [];
  const c = new ArxanaClient('http://127.0.0.1:1/', async (url, opts) => {
    sent.push(JSON.parse(opts.body));
    return { status: 200, text: async () => JSON.stringify({ ok: { 'result/status': 'resolved' } }) };
  });
  const r = await c.scopeAtPosition({ 'artifact/revision-id': 'arxana:revision:sha256:ab', 'position/codepoint': 3000 });
  assert.equal(r.ok['result/status'], 'resolved');
  assert.equal(sent[0].operation, ':scope/at-position');
  assert.equal(sent[0].body['position/codepoint'], 3000);
});
