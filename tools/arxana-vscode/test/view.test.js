'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const { resolveView, correspondenceView, refusalView, renderHtml } = require('../lib/view');
const fx = require('../fixtures/draft8-v1.json');

const transcript = JSON.parse(fs.readFileSync('/home/joe/code/futon4a/examples/demo-transcript.json', 'utf8'));
const step = (name) => transcript.steps.find((s) => s.step === name).response;
const resolveOk = step(':resolve/emacs').ok;
const corrOk = step(':correspondence').ok;

function ambiguousResolveOk() {
  // Authority shape, two candidates: same revision/strategy/evidence, a
  // second candidate with a different id/range (simulated server output).
  const clone = JSON.parse(JSON.stringify(resolveOk));
  const first = clone['result/candidates'][0];
  clone['result/candidates'].push({
    ...first,
    'candidate/id': 'ff'.repeat(32),
    location: { ...first.location, range: { start: 900, end: 940 } },
  });
  return clone;
}

test('unique resolution displays revision, strategy/evidence, candidate and semantic digest', () => {
  const model = resolveView(resolveOk);
  assert.equal(model.status, 'resolved');
  assert.equal(model.artifactRevisionId, fx.artifacts['intro-generated.tex']['revision-id']);
  assert.equal(model.semanticDigest, '0397248a041571f99764c26675c498f57971bbfbc9ca6391211e199eaa5cff8c');
  assert.equal(model.candidates.length, 1);
  assert.equal(model.candidates[0].verification, 'verified-hash');
  assert.equal(model.candidates[0].range.start, 792);
  assert.equal(model.evidence[0].stageDiagnostics.join(), 'context/hash');
  const html = renderHtml('t', model);
  assert.ok(html.includes('0397248a041571f99764c26675c498f57971bbfbc9ca6391211e199eaa5cff8c'));
  assert.ok(html.includes('f5533f5afc890a1cd6315c3aef8830861634f6c7d4dec4029b73c9704b9f8dd3'));
});

test('ambiguous resolution displays every candidate and selects none', () => {
  const model = resolveView(ambiguousResolveOk());
  assert.equal(model.ambiguous, true);
  assert.equal(model.requiresChoice, true);
  assert.equal(model.candidates.length, 2);
  // no chosen/selected/active field exists anywhere in the model
  const flat = JSON.stringify(model);
  assert.ok(!/"selected"/.test(flat) && !/"chosen"/.test(flat) && !/"active"/.test(flat));
  const html = renderHtml('t', model);
  assert.ok(html.includes('f5533f5afc890a1cd6315c3aef8830861634f6c7d4dec4029b73c9704b9f8dd3'));
  assert.ok(html.includes('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'.slice(0, 64)));
  assert.ok(html.includes('ambiguous'));
});

test('selection containment is a marker, never a selection', () => {
  const model = resolveView(resolveOk, { start: 800, end: 810 });
  assert.equal(model.candidates[0].containsSelection, true);
  const outside = resolveView(resolveOk, { start: 0, end: 5 });
  assert.equal(outside.candidates[0].containsSelection, false);
});

test('correspondence displays direction, partial status, provenance and navigations', () => {
  const paths = {};
  for (const [l, a] of Object.entries(fx.artifacts)) paths[l] = a.path;
  const model = correspondenceView(corrOk, paths);
  assert.equal(model.count, 1);
  assert.equal(model.semanticDigest, '9c91f35ee33414db14fe93b7a2bbbaff3eda8fc496bca06624b6538f80070d00');
  const a = model.assertions[0];
  assert.equal(a.direction, 'source-to-target');
  assert.equal(a.verification, 'partial');
  assert.deepEqual(a.method, ['quote', 'source-map']);
  assert.equal(a.producer.tool, 'latexml_oxide');
  assert.equal(a.producer['build/id'], 'arxana-build:sha256:7ae3821ffcf2265d');
  assert.equal(a.source.manifestationId.startsWith('arxana:manifestation:sha256:'), true);
  assert.deepEqual(a.navigateSource, { kind: 'source', codepointOffset: 792, revisionId: fx.artifacts['intro-generated.tex']['revision-id'] });
  assert.deepEqual(a.navigateTarget, { kind: 'target', line: 10, path: fx.artifacts['draft8.html'].path });
  const html = renderHtml('t', model);
  assert.ok(html.includes('source-to-target'));
  assert.ok(html.includes('The object of study is a finite family'));
});

test('capabilities view renders advertised operations and refusal on missing', () => {
  const capsOk = { operations: { 'scope/resolve': 1, 'correspondence/query': 1 } };
  const good = renderHtml('t', { capabilities: capsOk, missing: [], ready: true });
  assert.ok(good.includes('scope/resolve') && good.includes('advertised'));
  const bad = renderHtml('t', { capabilities: capsOk, missing: ['scope/resolve'], ready: false });
  assert.ok(bad.includes('refused: operations not advertised: scope/resolve'));
});

test('typed refusal view is object-qualified', () => {
  const model = refusalView('resolution', { kind: 'client/unavailable', detail: { url: 'x' } });
  const html = renderHtml('t', model);
  assert.ok(html.includes('resolution: refused'));
  assert.ok(html.includes('client/unavailable'));
  assert.ok(!html.includes('resolved'));
});
