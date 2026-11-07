'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const { matchRevision, sha256Hex, revisionIdFromSha } = require('../lib/digest');
const fx = require('../fixtures/draft8-v1.json');

const revisions = {};
for (const [logical, a] of Object.entries(fx.artifacts)) revisions[logical] = a['revision-id'];

test('exact fixture bytes digest-match their committed revision id', () => {
  for (const [logical, a] of Object.entries(fx.artifacts)) {
    const bytes = fs.readFileSync(a.path);
    const m = matchRevision(bytes, revisions);
    assert.equal(m.kind, 'matched', logical);
    assert.equal(m.logicalId, logical);
    assert.equal(m.revisionId, a['revision-id']);
  }
});

test('modified bytes are a typed refusal, never a guessed revision', () => {
  const bytes = fs.readFileSync(fx.artifacts['intro-generated.tex'].path);
  const edited = Buffer.concat([bytes, Buffer.from('% one edit\n', 'utf8')]);
  const m = matchRevision(edited, revisions);
  assert.equal(m.kind, 'unmatched');
  assert.equal(m.sha256, sha256Hex(edited));
  assert.ok(!Object.prototype.hasOwnProperty.call(m, 'revisionId'));
});

test('revision id shape is authority-owned (client never mints)', () => {
  assert.equal(revisionIdFromSha(sha256Hex(Buffer.from('x'))).startsWith('arxana:revision:sha256:'), true);
});
