'use strict';
const test = require('node:test'); const assert = require('node:assert/strict');
const fs = require('fs'); const path = require('path');
const f2 = require('../lib/f2');

test('F2 patch uses codepoints for astral edits at both ends and middle', () => {
  const cases = [
    ['🈳abc', 'X🈳abc', { start: 0, end: 0, replacement: 'X' }],
    ['abc🈳', 'abc🈳Z', { start: 4, end: 4, replacement: 'Z' }],
    ['a🈳z', 'a🈳!z', { start: 2, end: 2, replacement: '!' }],
  ];
  for (const [before, after, expected] of cases) {
    assert.deepEqual(f2.minimalPatch(before, after, 'vs', 'op').hunks[0], expected);
  }
});

test('F2 decoration consumes the captured stored relocation shape', () => {
  const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/f2-relocation-live.json')));
  const relocation = f2.relocationFromRecord(fx.subjectId, fx.record, fx.revisionId, fx.text);
  assert.equal(relocation.status, ':resolved');
  assert.deepEqual(relocation.candidates[0].codepointRange, { start: 51, end: 160 });
  // One astral mark occurs before the end boundary, so UTF-16 end is +1.
  assert.deepEqual(relocation.candidates[0].utf16Range, { start: 51, end: 161 });
  const view = f2.scopePresentation([relocation]);
  assert.equal(view.counts.resolved, 1); assert.equal(view.decorations.length, 1);
  assert.deepEqual(view.decorations[0].recordIds, [fx.subjectId, fx.record.id]);
});

test('F2 submit view never auto-chooses and distinguishes landed relocation failure', () => {
  const conflict = f2.submitPresentation({ ok: { 'result/status': ':conflict',
    'conflict/id': 'c', 'alternative/ids': ['a', 'b'], 'head/revision-id': 'r' } });
  assert.deepEqual(conflict.alternativeIds, ['a', 'b']);
  assert.equal(conflict.kind, 'conflict'); assert.equal(conflict.chosen, undefined);
  const failed = f2.submitPresentation({ ok: { 'result/status': ':applied',
    'head/revision-id': 'r2', relocation: { 'result/status': ':failed' } } });
  assert.equal(failed.kind, 'landed-relocation-failed');
  assert.match(failed.message, /edit landed/);
});
