'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const { correspondenceView } = require('../lib/view');
const { navigationChoice } = require('../lib/navigation');

const transcript = JSON.parse(fs.readFileSync('/home/joe/code/futon4a/examples/demo-transcript.json', 'utf8'));
const corrOk = transcript.steps.find((s) => s.step === ':correspondence').response.ok;
const paths = { 'draft8.html': '/tmp/draft8.html' };

function viewWithAssertionCount(n) {
  const clone = JSON.parse(JSON.stringify(corrOk));
  const first = clone['result/assertions'][0];
  while (clone['result/assertions'].length < n) {
    clone['result/assertions'].push({
      ...first,
      id: 'assertion-extra-' + clone['result/assertions'].length,
    });
  }
  clone['result/count'] = clone['result/assertions'].length;
  return correspondenceView(clone, paths);
}

test('single assertion navigates automatically', () => {
  const view = correspondenceView(corrOk, paths);
  assert.equal(view.assertions.length, 1);
  const choice = navigationChoice(view.assertions, null);
  assert.equal(choice.assertion.assertionId, view.assertions[0].assertionId);
});

test('multiple assertions never select the first implicitly', () => {
  const view = viewWithAssertionCount(3);
  assert.equal(view.assertions.length, 3);
  // No choice made yet: the decision must be "choose among ALL ids",
  // never a navigation of assertions[0].
  const pending = navigationChoice(view.assertions, null);
  assert.ok(pending.choose, 'multi-assertion state must demand an explicit choice');
  assert.equal(pending.choose.length, 3);
  assert.ok(!pending.assertion, 'no implicit assertion selection');
  assert.deepEqual(pending.choose, view.assertions.map((a) => a.assertionId));

  // Cancelled choice (null) is still a typed refusal listing every id —
  // not a fallback to the first assertion.
  const cancelled = navigationChoice(view.assertions, '__cancelled__');
  assert.equal(cancelled.refusal.kind, 'client/selection-required');
  assert.deepEqual(cancelled.refusal.detail.assertionIds,
                   view.assertions.map((a) => a.assertionId));

  // An explicit choice navigates exactly the chosen assertion, not [0].
  const second = view.assertions[1].assertionId;
  const chosen = navigationChoice(view.assertions, second);
  assert.equal(chosen.assertion.assertionId, second);
});

test('zero assertions is a typed refusal, not a crash', () => {
  const choice = navigationChoice([], null);
  assert.equal(choice.refusal.kind, 'client/no-assertions');
});
