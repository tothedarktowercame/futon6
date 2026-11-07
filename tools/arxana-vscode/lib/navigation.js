'use strict';

/** D4 ambiguity preservation for arxana.openCorrespondenceTarget.
 *
 * The old command did `assertions[0]` and navigated unconditionally: silent
 * first-assertion selection. This module holds the decision as a pure
 * function so the no-first-selection rule is testable without vscode:
 *
 *   - zero assertions           → typed refusal client/no-assertions
 *   - exactly one assertion     → navigate it automatically
 *   - multiple assertions       → { choose: [ids] } — the caller must obtain
 *                                 an EXPLICIT user choice among all IDs;
 *                                 only the chosen id navigates
 *   - choice cancelled/unknown  → typed refusal client/selection-required
 *                                 listing every assertion id
 *
 * Authority semantics are untouched; this is client-owned selection only.
 */
function navigationChoice(assertions, chosenAssertionId) {
  const list = assertions || [];
  const ids = list.map((a) => a.assertionId);
  if (list.length === 0) {
    return { refusal: { kind: 'client/no-assertions', detail: {} } };
  }
  if (list.length === 1) {
    return { assertion: list[0] };
  }
  if (chosenAssertionId == null) {
    return { choose: ids };
  }
  const a = list.find((x) => String(x.assertionId) === String(chosenAssertionId));
  if (!a) {
    return { refusal: { kind: 'client/selection-required', detail: { assertionIds: ids } } };
  }
  return { assertion: a };
}

module.exports = { navigationChoice };
