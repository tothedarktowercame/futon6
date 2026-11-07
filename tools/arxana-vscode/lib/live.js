'use strict';
// Live cursor→scope session plumbing for the APM two-pane demo. Pure
// functions (injectable clock) so rapid-movement suppression is unit
// tested without vscode.
//
// Authority boundaries preserved: everything here is presentation. The
// scope/verification/assertion data always comes verbatim from the last
// authority response; this module only decides WHEN a response may paint
// (sequence gate + debounce) and WHAT presentation state it induces
// (including clearing on stale/missing).

/** Debounce with an injected clock and timer handle, for tests.
 * push(fn) schedules fn after delayMs of quiet; cancel() drops it. */
function debouncer(delayMs, clock) {
  const now = clock ? clock.now : () => Date.now();
  const set = clock ? clock.setTimeout : (f) => setTimeout(f, delayMs);
  const clear = clock ? clock.clearTimeout : (h) => clearTimeout(h);
  let handle = null;
  return {
    push(fn) {
      if (handle !== null) clear(handle);
      handle = set(fn);
    },
    cancel() {
      if (handle !== null) clear(handle);
      handle = null;
    },
  };
}

/** Sequence gate: monotonically increasing sequence numbers; a response may
 * paint only if its sequence is still the newest submitted. Out-of-order or
 * stale responses are refused, never queued (they repaint nothing). */
function sequenceGate() {
  let latest = 0;
  return {
    next() { latest += 1; return latest; },
    admit(seq) {
      if (!Number.isInteger(seq) || seq <= 0) return false;
      return seq === latest;
    },
    current() { return latest; },
  };
}

/** Hold only the newest presentation until the webview says its listener is
 * ready.  This prevents the first cursor result from being lost during page
 * construction without replaying stale intermediate highlights. */
function latestDelivery(send) {
  let ready = false;
  let pending = null;
  return {
    publish(message) {
      if (ready) send(message);
      else pending = message;
    },
    markReady() {
      ready = true;
      if (pending !== null) {
        const message = pending;
        pending = null;
        send(message);
      }
    },
    reset() {
      ready = false;
      pending = null;
    },
    ready() { return ready; },
  };
}

/** Circular guided-tour navigation.  The caller supplies only fixture-backed
 * scopes; this function chooses presentation order, never correspondence. */
function tourMove(current, count, direction) {
  if (!Number.isInteger(count) || count < 1) return null;
  const at = Number.isInteger(current) ? current : -1;
  if (direction === 'previous') return (at <= 0 ? count - 1 : at - 1);
  return (at + 1) % count;
}

/** Presentation projection of a :scope/at-position authority response.
 * Never selects among ambiguous scopes: an ambiguous response paints every
 * scope's targets (all assertion ids visible) and marks status 'ambiguous';
 * an error clears all highlights. Returns:
 *   {status:'resolved'|'ambiguous'|'cleared', sourceRange, targets:[...], refusal?}
 * sourceRange is the INNERMOST scope's range for 'resolved' (the only case
 * where an innermost is established by the authority, not by the client). */
function presentationFor(response) {
  if (!response || typeof response !== 'object') {
    return { status: 'cleared', sourceRange: null, targets: [], refusal: { kind: 'client/unparseable-response' } };
  }
  if (response.error) {
    // Typed authority refusal (e.g. arxana.scope/no-enclosing-scope) or
    // transport failure: clear old highlights, surface the refusal.
    return { status: 'cleared', sourceRange: null, targets: [],
             refusal: { kind: (response.error['error/code'] || 'client/unknown'),
                        category: response.error['error/category'],
                        detail: response.error['error/detail'] || {} } };
  }
  const ok = response.ok || {};
  const scopes = ok['result/scopes'] || [];
  if (scopes.length === 0) {
    return { status: 'cleared', sourceRange: null, targets: [], refusal: { kind: 'client/empty-result' } };
  }
  const targets = scopes.map((s) => ({
    assertionId: s['assertion/id'],
    subjectId: s['subject/id'],
    domId: s.target && s.target.locator && s.target.locator.value,
    verification: s.verification,
    direction: s.direction,
    range: s.source && s.source.range,
    innermost: !!(ok['result/innermost']
                  && ok['result/innermost']['assertion/id'] === s['assertion/id']),
  }));
  const innermost = ok['result/innermost'];
  return {
    status: ok['result/status'] === 'ambiguous' ? 'ambiguous' : 'resolved',
    sourceRange: innermost ? innermost.source && innermost.source.range : null,
    targets,
    digest: ok['semantic/digest'],
  };
}

module.exports = { debouncer, sequenceGate, latestDelivery, tourMove, presentationFor };
