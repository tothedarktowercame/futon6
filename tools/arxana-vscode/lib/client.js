'use strict';
// arxana.v1 HTTP client. Projection only: sends envelopes, returns either
// the authority `ok` body, the authority `error` body, or a typed client
// refusal. No local identity authority, no semantic interpretation.

const PROTOCOL = 'arxana.v1';

class TypedRefusal extends Error {
  constructor(kind, detail) {
    super(`arxana client refusal: ${kind}`);
    this.kind = kind;       // e.g. 'client/unavailable'
    this.detail = detail || {};
  }
}

function randomMessageId() {
  // Transport metadata (excluded from semantic digests); NOT an identity.
  return `codeserver-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function envelope(operation, body, principal) {
  return {
    'protocol/version': PROTOCOL,
    'message/id': randomMessageId(),
    'operation': `:${operation}`,
    'principal/id': principal || 'futon6-codeserver-demo/vscode',
    'sent-at': new Date().toISOString(),
    'body': body || {},
  };
}

async function post(url, payload, fetchImpl) {
  const f = fetchImpl || fetch;
  let resp;
  try {
    resp = await f(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    throw new TypedRefusal('client/unavailable', { url, cause: String(e && e.message || e) });
  }
  if (resp.status !== 200) {
    throw new TypedRefusal('client/http-status', { url, status: resp.status });
  }
  let text;
  try {
    text = await resp.text();
    return JSON.parse(text);
  } catch (e) {
    throw new TypedRefusal('client/unparseable-response', { url, cause: String(e && e.message || e) });
  }
}

function unwrap(responseEnvelope) {
  if (!responseEnvelope || typeof responseEnvelope !== 'object') {
    throw new TypedRefusal('client/unparseable-response', { received: typeof responseEnvelope });
  }
  if (Object.prototype.hasOwnProperty.call(responseEnvelope, 'ok')) return { ok: responseEnvelope.ok };
  if (Object.prototype.hasOwnProperty.call(responseEnvelope, 'error')) return { error: responseEnvelope.error };
  throw new TypedRefusal('client/unparseable-response', { keys: Object.keys(responseEnvelope) });
}

class ArxanaClient {
  constructor(baseUrl, fetchImpl) {
    this.endpoint = baseUrl.replace(/\/$/, '') + '/arxana/v1';
    this.fetchImpl = fetchImpl;
  }

  async send(operation, body) {
    const env = await post(this.endpoint, envelope(operation, body), this.fetchImpl);
    return unwrap(env);
  }

  async capabilities() {
    const { ok, error } = await this.send('capabilities/handshake', {});
    if (error) return { error };
    // The capability view must prove the operations this demo consumes are
    // advertised; silence is not advertisement.
    const ops = (ok && ok.operations) || {};
    const required = ['scope/resolve', 'correspondence/query'];
    const missing = required.filter((op) => !ops[op]);
    return { ok, advertised: { operations: ops }, missing, ready: missing.length === 0 };
  }

  async scopeResolve(body) {
    return this.send('scope/resolve', body);
  }

  async correspondenceQuery(body) {
    return this.send('correspondence/query', body);
  }

  async scopeAtPosition(body) {
    return this.send('scope/at-position', body);
  }
}

module.exports = { ArxanaClient, TypedRefusal, envelope, unwrap };
