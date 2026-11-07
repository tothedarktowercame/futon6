'use strict';
const test = require('node:test');
const assert = require('node:assert');
const http = require('http');
const fs = require('fs');
const { ArxanaClient, TypedRefusal } = require('../lib/client');

// Recorded authority envelopes: the committed futon4a demo transcript.
const transcript = JSON.parse(fs.readFileSync('/home/joe/code/futon4a/examples/demo-transcript.json', 'utf8'));
const step = (name) => transcript.steps.find((s) => s.step === name).response;
const capsOk = step(':capabilities').ok;
const resolveOk = step(':resolve/emacs').ok;
const corrOk = step(':correspondence').ok;

function mockServer(responses) {
  const server = http.createServer((req, res) => {
    if (req.method === 'POST' && req.url === '/arxana/v1') {
      let body = '';
      req.on('data', (c) => { body += c; });
      req.on('end', () => {
        const msg = JSON.parse(body);
        const op = msg.operation.replace(/^:/, '');
        const r = responses[op];
        if (!r) { res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: { 'error/category': ':missing' } })); return; }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(r));
      });
    } else {
      res.writeHead(404); res.end('{}');
    }
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${server.address().port}` })));
}

test('capabilities proves scope/resolve and correspondence/query advertised', async () => {
  const { server, url } = await mockServer({ 'capabilities/handshake': { ok: capsOk } });
  try {
    const caps = await new ArxanaClient(url).capabilities();
    assert.deepEqual(caps.missing, []);
    assert.equal(caps.ready, true);
    assert.ok(caps.advertised.operations['scope/resolve']);
    assert.ok(caps.advertised.operations['correspondence/query']);
  } finally { server.close(); }
});

test('a stale/limited capability response is a typed refusal, not success', async () => {
  const reduced = JSON.parse(JSON.stringify(capsOk));
  delete reduced.operations['scope/resolve'];
  const { server, url } = await mockServer({ 'capabilities/handshake': { ok: reduced } });
  try {
    const caps = await new ArxanaClient(url).capabilities();
    assert.equal(caps.ready, false);
    assert.deepEqual(caps.missing, ['scope/resolve']);
  } finally { server.close(); }
});

test('endpoint unavailable renders a typed refusal (no fake success)', async () => {
  // nothing listens here: request a port then close it
  const { server, url } = await mockServer({});
  server.close();
  await assert.rejects(
    () => new ArxanaClient(url).capabilities(),
    (e) => e instanceof TypedRefusal && e.kind === 'client/unavailable',
  );
});

test('scope/resolve round-trip returns the authority ok body intact', async () => {
  const { server, url } = await mockServer({ 'scope/resolve': { ok: resolveOk } });
  try {
    const { ok } = await new ArxanaClient(url).scopeResolve({ 'subject/id': 'draft8/introduction' });
    assert.equal(ok['semantic/digest'], '0397248a041571f99764c26675c498f57971bbfbc9ca6391211e199eaa5cff8c');
    assert.equal(ok['result/status'], ':resolved');
  } finally { server.close(); }
});

test('authority error envelope passes through as an error result', async () => {
  const { server, url } = await mockServer({ 'scope/resolve': step(':typed-refusal/stale-ingest') });
  try {
    const { ok, error } = await new ArxanaClient(url).scopeResolve({});
    assert.equal(ok, undefined);
    assert.equal(error['error/category'], ':stale');
    assert.deepEqual(error['error/recovery'], [':refresh', ':rebase-and-resubmit', ':export-patch']);
  } finally { server.close(); }
});
