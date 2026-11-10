// node --test tools/futon-relay-channel/server.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { scanOnce } from './server.mjs';

function fixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'relay-inbox-'));
  const sent = [], logs = [];
  const ctx = { dir, seat: 'claude-cs', state: { pushed: new Set(), bad: new Set() },
    send: (p) => sent.push(p), log: (s) => logs.push(s) };
  const deliver = (id, body) => fs.writeFileSync(path.join(dir, id + '.json'),
    typeof body === 'string' ? body : JSON.stringify(body));
  return { dir, sent, logs, ctx, deliver };
}

const delivery = (id) => ({ 'job-id': id, prompt: 'hello ' + id, from: 'claude-13',
  'bell-type': 'request', 'ack-url': `/api/alpha/invoke/jobs/${id}/ack` });

test('a new delivery is pushed once, with the ack instruction and meta', () => {
  const f = fixture();
  f.deliver('job-1', delivery('job-1'));
  assert.deepEqual(scanOnce(f.ctx).pushed, ['job-1']);
  assert.deepEqual(scanOnce(f.ctx).pushed, []);
  assert.equal(f.sent.length, 1);
  assert.match(f.sent[0].content, /^\[Agency bell job-1 from claude-13, request\]/);
  assert.match(f.sent[0].content, /clink-inbox\.py --agent claude-cs ack job-1/);
  assert.match(f.sent[0].content, /hello job-1$/);
  assert.deepEqual(f.sent[0].meta, { 'job-id': 'job-1', from: 'claude-13',
    'ack-url': '/api/alpha/invoke/jobs/job-1/ack' });
});

test('a restart (fresh state) pushes unacked deliveries again; acked ones are gone', () => {
  const f = fixture();
  f.deliver('job-1', delivery('job-1'));
  f.deliver('job-2', delivery('job-2'));
  scanOnce(f.ctx);
  // job-1 acked: Agency moves it to consumed/.
  fs.mkdirSync(path.join(f.dir, 'consumed'));
  fs.renameSync(path.join(f.dir, 'job-1.json'), path.join(f.dir, 'consumed', 'job-1.json'));
  const restarted = { ...f.ctx, state: { pushed: new Set(), bad: new Set() } };
  assert.deepEqual(scanOnce(restarted).pushed, ['job-2']);
});

test('malformed JSON is skipped with one log line, and retried only if the file changes', () => {
  const f = fixture();
  f.deliver('bad', '{not json');
  f.deliver('noprompt', { 'job-id': 'noprompt' });
  const r = scanOnce(f.ctx);
  assert.deepEqual(r.skipped.sort(), ['bad', 'noprompt']);
  assert.equal(f.logs.filter((l) => l.startsWith('skip malformed')).length, 2);
  scanOnce(f.ctx);
  assert.equal(f.logs.filter((l) => l.startsWith('skip malformed')).length, 2);
  // The file is rewritten correctly: it is now pushed.
  const later = new Date(Date.now() + 5000);
  f.deliver('bad', delivery('bad'));
  fs.utimesSync(path.join(f.dir, 'bad.json'), later, later);
  assert.deepEqual(scanOnce(f.ctx).pushed, ['bad']);
  assert.equal(f.sent.length, 1);
});

test('a failed send leaves the delivery to be pushed on the next pass', () => {
  const f = fixture();
  f.deliver('job-1', delivery('job-1'));
  let fail = true;
  const ctx = { ...f.ctx, send: (p) => { if (fail) throw new Error('stdout closed'); f.sent.push(p); } };
  assert.deepEqual(scanOnce(ctx).failed, ['job-1']);
  fail = false;
  assert.deepEqual(scanOnce(ctx).pushed, ['job-1']);
});

test('temp files and consumed/ are ignored', () => {
  const f = fixture();
  fs.writeFileSync(path.join(f.dir, 'job-9.json.tmp'), '{');
  fs.mkdirSync(path.join(f.dir, 'consumed'));
  fs.writeFileSync(path.join(f.dir, 'consumed', 'job-8.json'), JSON.stringify(delivery('job-8')));
  assert.deepEqual(scanOnce(f.ctx), { pushed: [], skipped: [], failed: [] });
});
