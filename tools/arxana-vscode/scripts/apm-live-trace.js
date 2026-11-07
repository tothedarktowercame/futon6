'use strict';
// Live acceptance trace for the APM two-pane demo (M-jvm-arxana):
// drives the real service (:scope/at-position) over the real fixture,
// through the extension's own live-session logic (debounce + sequence gate
// + presentation projection), and records a concise trace with scope and
// assertion ids and semantic digests. Also verifies every returned dom-id
// exists in the committed rendered page bytes.

const fs = require('fs');
const path = require('path');
const { ArxanaClient } = require('../lib/client');
const live = require('../lib/live');

const URL = process.env.ARXANA_URL || 'http://127.0.0.1:7184';
const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'fixtures', 'apm', 'manifest.json'), 'utf8'));
const html = fs.readFileSync(path.join(__dirname, '..', 'fixtures', 'apm', 'index.html'), 'utf8');
const texRev = 'arxana:revision:sha256:' + manifest.artifacts['apm-a93J02.tex'].sha256;
const client = new ArxanaClient(URL);

function scopeOf(id) { return manifest.scopes.find((s) => s.id === id); }
function midOf(s) { return Math.floor((s['tex-start'] + s['tex-end']) / 2); }

async function main() {
  const trace = { generated: new Date().toISOString(), service: URL, steps: [] };
  const caps = await client.capabilities();
  trace.capabilities = { advertised: caps.advertised.operations['scope/at-position'] === 1, ready: caps.ready };
  trace.steps.push({ step: 'capabilities', ok: trace.capabilities.advertised });

  // 1-3: move among three source scopes (paragraph, Lean subsection, final paragraph)
  for (const id of ['a93J02/informal-p1', 'a93J02/subsection-2', 'a93J02/informal-p16']) {
    const s = scopeOf(id);
    const r = await client.scopeAtPosition({ 'artifact/revision-id': texRev, 'position/codepoint': midOf(s) });
    const p = live.presentationFor(r);
    trace.steps.push({
      step: 'move', scope: id, position: midOf(s),
      status: p.status,
      innermost: p.sourceRange,
      targets: p.targets.map((t) => ({
        subject: t.subjectId, assertion: t.assertionId, dom: t.domId,
        verification: t.verification, innermost: t.innermost,
        presentInRenderedPage: t.domId ? html.includes(`id="${t.domId}"`) : false,
      })),
      digest: p.digest,
    });
  }

  // 4: preamble position — typed refusal, highlights clear
  const miss = await client.scopeAtPosition({ 'artifact/revision-id': texRev, 'position/codepoint': 10 });
  const missP = live.presentationFor(miss);
  trace.steps.push({
    step: 'typed-refusal', position: 10,
    status: missP.status, refusal: missP.refusal,
  });

  // 5: rapid movement — three positions fired through the extension's gate;
  // only the newest may paint (real service, real latency ordering).
  const gate = live.sequenceGate();
  const positions = ['a93J02/informal-p2', 'a93J02/subsection-2', 'a93J02/informal-p16'].map(scopeOf).map(midOf);
  const painted = [];
  const responses = await Promise.all(positions.map(async (pos) => {
    const seq = gate.next();
    const r = await client.scopeAtPosition({ 'artifact/revision-id': texRev, 'position/codepoint': pos });
    return { seq, r };
  }));
  for (const { seq, r } of responses) {
    if (gate.admit(seq)) painted.push(live.presentationFor(r));
  }
  trace.steps.push({
    step: 'rapid-movement-suppression',
    submitted: 3,
    painted: painted.length,
    paintedStatus: painted.length === 1 ? painted[0].status : 'INVALID',
    paintedInnermost: painted.length === 1 ? painted[0].sourceRange : null,
  });

  fs.writeFileSync(path.join(__dirname, '..', 'demo-captures', 'apm-live-trace.json'), JSON.stringify(trace, null, 1));
  console.log(JSON.stringify(trace, null, 1));
}

main().catch((e) => { console.error('TRACE FAILED', e); process.exit(1); });
