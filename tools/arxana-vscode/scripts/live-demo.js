'use strict';
// Live acceptance run against the isolated loopback JVM arxana.v1 service.
// Usage: node scripts/live-demo.js [url]     (default http://127.0.0.1:7084)
// Executes the demo's acceptance flow through the extension's own lib code
// and writes rendered view captures to demo-captures/.

const fs = require('fs');
const path = require('path');
const { ArxanaClient } = require('../lib/client');
const { resolveView, correspondenceView, refusalView, renderHtml } = require('../lib/view');

const url = process.argv[2] || 'http://127.0.0.1:7084';
const fx = require('../fixtures/draft8-v1.json');
const outDir = path.join(__dirname, '..', 'demo-captures');
fs.mkdirSync(outDir, { recursive: true });

function write(name, title, model) {
  fs.writeFileSync(path.join(outDir, name), renderHtml(title, model));
  console.log(`wrote demo-captures/${name}`);
}

(async () => {
  const c = new ArxanaClient(url);
  const failures = [];
  const check = (label, cond, detail) => {
    console.log(`${cond ? 'PASS' : 'FAIL'} ${label}${cond ? '' : ` — ${JSON.stringify(detail)}`}`);
    if (!cond) failures.push(label);
  };

  // 1. Capability view proves scope/resolve and correspondence/query advertised.
  try {
    const caps = await c.capabilities();
    if (caps.error) {
      write('capabilities.html', 'Arxana capabilities', refusalView('capability', caps.error));
      check('capabilities advertised', false, caps.error);
    } else {
      write('capabilities.html', 'Arxana capabilities', { capabilities: caps.ok, missing: caps.missing, ready: caps.ready });
      check('scope/resolve advertised', !!caps.advertised.operations['scope/resolve'], caps.missing);
      check('correspondence/query advertised', !!caps.advertised.operations['correspondence/query'], caps.missing);
    }
  } catch (e) {
    write('capabilities.html', 'Arxana capabilities', refusalView('capability', e));
    check('capabilities advertised', false, `${e.kind || ''} ${e.message || ''}`);
  }

  // 2. draft8 source resolution: revision, strategy/evidence, candidate, digest.
  const intro = fx.artifacts['intro-generated.tex'];
  try {
    const { ok, error } = await c.scopeResolve({
      'artifact/revision-id': intro['revision-id'],
      'strategy/revision-id': fx['strategy/revision-id'],
      'subject/id': fx['subject/id'],
    });
    if (error) {
      write('resolve.html', 'Arxana resolve', refusalView('resolution', error));
      check('scope resolved', false, error);
    } else {
      const model = resolveView(ok);
      write('resolve.html', 'Arxana resolve', model);
      check('artifact revision matches fixture', model.artifactRevisionId === intro['revision-id']);
      check('semantic digest equals committed transcript', model.semanticDigest === '0397248a041571f99764c26675c498f57971bbfbc9ca6391211e199eaa5cff8c', model.semanticDigest);
      check('candidate listed with codepoint range', model.candidates[0] && model.candidates[0].range && model.candidates[0].range.start === 792, model.candidates);
      check('evidence finder stages retained', model.evidence.length >= 1 && model.evidence[0].stageDiagnostics.includes('context/hash'));
    }
  } catch (e) {
    write('resolve.html', 'Arxana resolve', refusalView('resolution', e));
    check('scope resolved', false, e.kind);
  }

  // 3. source→HTML correspondence: direction/partial/provenance + navigation.
  try {
    const { ok, error } = await c.correspondenceQuery({ 'subject/id': fx['subject/id'] });
    if (error) {
      write('correspondence.html', 'Arxana correspondence', refusalView('correspondence', error));
      check('correspondence queried', false, error);
    } else {
      const paths = {};
      for (const [l, a] of Object.entries(fx.artifacts)) paths[l] = a.path;
      const model = correspondenceView(ok, paths);
      write('correspondence.html', 'Arxana correspondence', model);
      const a = model.assertions[0];
      check('correspondence semantic digest equals committed transcript', model.semanticDigest === '9c91f35ee33414db14fe93b7a2bbbaff3eda8fc496bca06624b6538f80070d00', model.semanticDigest);
      check('directional partial with provenance', a && a.direction === 'source-to-target' && a.verification === 'partial' && !!a.producer['build/id']);
      check('source navigation projection (codepoint 792)', a.navigateSource && a.navigateSource.codepointOffset === 792);
      check('target navigation projection (draft8.html line)', a.navigateTarget && !!a.navigateTarget.path && a.navigateTarget.line === 10);
    }
  } catch (e) {
    write('correspondence.html', 'Arxana correspondence', refusalView('correspondence', e));
    check('correspondence queried', false, e.kind);
  }

  // 4. Typed refusal when the endpoint is absent.
  const dead = new ArxanaClient('http://127.0.0.1:1');
  try {
    await dead.capabilities();
    check('unavailable endpoint refuses', false, 'no refusal');
  } catch (e) {
    write('refusal.html', 'Arxana capabilities', refusalView('capability', e));
    check('unavailable endpoint refuses', e.kind === 'client/unavailable', e.kind);
  }

  console.log(failures.length ? `\n${failures.length} FAILURES` : '\nALL LIVE CHECKS PASS');
  process.exit(failures.length ? 1 : 0);
})();
