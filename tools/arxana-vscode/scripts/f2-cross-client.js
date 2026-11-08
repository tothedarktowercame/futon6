'use strict';
// Headless cross-client acceptance: the extension's pure client code and the
// actual futon4a Emacs client share one private F2 authority.
const { spawnSync } = require('child_process');
const crypto = require('crypto');
const { ArxanaClient } = require('../lib/client');
const f2 = require('../lib/f2');

const service = process.env.F2_SERVICE_URL;
const ready = process.env.F2_READY_FILE;
const futon4a = process.env.FUTON4A_ROOT || '/home/joe/code/futon4a';
const manifest = process.env.F2_MANIFEST_REVISION;
if (!service || !ready || !manifest) throw new Error('F2_SERVICE_URL, F2_READY_FILE and F2_MANIFEST_REVISION are required');
const client = new ArxanaClient(service);

function check(value, message) { if (!value) throw new Error(`FAIL: ${message}`); }
function digest(text) { return crypto.createHash('sha256').update(text, 'utf8').digest('hex'); }
function emacs(form) {
  const r = spawnSync('emacs', ['-Q', '--batch', '-L', `${futon4a}/lisp`, '--eval', form],
    { encoding: 'utf8', env: { ...process.env, F2_READY_FILE: ready } });
  if (r.status !== 0) throw new Error(`Emacs failed: ${r.stderr}\n${r.stdout}`);
  return r.stdout;
}
function emacsEdit(body) {
  return emacs(`(progn (require 'futon4a-formal-f2)
    (setq futon4a-formal-f2-ready-file (getenv "F2_READY_FILE"))
    (futon4a-formal-f2-configure-from-ready)
    (let ((b (futon4a-formal-f2-open " *F2 cross Emacs*")))
      (with-current-buffer b ${body} (futon4a-formal-f2-submit))))`);
}
async function submit(base, before, after, op) {
  const patch = f2.minimalPatch(before, after, 'futon6-codeserver-demo/vscode', op);
  patch.base = base;
  return client.textSubmitPatch(f2.ARTIFACT_ID, patch);
}

(async () => {
  // Two clients hold the same base. Emacs lands a prefix; VS Code rebases a
  // disjoint suffix. Exact authority bytes, not substring presence, decide.
  const initial = (await f2.loadHead(client)).ok;
  emacsEdit('(goto-char (point-min)) (insert "EMACS\\n")');
  const vsText = `${initial.text}\nVSCODE`;
  let response = await submit(initial.revisionId, initial.text, vsText, 'cross-vs-disjoint');
  check(response.ok && response.ok['result/status'] === ':rebased', 'VS Code stale edit rebased');
  let head = (await f2.loadHead(client)).ok;
  check(head.text === `EMACS\n${initial.text}\nVSCODE`, 'exact head is Emacs prefix + base + VS suffix');

  // Edit inside a real relocated scope. The authority records it as missing;
  // the client model lists missing non-spatially rather than painting it.
  let scopes = await f2.loadScopes(client, manifest, head.revisionId, head.text);
  const drawn = scopes.presentation.decorations[0];
  check(drawn, 'a stored relocated scope is drawable');
  const inside = head.text.slice(0, drawn.range.start + 1) + 'ZZ' + head.text.slice(drawn.range.start + 1);
  response = await submit(head.revisionId, head.text, inside, 'cross-vs-inside');
  check(response.ok && response.ok['head/revision-id'], 'inside-scope edit landed');
  head = (await f2.loadHead(client)).ok;
  scopes = await f2.loadScopes(client, manifest, head.revisionId, head.text);
  check(scopes.presentation.missing.length > 0, 'inside-scope edit is shown missing');

  // Both clients now hold HEAD. Emacs lands one same-range edit, while the
  // VS client retains its own different edit against the remembered base.
  const conflictBase = head;
  emacsEdit('(goto-char (point-min)) (delete-char 5) (insert "AAA-CONFLICT")');
  const vsConflictText = `BBB-CONFLICT${conflictBase.text.slice(5)}`;
  response = await submit(conflictBase.revisionId, conflictBase.text, vsConflictText, 'cross-vs-conflict');
  const model = f2.submitPresentation(response);
  check(model.kind === 'conflict' && model.alternativeIds.length === 2,
    `same range stays an unchosen conflict (${JSON.stringify(response)})`);
  const afterEmacs = (await f2.loadHead(client)).ok;
  check(afterEmacs.text.startsWith('AAA-CONFLICT'), 'head remains Emacs side before choice');
  check(vsConflictText.startsWith('BBB-CONFLICT'), 'losing VS document retains its edit');

  const alternatives = await f2.loadAlternatives(client, model.alternativeIds);
  const chosen = alternatives.ok.find((a) => a.text.startsWith('BBB-CONFLICT'));
  check(chosen, 'VS alternative is available with exact text');
  const choice = await client.textChoose({ 'artifact/id': f2.ARTIFACT_ID,
    'conflict/id': model.conflictId, 'alternative/id': chosen.id,
    actor: 'futon6-codeserver-demo/vscode', 'chosen-at': new Date().toISOString() });
  check(choice.ok, 'explicit VS choice landed');
  head = (await f2.loadHead(client)).ok;
  check(head.text === chosen.text, 'authority head equals chosen alternative exactly');

  const out = emacs(`(progn (require 'futon4a-formal-f2)
    (setq futon4a-formal-f2-ready-file (getenv "F2_READY_FILE"))
    (futon4a-formal-f2-configure-from-ready)
    (let ((b (futon4a-formal-f2-open " *F2 reopen*")))
      (with-current-buffer b (princ (concat "HEAD-SHA=" (secure-hash 'sha256 (buffer-string)))))))`);
  check(out.includes(`HEAD-SHA=${digest(head.text)}`), 'reopened Emacs client sees chosen head');
  console.log(JSON.stringify({ status: 'PASS', exactHeadSha256: digest(head.text),
    missingScopes: scopes.presentation.missing.length, conflictAlternatives: model.alternativeIds.length }));
})().catch((e) => { console.error(e.stack || e); process.exit(1); });
