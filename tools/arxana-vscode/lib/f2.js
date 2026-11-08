'use strict';

const crypto = require('crypto');
const { utf16ToCodepoint, codepointRangeToUtf16Range } = require('./utf16');

const ARTIFACT_ID = 'formal-arxana:f1-conversation';

function minimalPatch(before, after, author, operationId) {
  let prefix16 = 0;
  while (prefix16 < before.length && prefix16 < after.length) {
    const a = before.codePointAt(prefix16); const b = after.codePointAt(prefix16);
    if (a !== b) break;
    prefix16 += a > 0xffff ? 2 : 1;
  }
  let aEnd16 = before.length; let bEnd16 = after.length;
  while (aEnd16 > prefix16 && bEnd16 > prefix16) {
    const aStart = before.offsetByCodePoints ? before.offsetByCodePoints(aEnd16, -1) : previousBoundary(before, aEnd16);
    const bStart = after.offsetByCodePoints ? after.offsetByCodePoints(bEnd16, -1) : previousBoundary(after, bEnd16);
    if (before.slice(aStart, aEnd16) !== after.slice(bStart, bEnd16)) break;
    aEnd16 = aStart; bEnd16 = bStart;
  }
  return {
    author,
    'op-id': operationId,
    hunks: [{
      start: utf16ToCodepoint(before, prefix16),
      end: utf16ToCodepoint(before, aEnd16),
      replacement: after.slice(prefix16, bEnd16),
    }],
  };
}

function previousBoundary(text, end) {
  let p = end - 1;
  if (p > 0) {
    const unit = text.charCodeAt(p);
    if (unit >= 0xdc00 && unit <= 0xdfff) p--;
  }
  return p;
}

function stableOperationId(baseRevision, text, principal = 'futon6-codeserver-demo/vscode') {
  return `vscode:${crypto.createHash('sha256').update(baseRevision).update('\0').update(text).update('\0').update(principal).digest('hex')}`;
}

function relocationFromRecord(subjectId, record, revisionId, text) {
  const content = record && record.content;
  if (!content || content['target/revision-id'] !== revisionId) return null;
  const status = content['resolution/status'];
  const candidates = (content.candidates || []).map((candidate) => {
    const cp = candidate.location && candidate.location.range;
    if (!cp) return { recordIds: [subjectId, record.id], candidate };
    return { recordIds: [subjectId, record.id], candidate,
      codepointRange: cp,
      utf16Range: codepointRangeToUtf16Range(text, cp.start, cp.end) };
  });
  return { subjectId, attestationId: record.id, status, candidates };
}

function scopePresentation(relocations) {
  const decorations = []; const missing = [];
  for (const r of relocations) {
    if (r.status === ':missing') missing.push({ subjectId: r.subjectId, attestationId: r.attestationId });
    for (const c of r.candidates) {
      if (!c.utf16Range) continue;
      decorations.push({ status: r.status, range: c.utf16Range, recordIds: c.recordIds });
    }
  }
  return { decorations, missing,
    counts: {
      resolved: relocations.filter((r) => r.status === ':resolved').length,
      ambiguous: relocations.filter((r) => r.status === ':ambiguous').length,
      missing: missing.length,
    } };
}

function submitPresentation(response) {
  if (response.error) return { kind: 'refusal', error: response.error,
    recovery: response.error['error/recovery'] || [] };
  const ok = response.ok || {};
  if (ok['result/status'] === ':conflict') return { kind: 'conflict',
    conflictId: ok['conflict/id'], alternativeIds: ok['alternative/ids'] || [],
    headRevisionId: ok['head/revision-id'] };
  if (ok.relocation && ok.relocation['result/status'] === ':failed') return {
    kind: 'landed-relocation-failed', headRevisionId: ok['head/revision-id'],
    message: 'edit landed; markup not relocated', relocation: ok.relocation };
  return { kind: 'landed', status: ok['result/status'], headRevisionId: ok['head/revision-id'] };
}

async function loadHead(client, artifactId = ARTIFACT_ID) {
  const head = await client.revisionHead(artifactId);
  if (head.error) return head;
  const revisionId = head.ok['head/revision-id'];
  const revision = await client.revisionGet(revisionId);
  if (revision.error) return revision;
  return { ok: { revisionId, text: revision.ok['revision/text'] } };
}

async function loadScopes(client, manifestRevisionId, revisionId, text) {
  if (!manifestRevisionId) return { relocations: [], presentation: scopePresentation([]) };
  const outline = await client.formalF1Outline(manifestRevisionId);
  if (outline.error) return { error: outline.error };
  const relocations = [];
  for (const fragment of outline.ok['projection/fragments'] || []) {
    for (const id of fragment['record/ids'] || []) {
      const got = await client.entityGet(id);
      if (got.error) return { error: got.error };
      const r = relocationFromRecord(fragment['interpretation/id'], got.ok.record, revisionId, text);
      if (r) relocations.push(r);
    }
  }
  return { relocations, presentation: scopePresentation(relocations) };
}

async function loadAlternatives(client, ids) {
  const result = [];
  for (const id of ids) {
    const got = await client.entityGet(id);
    if (got.error) return { error: got.error };
    result.push({ id, text: got.ok.record['asserted-content']['result/text'], record: got.ok.record });
  }
  return { ok: result };
}

module.exports = { ARTIFACT_ID, minimalPatch, stableOperationId, relocationFromRecord,
  scopePresentation, submitPresentation, loadHead, loadScopes, loadAlternatives };
