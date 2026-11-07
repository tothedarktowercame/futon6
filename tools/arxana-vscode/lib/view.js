'use strict';
// Pure view models + webview HTML. Decorations and panels carry only
// response/candidate IDs (R4/D5): no client-side semantic truth, no
// auto-selection among ambiguous candidates (D5 client-boundary test 4).

function kw(s) { return String(s).replace(/^:/, ''); }

/** Build the resolve view model from an authority `ok` body.
 * selectionCodepoints (optional {start,end}) marks which candidates
 * contain the editor selection — it never selects one. */
function resolveView(okBody, selectionCodepoints) {
  const candidates = ((okBody && okBody['result/candidates']) || []).map((c, i) => {
    const range = c.location && c.location.range;
    const containsSelection = !!(selectionCodepoints && range
      && range.start <= selectionCodepoints.start
      && selectionCodepoints.end <= range.end);
    return {
      index: i,
      candidateId: c['candidate/id'],
      revisionId: c.location && c.location['artifact/revision-id'],
      unit: c.location ? kw(c.location.unit) : null,
      range,
      verification: kw(c.verification || ''),
      matchedTextDigest: c['matched-text-digest'],
      finderId: c['finder/id'],
      scoreComponents: c['score-components'] || {},
      containsSelection,
    };
  });
  const status = kw(okBody['result/status'] || '');
  const ambiguous = candidates.length > 1;
  return {
    object: 'resolution',
    status,
    artifactRevisionId: okBody['artifact/revision-id'],
    strategyRevisionId: okBody['strategy/revision-id'],
    semanticDigest: okBody['semantic/digest'],
    protocolSemanticsVersion: kw(okBody['protocol/semantics-version'] || ''),
    evaluatorVersion: okBody['evaluator/version'],
    evidence: (okBody['result/evidence'] || []).map((e) => ({
      finderId: e['finder/id'],
      finderType: kw(e['finder/type'] || ''),
      outcome: kw(e['evidence/outcome'] || ''),
      seq: e['evidence/seq'],
      stageDiagnostics: (e['evidence/diagnostics'] || []).map((d) => kw(d.stage || '')),
    })),
    evidenceDigests: okBody['result/evidence-digests'] || [],
    diagnostics: okBody['result/diagnostics'] || [],
    candidates,
    ambiguous,
    // The client displays ALL candidates and requires an explicit user
    // choice for navigation; this flag exists so the UI can say so.
    requiresChoice: ambiguous,
  };
}

/** Build the correspondence view model. Navigation targets are locator
 * projections: { kind:'source', logicalId, codepointOffset } or
 * { kind:'target', path, line } — derived only from fields the authority
 * returned plus the fixture manifest's path map. */
function correspondenceView(okBody, pathsByLogicalId) {
  const assertions = ((okBody && okBody['result/assertions']) || []).map((a) => {
    const source = a.source || {};
    const target = a.target || {};
    return {
      assertionId: a.id,
      direction: kw(a.direction || ''),
      verification: kw(a.verification || ''),
      cardinality: kw(a.cardinality || ''),
      method: (a.method || []).map(kw),
      producer: a.producer || {},
      source: {
        manifestationId: source['manifestation/id'],
        artifactRevisionId: source['artifact/revision-id'],
        locator: source.locator || {},
      },
      target: {
        manifestationId: target['manifestation/id'],
        artifactRevisionId: target['artifact/revision-id'],
        locator: target.locator || {},
      },
      evidence: a.evidence || [],
      // Navigation projections (client-owned, from returned locators):
      navigateSource: source.locator && source.locator.kind === ':codepoint-point'
        ? { kind: 'source', codepointOffset: source.locator.offset, revisionId: source['artifact/revision-id'] }
        : null,
      navigateTarget: target.locator && target.locator.kind === ':source-map-coordinate'
        ? { kind: 'target', line: target.locator.line, path: pathsByLogicalId && pathsByLogicalId['draft8.html'] }
        : null,
    };
  });
  return {
    object: 'correspondence',
    count: okBody['result/count'],
    semanticDigest: okBody['semantic/digest'],
    assertions,
  };
}

/** Typed-refusal view: object-qualified status words (D5), never bare. */
function refusalView(object, refusal) {
  return {
    object,
    refusal: {
      kind: refusal.kind || 'client/unknown',
      detail: refusal.detail || (refusal.message && { message: refusal.message }) || {},
    },
  };
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function dl(rows) {
  return '<dl>' + rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v === undefined || v === null ? '' : v)}</dd>`).join('') + '</dl>';
}

/** Webview HTML for a view model (any of the three above, or a
 * capabilities view). Pure: same model, same HTML. */
function renderHtml(title, model) {
  const sections = [];
  if (model.refusal) {
    sections.push(`<section><h2>${esc(model.object)}: refused</h2>`
      + `<p class="refusal">${esc(model.object)}: ${esc(model.refusal.kind)}</p>`
      + `<pre>${esc(JSON.stringify(model.refusal.detail, null, 2))}</pre></section>`);
  }
  if (model.capabilities) {
    const ops = Object.entries(model.capabilities.operations || {}).sort();
    sections.push('<section><h2>Advertised operations</h2><ul>'
      + ops.map(([op]) => `<li>${esc(op)}${model.missing.includes(op) ? ' <strong class="missing">MISSING</strong>' : ''}</li>`).join('')
      + '</ul>'
      + (model.ready ? '<p class="ok">scope/resolve and correspondence/query advertised.</p>'
        : `<p class="refusal">refused: operations not advertised: ${esc(model.missing.join(', '))}</p>`)
      + '</section>');
  }
  if (model.object === 'resolution' && !model.refusal) {
    sections.push('<section><h2>resolution: ' + esc(model.status) + '</h2>'
      + dl([
        ['artifact revision', model.artifactRevisionId],
        ['strategy revision', model.strategyRevisionId],
        ['semantic digest', model.semanticDigest],
        ['semantics version', model.protocolSemanticsVersion],
        ['evaluator', model.evaluatorVersion],
      ])
      + '<h3>Evidence (authority)</h3><ul>'
      + model.evidence.map((e) => `<li>${esc(e.finderType)} seq ${esc(e.seq)} → ${esc(e.outcome)} (stages: ${esc(e.stageDiagnostics.join(', '))})</li>`).join('')
      + '</ul>'
      + `<h3>Candidates (${model.candidates.length})</h3>`
      + (model.requiresChoice ? '<p class="ambiguous">resolution: ambiguous — all candidates shown; navigation requires an explicit choice.</p>' : '')
      + '<ol>' + model.candidates.map((c) => `<li><code>${esc(c.candidateId)}</code> ${esc(c.verification)} cp ${esc(c.range && c.range.start)}–${esc(c.range && c.range.end)}${c.containsSelection ? ' <em>(contains selection)</em>' : ''}<br>matched-text digest ${esc(c.matchedTextDigest)}</li>`).join('') + '</ol>'
      + '</section>');
  }
  if (model.object === 'correspondence' && !model.refusal) {
    sections.push('<section><h2>correspondence</h2>'
      + dl([['assertions', model.count], ['semantic digest', model.semanticDigest]])
      + model.assertions.map((a) => '<article><h3>' + esc(a.assertionId) + '</h3>'
        + dl([
          ['direction', a.direction],
          ['verification (partial/status)', a.verification],
          ['cardinality', a.cardinality],
          ['method', a.method.join(', ')],
          ['producer', `${a.producer.tool || ''} ${a.producer.version || ''} build ${a.producer['build/id'] || ''}`],
          ['source manifestation', a.source.manifestationId],
          ['source revision', a.source.artifactRevisionId],
          ['source locator', JSON.stringify(a.source.locator)],
          ['target manifestation', a.target.manifestationId],
          ['target revision', a.target.artifactRevisionId],
          ['target locator', JSON.stringify(a.target.locator)],
          ['evidence', (a.evidence || []).map((e) => `${kw(e.type)}: ${e.value || ''}`).join(' | ')],
        ])
        + (a.navigateSource ? `<p class="nav">navigate source: codepoint ${esc(a.navigateSource.codepointOffset)}</p>` : '')
        + (a.navigateTarget ? `<p class="nav">navigate target: ${esc(a.navigateTarget.path)} line ${esc(1 + a.navigateTarget.line)}</p>` : '')
        + '</article>').join('')
      + '</section>');
  }
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${esc(title)}</title><style>
body{font-family:sans-serif;margin:1em}dt{font-weight:bold;display:inline-block;margin-right:.5em}dd{display:inline;margin-right:1.5em}
code,pre{background:#f4f4f4}.refusal{color:#a00}.ok{color:#070}.ambiguous{color:#a60}article{border-top:1px solid #ccc;padding-top:.5em}
</style></head><body><h1>${esc(title)}</h1>${sections.join('')}</body></html>`;
}

module.exports = { resolveView, correspondenceView, refusalView, renderHtml };
