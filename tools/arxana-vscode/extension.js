'use strict';
// Arxana code-server demo extension (T-codeserver-arxana-demo, M-jvm-arxana).
// Projection only: consumes arxana.v1 (capabilities/handshake, scope/resolve,
// correspondence/query) from the isolated loopback JVM surface. No
// futon1/futon1b calls, no arxana:* id minting, no local first-candidate
// selection, no store writes.

const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const { ArxanaClient, TypedRefusal } = require('./lib/client');
const { matchRevision } = require('./lib/digest');
const {
  utf16RangeToCodepointRange, codepointRangeToUtf16Range, MidSurrogateError,
} = require('./lib/utf16');
const { resolveView, correspondenceView, refusalView, renderHtml } = require('./lib/view');
const { navigationChoice } = require('./lib/navigation');
const live = require('./lib/live');
const apmweb = require('./lib/apmweb');
const { utf16ToCodepoint } = require('./lib/utf16');
const { revisionIdFromSha } = require('./lib/digest');

let panel = null;

function fixture() {
  return JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'draft8-v1.json'), 'utf8'));
}

function client() {
  const url = vscode.workspace.getConfiguration('arxana').get('serviceUrl')
    || fixture()['service']['default-url'];
  return new ArxanaClient(url);
}

function showPanel(title) {
  if (!panel) {
    panel = vscode.window.createWebviewPanel('arxana', title, vscode.ViewColumn.Beside, {});
    panel.onDidDispose(() => { panel = null; });
  } else {
    panel.title = title;
    panel.reveal();
  }
  return panel;
}

function display(title, model) {
  showPanel(title).webview.html = renderHtml(title, model);
}

function revisionsByLogicalId() {
  const out = {};
  for (const [logical, a] of Object.entries(fixture().artifacts)) out[logical] = a['revision-id'];
  return out;
}

function pathsByLogicalId() {
  const out = {};
  for (const [logical, a] of Object.entries(fixture().artifacts)) out[logical] = a.path;
  return out;
}

/** Digest-match the active document against fixture revisions (R4: position
 * conversion only against digest-matched revision bytes). */
function matchActiveDocument(editor) {
  const bytes = Buffer.from(editor.document.getText(), 'utf8');
  const m = matchRevision(bytes, revisionsByLogicalId());
  if (m.kind !== 'matched') {
    throw new TypedRefusal('client/revision-unmatched', {
      sha256: m.sha256,
      hint: 'open one of the fixture sources listed in fixtures/draft8-v1.json unmodified',
    });
  }
  return m;
}

function activate(context) {
  const reg = (id, fn) => context.subscriptions.push(vscode.commands.registerCommand(id, fn));

  reg('arxana.showCapabilities', async () => {
    try {
      const caps = await client().capabilities();
      display('Arxana capabilities', caps.error
        ? refusalView('capability', caps.error)
        : { capabilities: caps.ok, missing: caps.missing, ready: caps.ready });
    } catch (e) {
      display('Arxana capabilities', refusalView('capability', e));
    }
  });

  reg('arxana.resolveScope', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) { vscode.window.showWarningMessage('Arxana: open a fixture source document first'); return; }
    let m;
    let selectionCodepoints = null;
    try {
      m = matchActiveDocument(editor);
      const text = editor.document.getText();
      const s = editor.selection;
      // Selection is converted only against the digest-matched revision bytes.
      selectionCodepoints = utf16RangeToCodepointRange(text, editor.document.offsetAt(s.start), editor.document.offsetAt(s.end));
    } catch (e) {
      display('Arxana resolve', refusalView('resolution', e));
      return;
    }
    try {
      const fx = fixture();
      const { ok, error } = await client().scopeResolve({
        'artifact/revision-id': m.revisionId,
        'strategy/revision-id': fx['strategy/revision-id'],
        'subject/id': fx['subject/id'],
      });
      if (error) { display('Arxana resolve', refusalView('resolution', error)); return; }
      const model = resolveView(ok, selectionCodepoints);
      display('Arxana resolve', model);
      paintCandidates(editor, model);
    } catch (e) {
      display('Arxana resolve', refusalView('resolution', e));
    }
  });

  reg('arxana.correspondence', async () => {
    try {
      const { ok, error } = await client().correspondenceQuery({ 'subject/id': fixture()['subject/id'] });
      if (error) { display('Arxana correspondence', refusalView('correspondence', error)); return; }
      display('Arxana correspondence', correspondenceView(ok, pathsByLogicalId()));
    } catch (e) {
      display('Arxana correspondence', refusalView('correspondence', e));
    }
  });

  // Navigation from a correspondence target back to the source, and source →
  // rendered target, as far as the fixture's locators permit.
  reg('arxana.openCorrespondenceTarget', async () => {
    try {
      const { ok, error } = await client().correspondenceQuery({ 'subject/id': fixture()['subject/id'] });
      if (error) { display('Arxana correspondence', refusalView('correspondence', error)); return; }
      const view = correspondenceView(ok, pathsByLogicalId());
      // D4 ambiguity preservation: never silently select the first
      // assertion (see lib/navigation.js navigationChoice).
      let choice = navigationChoice(view.assertions);
      if (choice.choose) {
        const picked = await vscode.window.showQuickPick(
          choice.choose.map((id) => ({ label: String(id) })),
          { placeHolder: 'Multiple correspondence assertions: choose one to open (no implicit first selection)' });
        choice = navigationChoice(view.assertions, picked ? picked.label : null);
      }
      if (choice.refusal) {
        display('Arxana correspondence', refusalView('navigation', new TypedRefusal(choice.refusal.kind, choice.refusal.detail)));
        return;
      }
      const a = choice.assertion;
      if (!a || !a.navigateTarget) {
        display('Arxana correspondence', refusalView('navigation', new TypedRefusal('client/locator-unsupported', {})));
        return;
      }
      const doc = await vscode.workspace.openTextDocument(a.navigateTarget.path);
      const ed = await vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
      const pos = ed.document.positionAt(0);
      const line = Math.min(a.navigateTarget.line, ed.document.lineCount - 1);
      ed.selection = new vscode.Selection(line, 0, line, 0);
      ed.revealRange(new vscode.Range(line, 0, line, 0), vscode.TextEditorRevealType.InCenter);
      vscode.window.showInformationMessage(`Arxana: revealed correspondence target (line ${1 + a.navigateTarget.line}); locator is a projection, not identity`);
    } catch (e) {
      display('Arxana correspondence', refusalView('navigation', e));
    }
  });

  reg('arxana.goToCandidate', async (candidate) => {
    // Explicit user choice only (never first-candidate auto-selection).
    const editor = vscode.window.activeTextEditor;
    if (!editor || !candidate) return;
    try {
      const m = matchActiveDocument(editor);
      if (m.revisionId !== candidate.revisionId) {
        throw new TypedRefusal('client/revision-unmatched', { active: m.revisionId });
      }
      const text = editor.document.getText();
      const u = codepointRangeToUtf16Range(text, candidate.range.start, candidate.range.end);
      const startPos = editor.document.positionAt(u.start);
      const endPos = editor.document.positionAt(u.end);
      editor.selection = new vscode.Selection(startPos, endPos);
      editor.revealRange(new vscode.Range(startPos, endPos), vscode.TextEditorRevealType.InCenter);
    } catch (e) {
      display('Arxana candidate', refusalView('navigation', e));
    }
  });

  // ── APM two-pane demo (M-jvm-arxana increment, 2026-11-07) ─────────────
  // Left pane: the authored .tex. Right pane: the RENDERED LaTeXML-oxide
  // page in a webview. Cursor movement resolves the smallest enclosing
  // REGISTERED scope via :scope/at-position, then every corresponding
  // target is highlighted in both panes. Debounced + sequence-gated so
  // rapid movement cannot repaint an older scope; typed refusals clear.
  {
    const apmDir = path.join(__dirname, 'fixtures', 'apm');
    const apmManifest = JSON.parse(fs.readFileSync(path.join(apmDir, 'manifest.json'), 'utf8'));
    const texName = 'apm-a93J02.tex';
    const texSha = apmManifest.artifacts[texName]['sha256'];
    const texRevisionId = revisionIdFromSha(texSha);
    let apmPanel = null;
    let apmEditor = null;
    let lastPresentation = null;
    let matchedDocumentVersion = null;
    let matchedRevision = null;
    const gate = live.sequenceGate();
    const debounce = live.debouncer(75);
    const log = vscode.window.createOutputChannel('Arxana APM demo');
    const webviewDelivery = live.latestDelivery((message) => {
      if (apmPanel) apmPanel.webview.postMessage(message);
    });

    const scopeDecoration = vscode.window.createTextEditorDecorationType({
      backgroundColor: 'rgba(152,195,121,0.18)',
      border: '1px solid rgba(152,195,121,0.5)',
    });
    const enclosingDecoration = vscode.window.createTextEditorDecorationType({
      borderBottom: '1px dashed rgba(209,154,102,0.6)',
    });

    function clearSourceHighlights() {
      if (apmEditor) {
        apmEditor.setDecorations(scopeDecoration, []);
        apmEditor.setDecorations(enclosingDecoration, []);
      }
    }

    function paint(presentation) {
      lastPresentation = presentation;
      clearSourceHighlights();
      if (apmPanel) {
        webviewDelivery.publish({
          type: 'arxana/highlight',
          status: presentation.status,
          targets: presentation.targets,
          refusal: presentation.refusal || null,
        });
      }
      if (presentation.status === 'cleared') {
        if (presentation.refusal) {
          log.appendLine(`cleared: typed refusal ${presentation.refusal.kind}`
            + (presentation.refusal.category ? ` (${presentation.refusal.category})` : ''));
        }
        return;
      }
      if (!apmEditor || !presentation.sourceRange) return;
      const text = apmEditor.document.getText();
      try {
        const inner = codepointRangeToUtf16Range(text, presentation.sourceRange.start, presentation.sourceRange.end);
        apmEditor.setDecorations(scopeDecoration, [{
          range: new vscode.Range(apmEditor.document.positionAt(inner.start), apmEditor.document.positionAt(inner.end)),
          hoverMessage: `arxana innermost scope (${presentation.status}) digest ${presentation.digest || 'n/a'}`,
        }]);
        const enclosing = presentation.targets.filter((t) => !t.innermost && t.range);
        apmEditor.setDecorations(enclosingDecoration, enclosing.map((t) => {
          const u = codepointRangeToUtf16Range(text, t.range.start, t.range.end);
          return {
            range: new vscode.Range(apmEditor.document.positionAt(u.start), apmEditor.document.positionAt(u.end)),
            hoverMessage: `arxana enclosing scope ${t.subjectId} (${t.verification}) ${t.assertionId}`,
          };
        }));
      } catch (e) {
        log.appendLine(`paint refused: ${e && e.message}`);
      }
    }

    async function resolveAt(editor) {
      const seq = gate.next();
      try {
        // Digest-match the live document against the pinned revision first:
        // position conversion is legal only against matched bytes.
        if (matchedDocumentVersion !== editor.document.version) {
          const bytes = Buffer.from(editor.document.getText(), 'utf8');
          matchedRevision = matchRevision(bytes, { [texName]: texRevisionId });
          matchedDocumentVersion = editor.document.version;
        }
        const m = matchedRevision;
        if (m.kind !== 'matched') {
          paint({ status: 'cleared', sourceRange: null, targets: [],
                  refusal: { kind: 'client/revision-unmatched', sha256: m.sha256 } });
          return;
        }
        const text = editor.document.getText();
        const offset = editor.document.offsetAt(editor.selection.active);
        const codepoint = utf16ToCodepoint(text, offset);
        const response = await client().scopeAtPosition({
          'artifact/revision-id': texRevisionId,
          'position/codepoint': codepoint,
        });
        if (!gate.admit(seq)) return; // stale/out-of-order: repaint nothing
        paint(live.presentationFor(response));
      } catch (e) {
        if (!gate.admit(seq)) return;
        paint(live.presentationFor({ error: { 'error/code': (e && e.kind) || 'client/unknown' } }));
      }
    }

    reg('arxana.apmDemo', async () => {
      const doc = await vscode.workspace.openTextDocument(path.join(apmDir, texName));
      apmEditor = await vscode.window.showTextDocument(doc, vscode.ViewColumn.One);
      if (!apmPanel) {
        apmPanel = vscode.window.createWebviewPanel('arxana-apm', 'APM a93J02 — rendered (LaTeXML-oxide)',
          vscode.ViewColumn.Beside, { enableScripts: true });
        apmPanel.webview.html = apmweb.apmWebviewHtml(apmDir, apmPanel.webview.cspSource);
        apmPanel.webview.onDidReceiveMessage((msg) => {
          if (!msg) return;
          if (msg.type === 'arxana/ready') {
            webviewDelivery.markReady();
            if (apmEditor) resolveAt(apmEditor);
            return;
          }
          if (msg.type === 'arxana/targets-missing') {
            log.appendLine(`webview: target elements absent from rendered page: ${msg.domIds.join(', ')}`);
          }
          if (msg.type === 'arxana/target-click' && lastPresentation) {
            // Target→source: reveal the source range of the SAME authority
            // assertion that produced this highlight (presentation-only).
            const t = lastPresentation.targets.find((x) => x.domId === msg.domId);
            if (t && t.range && apmEditor) {
              const text = apmEditor.document.getText();
              const u = codepointRangeToUtf16Range(text, t.range.start, t.range.end);
              const start = apmEditor.document.positionAt(u.start);
              const end = apmEditor.document.positionAt(u.end);
              apmEditor.selection = new vscode.Selection(start, end);
              apmEditor.revealRange(new vscode.Range(start, end), vscode.TextEditorRevealType.InCenter);
            }
          }
        });
        apmPanel.onDidDispose(() => {
          apmPanel = null;
          webviewDelivery.reset();
          clearSourceHighlights();
        });
      } else {
        apmPanel.reveal();
      }
      vscode.window.showInformationMessage('Arxana APM demo: move the cursor in the .tex pane; scopes resolve through the service');
    });

    context.subscriptions.push(vscode.window.onDidChangeTextEditorSelection((ev) => {
      if (apmPanel && ev.textEditor === apmEditor) debounce.push(() => resolveAt(apmEditor));
    }));
  }
}

/** Decorations carry response/candidate IDs only (D5): the hover message is
 * the candidate id and verification, never interpreted semantics. */
let candidateDecoration = null;
function paintCandidates(editor, model) {
  if (!candidateDecoration) {
    candidateDecoration = vscode.window.createTextEditorDecorationType({
      backgroundColor: 'rgba(0,120,215,0.15)',
      border: '1px solid rgba(0,120,215,0.4)',
    });
  }
  const decorationType = candidateDecoration;
  const text = editor.document.getText();
  const options = [];
  for (const c of model.candidates) {
    if (!c.range) continue;
    try {
      const u = codepointRangeToUtf16Range(text, c.range.start, c.range.end);
      options.push({
        range: new vscode.Range(editor.document.positionAt(u.start), editor.document.positionAt(u.end)),
        hoverMessage: `candidate ${c.candidateId} (${c.verification}) — authority response id, not local truth`,
      });
    } catch (_) { /* candidate range outside this document revision: skip painting */ }
  }
  editor.setDecorations(decorationType, options);
}

function deactivate() {}

module.exports = { activate, deactivate };
