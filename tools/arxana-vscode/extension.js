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
