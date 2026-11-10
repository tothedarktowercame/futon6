'use strict';
// futon relay (Joe, 2026-11-10): "let's make a little Javascript hack that
// relays the code directly and copy-pastes it into the extension, that way it
// will just come through as a normal turn from me".
//
// Emacs (futon4a-subprompt.el) writes each ㊡ subprompt, composed as
// "MOJIRA ASK: ... QUOTED BACKGROUND CONTEXT: ...", to
//   ~/.local/state/futon3c/relay/<seat>/<time>.txt
// This extension checks that folder, claims each file by renaming it (so two
// code-server windows never both take it), and opens the seat's Claude Code
// conversation with the text as its prompt, through Claude Code's own
// `claude-vscode.editor.open(sessionId, initialPrompt)`.  The seat's session id
// is the one Agency resumes: ~/.local/state/futon3c/session-id-<seat>.
// Used files move to <seat>/done/.

const vscode = require('vscode');
const fs = require('fs');
const os = require('os');
const path = require('path');

const STATE = path.join(os.homedir(), '.local', 'state', 'futon3c');

function cfg() {
  const c = vscode.workspace.getConfiguration('futonRelay');
  return { seat: c.get('seat', 'claude-cs'), intervalMs: c.get('intervalMs', 1500) };
}

function sessionId(seat) {
  try {
    const s = fs.readFileSync(path.join(STATE, `session-id-${seat}`), 'utf8').trim();
    return s || undefined;
  } catch (_) {
    return undefined;
  }
}

let busy = false;

async function check(log) {
  if (busy) return;
  busy = true;
  try {
    const { seat } = cfg();
    const dir = path.join(STATE, 'relay', seat);
    let names;
    try {
      names = fs.readdirSync(dir).filter((n) => n.endsWith('.txt')).sort();
    } catch (_) {
      return;
    }
    for (const name of names) {
      const claimed = path.join(dir, `${name}.claimed-${process.pid}`);
      try {
        fs.renameSync(path.join(dir, name), claimed);
      } catch (_) {
        continue; // another window took it
      }
      const text = fs.readFileSync(claimed, 'utf8');
      const sid = sessionId(seat);
      await vscode.commands.executeCommand('claude-vscode.editor.open', sid, text);
      fs.mkdirSync(path.join(dir, 'done'), { recursive: true });
      fs.renameSync(claimed, path.join(dir, 'done', name));
      log.appendLine(`${new Date().toISOString()} relayed ${name} into ${seat} (session ${sid || 'new'})`);
      vscode.window.setStatusBarMessage(`futon relay: prompt for ${seat} is in the Claude input`, 8000);
    }
  } catch (err) {
    log.appendLine(`${new Date().toISOString()} error: ${err && err.stack || err}`);
  } finally {
    busy = false;
  }
}

function activate(context) {
  const log = vscode.window.createOutputChannel('futon relay');
  const timer = setInterval(() => check(log), cfg().intervalMs);
  context.subscriptions.push(
    log,
    { dispose: () => clearInterval(timer) },
    vscode.commands.registerCommand('futonRelay.checkNow', () => check(log)));
  log.appendLine(`${new Date().toISOString()} watching ${path.join(STATE, 'relay', cfg().seat)}`);
}

function deactivate() {}

module.exports = { activate, deactivate };
