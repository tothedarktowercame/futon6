#!/usr/bin/env node
// futon-relay-channel: push a seat's Agency Clink inbox into a running Claude
// Code session as channel messages (stage 2, 2026-11-10).
//
// Agency writes each bell for a pull-only seat (:delivery-mode :inbox) to
// ~/.claude/agency-inbox/<seat>/<job-id>.json (atomic .tmp + move). This MCP
// server declares experimental["claude/channel"] and, for each delivery it has
// not pushed in this process, sends notifications/claude/channel with the
// prompt as content. It never acks: the session acks after answering
// (clink-inbox.py ack), which moves the file to consumed/. So an unacked
// delivery is pushed again after a session restart, and only then.
//
// Shapes from the Claude Code 2.1.322 binary: capability check
// capabilities.experimental["claude/channel"]; notification
// {method:"notifications/claude/channel", params:{content, meta?: {string: string}}}.
// Channels register only in the interactive TUI, started with
// --dangerously-load-development-channels server:futon-relay-channel.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function inboxDir(seat, home = os.homedir()) {
  return path.join(home, '.claude', 'agency-inbox', seat);
}

// One pass over the inbox. `state` persists between passes in one process:
// pushed job ids, and malformed files keyed by name+mtime+size so a file that
// changes is tried again. Returns what happened, for the log and the tests.
export function scanOnce({ dir, seat, state, send, log }) {
  const out = { pushed: [], skipped: [], failed: [] };
  let names;
  try { names = fs.readdirSync(dir); } catch (e) { return out; }
  for (const name of names.filter((n) => n.endsWith('.json')).sort()) {
    const file = path.join(dir, name);
    let st;
    try { st = fs.statSync(file); } catch { continue; }
    if (!st.isFile()) continue;
    const jobId = name.slice(0, -5);
    if (state.pushed.has(jobId)) continue;
    const sig = `${name}:${st.mtimeMs}:${st.size}`;
    if (state.bad.has(sig)) continue;
    let d;
    try {
      d = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (!d || typeof d.prompt !== 'string') throw new Error('no prompt field');
    } catch (e) {
      state.bad.add(sig);
      log(`skip malformed ${file}: ${e.message}`);
      out.skipped.push(jobId);
      continue;
    }
    const id = String(d['job-id'] || jobId);
    try {
      send(render(d, id, seat));
      state.pushed.add(jobId);
      out.pushed.push(id);
      log(`pushed ${id} from ${d.from || '?'}`);
    } catch (e) {
      log(`send failed for ${id}: ${e.message}`);
      out.failed.push(id);
    }
  }
  return out;
}

export function render(d, id, seat) {
  const from = String(d.from || 'unknown');
  const header = [
    `[Agency bell ${id} from ${from}` +
      (d['bell-type'] ? `, ${d['bell-type']}` : '') +
      (d['in-reply-to'] && d['in-reply-to'] !== 'None' ? `, in reply to ${d['in-reply-to']}` : '') + ']',
    `When you have answered, ack it with your answer so it goes back to ${from}: ` +
      `python3 ~/code/futon3c/scripts/clink-inbox.py --agent ${seat} ack ${id} --result-file <file with your answer>`,
    '',
  ].join('\n');
  const meta = { 'job-id': id, from };
  if (d['ack-url']) meta['ack-url'] = String(d['ack-url']);
  return { content: header + d.prompt, meta };
}

// ------------------------------------------------------------- MCP stdio

function main() {
  const seat = process.env.FUTON_RELAY_SEAT || 'claude-cs';
  const dir = process.env.FUTON_RELAY_INBOX || inboxDir(seat);
  const logFile = process.env.FUTON_RELAY_LOG ||
    path.join(os.homedir(), '.local', 'state', 'futon3c', 'relay', `channel-${seat}.log`);
  try { fs.mkdirSync(path.dirname(logFile), { recursive: true }); } catch {}
  const log = (s) => { try { fs.appendFileSync(logFile, `${new Date().toISOString()} [${process.pid}] ${s}\n`); } catch {} };
  const write = (m) => process.stdout.write(JSON.stringify(m) + '\n');
  const send = (params) => write({ jsonrpc: '2.0', method: 'notifications/claude/channel', params });
  const state = { pushed: new Set(), bad: new Set() };
  let started = false;
  const start = () => {
    if (started) return;
    started = true;
    try { fs.mkdirSync(dir, { recursive: true }); } catch {}
    try { fs.mkdirSync(dir, { recursive: true }); } catch {}
    log(`watching ${dir}`);
    const tick = () => scanOnce({ dir, seat, state, send, log });
    tick();
    try { fs.watch(dir, () => setTimeout(tick, 100)); } catch (e) { log(`fs.watch unavailable: ${e.message}`); }
    setInterval(tick, +(process.env.FUTON_RELAY_POLL_MS || 3000)).unref?.();
  };
  let buf = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', (chunk) => {
    buf += chunk;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line) continue;
      let m;
      try { m = JSON.parse(line); } catch { continue; }
      if (m.method === 'initialize') {
        write({ jsonrpc: '2.0', id: m.id, result: {
          // Echo the client's revision: the probe showed this negotiates the
          // "legacy" era, which keeps the unsolicited-notification path.
          protocolVersion: m.params?.protocolVersion,
          capabilities: { tools: {}, experimental: { 'claude/channel': {} } },
          serverInfo: { name: 'futon-relay-channel', version: '0.2.0' } } });
      } else if (m.method === 'notifications/initialized') {
        setTimeout(start, 1000);
      } else if (m.method === 'tools/list') {
        write({ jsonrpc: '2.0', id: m.id, result: { tools: [] } });
      } else if (m.id !== undefined) {
        write({ jsonrpc: '2.0', id: m.id, error: { code: -32601, message: 'method not found' } });
      }
    }
  });
  process.stdin.on('end', () => process.exit(0));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) main();
