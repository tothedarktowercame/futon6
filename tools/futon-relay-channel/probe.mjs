#!/usr/bin/env node
// Stage-1 feasibility probe: a minimal stdio MCP server that declares the
// claude/channel capability and pushes one channel notification a few seconds
// after initialization. Shapes taken from the Claude Code 2.1.322 binary:
//   capability check: capabilities.experimental["claude/channel"] (fn cPt)
//   notification:     {method:"notifications/claude/channel",
//                      params:{content:string, meta?:Record<string,string>}}
// Plain JSON-RPC over newline-delimited stdio; no SDK dependency.
import { appendFileSync } from 'node:fs';
const LOG = process.env.PROBE_LOG || '/dev/null';
const log = (s) => appendFileSync(LOG, new Date().toISOString() + ' ' + s + '\n');
const send = (m) => { const s = JSON.stringify(m); log('> ' + s); process.stdout.write(s + '\n'); };
let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
    if (!line) continue;
    log('< ' + line);
    let m; try { m = JSON.parse(line); } catch { continue; }
    if (m.method === 'initialize') {
      send({ jsonrpc: '2.0', id: m.id, result: {
        protocolVersion: process.env.PROBE_PROTOCOL || m.params.protocolVersion,
        capabilities: { tools: {}, experimental: { 'claude/channel': {} } },
        serverInfo: { name: 'futon-relay-channel-probe', version: '0.0.1' } } });
    } else if (m.method === 'notifications/initialized') {
      setTimeout(() => send({ jsonrpc: '2.0', method: 'notifications/claude/channel',
        params: { content: 'futon-relay channel probe ' + new Date().toISOString(),
                  meta: { source: 'futon-relay-probe' } } }), +(process.env.PROBE_DELAY_MS || 5000));
    } else if (m.method === 'tools/list') {
      send({ jsonrpc: '2.0', id: m.id, result: { tools: [] } });
    } else if (m.id !== undefined) {
      send({ jsonrpc: '2.0', id: m.id, error: { code: -32601, message: 'method not found' } });
    }
  }
});
