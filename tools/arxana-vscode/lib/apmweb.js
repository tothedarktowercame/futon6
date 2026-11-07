'use strict';
// Webview HTML builder for the APM two-pane demo: the RENDERED
// LaTeXML-oxide page (never raw HTML source) with the fixture's
// presentation stylesheets inlined. Presentation-resource identity note:
// LaTeXML.css / ltx-article.css are presentation resources pinned by the
// manifest sha256 alongside the semantic artifacts; inlining them into the
// webview document keeps the webview free of external resource loading.

const fs = require('fs');
const path = require('path');

const HIGHLIGHT_CSS = `
<style id="arxana-demo-highlighting">
.arxana-active { outline: 3px solid #d19a66; outline-offset: 2px; background: rgba(209,154,102,0.15); }
.arxana-innermost { outline-color: #98c379; }
.arxana-ambiguous-banner { position: fixed; top: 0; left: 0; right: 0;
  background: #2d2a3b; color: #e6edf7; font: 12px system-ui; padding: 4px 8px;
  border-bottom: 1px solid #d19a66; z-index: 99; }
#arxana-tour { position: fixed; left: 8px; right: 8px; bottom: 8px; z-index: 100;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  padding: 8px; border: 1px solid #98c379; border-radius: 8px;
  background: rgba(32,35,42,0.96); color: #e6edf7; font: 14px system-ui; }
#arxana-tour button { min-width: 92px; min-height: 44px; border: 1px solid #98c379;
  border-radius: 6px; background: #343a46; color: #fff; font: 600 14px system-ui; }
#arxana-tour-status { min-width: 7em; text-align: center; }
body { padding-bottom: 72px; }
</style>`;

const WEBVIEW_SCRIPT = `
<script>
'use strict';
const vscode = acquireVsCodeApi();
const tour = document.createElement('div');
tour.id = 'arxana-tour';
tour.innerHTML = '<button id="arxana-tour-prev" type="button">◀ Previous</button>'
  + '<span id="arxana-tour-status">Loading scopes…</span>'
  + '<button id="arxana-tour-next" type="button">Next ▶</button>';
document.body.appendChild(tour);
document.getElementById('arxana-tour-prev').addEventListener('click', () =>
  vscode.postMessage({ type: 'arxana/tour', direction: 'previous' }));
document.getElementById('arxana-tour-next').addEventListener('click', () =>
  vscode.postMessage({ type: 'arxana/tour', direction: 'next' }));
function clearHighlights() {
  document.querySelectorAll('.arxana-active').forEach((el) =>
    el.classList.remove('arxana-active', 'arxana-innermost'));
  const b = document.getElementById('arxana-ambiguous-banner');
  if (b) b.remove();
}
window.addEventListener('message', (event) => {
  const msg = event.data;
  if (!msg || msg.type !== 'arxana/highlight') return;
  if (msg.tour) {
    document.getElementById('arxana-tour-status').textContent =
      msg.tour.position + ' / ' + msg.tour.count;
  }
  clearHighlights();
  if (msg.status === 'cleared') return;   // stale/missing/typed refusal: old highlights cleared
  let first = null;
  const missing = [];
  for (const t of msg.targets || []) {
    const el = t.domId ? document.getElementById(t.domId) : null;
    if (!el) { missing.push(t.domId); continue; }
    el.classList.add('arxana-active');
    if (t.innermost) el.classList.add('arxana-innermost');
    if (!first) first = el;
  }
  if (msg.status === 'ambiguous') {
    const banner = document.createElement('div');
    banner.id = 'arxana-ambiguous-banner';
    banner.textContent = 'Arxana: ambiguous scope — all correspondence assertions shown: '
      + (msg.targets || []).map((t) => t.assertionId).join(', ');
    document.body.appendChild(banner);
  }
  if (first) first.scrollIntoView({ block: 'center' });
  if (missing.length) vscode.postMessage({ type: 'arxana/targets-missing', domIds: missing });
});
// Target→source navigation: click a highlighted element posts its scope
// identity back; the extension reveals the source range of the SAME
// authority assertion (presentation-only projection, no new identities).
document.addEventListener('click', (ev) => {
  const el = ev.target && ev.target.closest ? ev.target.closest('.arxana-active') : null;
  if (!el) return;
  vscode.postMessage({ type: 'arxana/target-click', domId: el.id });
});
// The extension may resolve a scope while this document is still parsing.
// Declare the listener ready only after both handlers above are installed.
vscode.postMessage({ type: 'arxana/ready' });
</script>`;

/** Build the full webview document. `htmlPath` etc. resolve inside the
 * extension's committed APM fixture directory. */
function apmWebviewHtml(fixtureDir, cspSource) {
  const html = fs.readFileSync(path.join(fixtureDir, 'index.html'), 'utf8');
  const latexmlCss = fs.readFileSync(path.join(fixtureDir, 'LaTeXML.css'), 'utf8');
  const articleCss = fs.readFileSync(path.join(fixtureDir, 'ltx-article.css'), 'utf8');
  // Inline the two stylesheets where the <link> tags were, drop the
  // highlighted-PDF demo style block (presentation artifact of the public
  // reference page, not part of this demo), inject our script before
  // </body>, and add a strict CSP (no external loads, inline styles/scripts
  // only from this document).
  return html
    .replace(/<link rel="stylesheet" href="LaTeXML.css"[^>]*>/,
             '<style>\n' + latexmlCss + '\n</style>')
    .replace(/<link rel="stylesheet" href="ltx-article.css"[^>]*>/,
             '<style>\n' + articleCss + '\n</style>')
    .replace(/<style id="lean-proofread-highlighting">[\s\S]*?<\/style>/, '')
    .replace('</head>',
      '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
      + 'style-src \'unsafe-inline\'; script-src \'unsafe-inline\'; '
      + 'font-src ' + cspSource + '; img-src ' + cspSource + ' data:;"></head>')
    .replace(/<\/body>/, HIGHLIGHT_CSS + WEBVIEW_SCRIPT + '</body>');
}

module.exports = { apmWebviewHtml };
