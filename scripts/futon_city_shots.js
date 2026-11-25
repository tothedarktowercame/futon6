// futon_city_shots.js — screenshots + readouts for the Futon City visual essay.
//
// Usage: node futon_city_shots.js <base-url> <out-dir>
//   base-url: where the pages are served (e.g. file:///var/www/.../wip or https://...)
//   out-dir:  writes <id>.jpg per shot and shots.json (readouts, links)
// Shot list: SHOTS below; each opens a published page at a deep link (the same
// hash the essay links to) and optionally crops by setting the SVG viewBox for
// the picture only.
// Machine config, not repo config: set PLAYWRIGHT_PATH to the playwright
// module directory, or have playwright resolvable from here.
const PW = process.env.PLAYWRIGHT_PATH || 'playwright';
const { chromium } = require(PW);
const fs = require('fs');
const path = require('path');

const [BASE, OUT] = process.argv.slice(2);
if (!BASE || !OUT) { console.error('usage: futon_city_shots.js <base-url> <out-dir>'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });

// id, page, hash, readout selector, optional crop viewBox [x y w h]
const SHOTS = [
  ['baseline', 'mission-efe-field.html', '', null, null],
  ['tide-low', 'archipelago-A-tide.html', '#cutoff=2026-01-26', '#tide-readout', null],
  ['tide-jun', 'archipelago-A-tide.html', '#cutoff=2026-06-01', '#tide-readout', null],
  ['tide-sep', 'archipelago-A-tide.html', '#cutoff=2026-09-01', '#tide-readout', null],
  ['tide-nov', 'archipelago-A-tide.html', '#cutoff=2026-11-01', '#tide-readout', null],
  ['roads-latest', 'archipelago-B-illumination.html', '', '#illum-readout', null],
  ['roads-jun', 'archipelago-B-illumination.html', '#window=2026-06-01', '#illum-readout', null],
  ['roads-stop', 'archipelago-B-illumination.html', '#focus=futon-theory/stop-the-line', '#illum-focus-info', null],
  ['c-sep', 'archipelago-C.html', '#cutoff=2026-09-01', '#c-readout', null],
  ['c-sep-interior', 'archipelago-C.html', '#cutoff=2026-09-01&interior=1', '#c-readout', null],
  ['c-sep-stop', 'archipelago-C.html', '#cutoff=2026-09-01&focus=futon-theory/stop-the-line', '#c-readout', null],
  ['d-work-far', 'archipelago-D.html', '#cutoff=2026-09-01&style=work-first&zoom=far', '#c-readout', null],
  ['d-terrain', 'archipelago-D.html', '#cutoff=2026-09-01&style=terrain-first', '#c-readout', null],
  ['d-agents', 'archipelago-D.html', '#cutoff=2026-09-01&style=agents-first', '#c-readout', null],
  ['d-light', 'archipelago-D.html', '#cutoff=2026-09-01&style=work-first&zoom=far&palette=light', '#c-readout', null],
  ['d-near', 'archipelago-D.html', '#cutoff=2026-09-01&style=work-first&zoom=near', '#c-readout', 'page'],
];

(async () => {
  const browser = await chromium.launch({ headless: true });
  const out = { generated: new Date().toISOString(), base: BASE, shots: [] };
  for (const [id, page, hash, sel, crop] of SHOTS) {
    const p = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    const errors = [];
    p.on('pageerror', e => errors.push(e.message));
    try {
      await p.goto(`${BASE}/${page}${hash}`, { waitUntil: 'load', timeout: 120000 });
      await p.waitForTimeout(1500);
      const readout = sel ? await p.$eval(sel, e => e.textContent.trim()).catch(() => null) : null;
      // Fit the whole SVG (or the page's own zoom viewBox when crop === 'page')
      // into 1000 px for the picture.
      await p.evaluate((crop) => {
        const s = document.getElementById('efe-field');
        if (crop !== 'page') s.setAttribute('viewBox', '0 0 3600 3600');
        s.style.width = '1000px'; s.style.height = '1000px';
        // the control panels (fixed HTML) cover parts of the map in a
        // picture; hide it in the screenshot only
        document.querySelectorAll('[id$="-controls"]').forEach(e => e.style.display = 'none');
      }, crop);
      const svg = await p.$('#efe-field');
      await svg.screenshot({ path: path.join(OUT, `${id}.jpg`), type: 'jpeg', quality: 78 });
      out.shots.push({ id, page, hash, readout, errors });
      console.log(id, '·', readout || '', errors.length ? `· ${errors.length} errors` : '');
    } catch (e) {
      out.shots.push({ id, page, hash, readout: null, errors: [String(e)] });
      console.log(id, 'FAILED', String(e).slice(0, 120));
    }
    await p.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(OUT, 'shots.json'), JSON.stringify(out, null, 1));
})();
