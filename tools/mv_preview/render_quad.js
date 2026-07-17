// Standalone Playwright renderer: load quad.html per asset, capture 4 framed views at 2048px, write PNGs.
// Replaces the killed Playwright MCP. Uses the same scroll-into-view + settle recipe that fixed stale captures.
const PW = '/home/ilintar/.npm/_npx/9833c18b2d85bc59/node_modules/playwright';
const { chromium } = require(PW);
const fs = require('fs');

const MV = '/devel/alt/trellis.cpp/tools/mv_preview';
const BASE = 'http://localhost:8178';
const SIZE = 2048;
const assets = process.argv.slice(2);           // e.g. chariot spacestation imp femaleknight
if (!assets.length) { console.error('usage: node render_quad.js <asset...>'); process.exit(1); }

(async () => {
  const browser = await chromium.launch({
    executablePath: '/home/ilintar/.cache/ms-playwright/chromium-1229/chrome-linux64/chrome',
    headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
           '--ignore-gpu-blocklist', '--no-sandbox'],
  });
  const page = await browser.newPage({ viewport: { width: SIZE * 2, height: SIZE * 2 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => console.log('  [pageerror]', String(e).slice(0, 200)));

  for (const name of assets) {
    const t0 = Date.now();
    await page.goto(`${BASE}/quad.html?src=${name}.glb&name=${name}&size=${SIZE}`, { waitUntil: 'load', timeout: 60000 });
    // wait for all 4 model-viewers to load
    await page.waitForFunction('window.allReady === true || window.mvError', null, { timeout: 120000 }).catch(() => {});
    const err = await page.evaluate('window.mvError');
    if (err) { console.log(`${name}: LOAD ERROR ${err}`); continue; }
    await page.waitForTimeout(1500);              // auto-framing settle

    const blobs = await page.evaluate(async (SIZE) => {
      const out = {};
      for (const view of ['front', 'right', 'back', 'left']) {
        const mv = window.viewers[view];
        mv.scrollIntoView({ block: 'center' });
        await new Promise(r => setTimeout(r, 700)); // scrolled-in viewer renders fresh in shared WebGL ctx
        const blob = await mv.toBlob({ mimeType: 'image/png', idealAspect: false });
        const buf = await blob.arrayBuffer();
        let bin = ''; const b = new Uint8Array(buf);
        for (let i = 0; i < b.length; i++) bin += String.fromCharCode(b[i]);
        out[view] = btoa(bin);
      }
      return out;
    }, SIZE);

    const sizes = {};
    for (const view of ['front', 'right', 'back', 'left']) {
      const png = Buffer.from(blobs[view], 'base64');
      fs.writeFileSync(`${MV}/${name}_${view}.png`, png);
      sizes[view] = png.length;
    }
    console.log(`${name}: OK ${((Date.now() - t0) / 1000).toFixed(0)}s  ` +
                Object.entries(sizes).map(([k, v]) => `${k}=${v}`).join(' '));
  }
  await browser.close();
  console.log('RENDER DONE');
})().catch(e => { console.error('FATAL', e); process.exit(1); });
