# model-viewer GLB preview harness

Renders GLB previews with the same Three.js-based `<model-viewer>` component the
Lemonade desktop app uses (`src/app/src/renderer/vendor/model-viewer.min.js`),
giving true PBR output (environment lighting, metallic-roughness) instead of the
flat-shaded Python rasterizer in `render_glb.py` / `render_glb_fast.py`.

## Usage

1. Symlink or copy the GLB next to this directory's files, then serve it:

   ```bash
   ln -sf /path/to/model.glb tools/mv_preview/model.glb
   python3 tools/mv_preview/serve.py   # http://127.0.0.1:8177, supports PUT uploads
   ```

2. Open `http://localhost:8177/index.html?src=model.glb&orbit=45deg 65deg auto`
   in any browser (or drive it headlessly with Playwright).

## Headless capture notes (Playwright)

- The page exposes `window.mvReady` (set on the component's `load` event) and
  `window.mvError`.
- Capture with `mv.toBlob({mimeType: 'image/png'})` and `PUT` the blob back to
  the server (`fetch('/shot.png', {method: 'PUT', body: blob})`) — headless
  screenshot-to-file paths are unreliable across MCP servers.
- **One navigation per camera angle.** In a hidden/headless tab the component
  does not repaint after `cameraOrbit` changes + `jumpCameraToGoal()`, so
  in-place orbit changes capture stale frames. Load the page (or an iframe)
  fresh with the `orbit` query parameter for each view; the initial render is
  always correct. Radius `auto` gives proper framing.
