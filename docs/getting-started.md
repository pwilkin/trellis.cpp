# Getting started with Trellis Studio

**Trellis Studio** is a desktop app that runs Microsoft TRELLIS.2 image→3D
reconstruction locally (via `trellis.cpp`) and previews the result in an
interactive 3D viewer. This guide gets a brand-new machine from zero to a
generated `.glb`.

## One-command install

The installer auto-detects your GPU runtime (**CUDA → ROCm → Vulkan** fallback),
downloads the matching `trellis-server` build plus the ~16.5 GB of model weights,
installs the desktop app, and writes the config the app reads on launch.

### Linux (x86-64)

```bash
curl -fsSL https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.sh | bash
```

### Windows (x64), in PowerShell

```powershell
irm https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.ps1 | iex
```

That's it — launch **Trellis Studio**, drop in an image, and click **Generate 3D**.

## What gets installed where

| what | Linux | Windows |
|------|-------|---------|
| server + runtime libs | `~/.local/share/trellis-studio/runtime/` | `%LOCALAPPDATA%\trellis-studio\runtime\` |
| model weights (~16.5 GB) | `~/.local/share/trellis-studio/models/` | `%LOCALAPPDATA%\trellis-studio\models\` |
| app config | `~/.config/trellis-studio/config.json` | `%APPDATA%\trellis-studio\config.json` |
| desktop app | `.AppImage` in the install dir | installed via the setup .exe (Start menu) |

## Installer options

Both scripts accept the same flags (`--flag value` on Linux, `-Flag value` on
Windows):

| flag | effect |
|------|--------|
| `--backend cuda\|rocm\|vulkan` | force a runtime instead of auto-detecting |
| `--gpu N` | GPU index (default `0`; `<0` = CPU) |
| `--port P` | server port (default `8080`) |
| `--dest DIR` | install location |
| `--models-dir DIR` | where to store weights (e.g. a bigger drive) |
| `--skip-models` | don't download weights (set the folder later in Settings) |
| `--skip-app` | don't download the desktop app |
| `-y` / `-Yes` | don't prompt for confirmation |

Examples:

```bash
# reuse weights already on a fast drive; force Vulkan
./install/install.sh --backend vulkan --models-dir /mnt/ssd/trellis --skip-models
```

## Backend detection

- **NVIDIA** → CUDA (the bundle ships the CUDA runtime; nothing else needed).
- **AMD / Intel / everything else** → **Vulkan**, which is self-contained and, on
  the validated Strix Halo iGPU, actually the fastest backend. The installer
  notes when an AMD card is ROCm-capable.
- **ROCm** is opt-in with `--backend rocm`. The published ROCm bundle needs a
  matching TheRock ROCm 7.x runtime on your library path (`LD_LIBRARY_PATH` /
  `PATH`); if the server won't start, re-run with `--backend vulkan`.

## Using the app

1. **Drop or pick an image** (or paste from the clipboard).
2. Adjust **Resolution** (512 light / 1024 cascade / 1536 high), **seed**,
   **background removal**, and **UV unwrap** if you like.
3. **Generate 3D** — this takes a few minutes; the live stage line shows progress.
4. **Rotate/zoom** in the preview; **Reset view** re-frames; **Save GLB…** exports.
5. Every result is saved to a local **gallery** (IndexedDB) — click a thumbnail to
   reload it, even after restarting the app.

## No app? Use it in a browser

The UI is a plain web bundle, so you can skip the desktop app entirely:

```bash
# start the server the installer downloaded
~/.local/share/trellis-studio/runtime/trellis-server \
  --models ~/.local/share/trellis-studio/models --port 8080
```

then open the built UI (or `npm run dev` in `app/`) and point it at
`127.0.0.1:8080` in **Settings**.

## Troubleshooting

- **"Server is offline"** right after generating — the pipeline is still loading
  the models; large weights take a moment on the first request.
- **The app can't read the server response / a CORS error appears** — the app
  needs a `trellis-server` build that sends CORS headers (v0.4.4+). The installer
  pulls the *latest* release, so update if you're on an older server bundle.
- **"setup needed" banner** — no `config.json` was found. Re-run the installer, or
  open **Settings** and point it at your models directory.
- **ROCm server won't start** — install the gfx-matched TheRock ROCm runtime, or
  switch to Vulkan (`--backend vulkan`).
