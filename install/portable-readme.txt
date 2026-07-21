Trellis Studio - portable (no-install) build
============================================

This is a self-contained build: everything stays inside this folder and nothing
is written to the system (no installer, no AppData / registry / config dirs).

Layout (all next to this app):
  portable.dat   <- marker that enables portable mode -- do not delete
  data/          <- config + generated GLBs are written here (created on first run)
  runtime/       <- put the trellis-server runtime here (see setup)
  models/        <- put the GGUF model weights here (see setup)

First-time setup
----------------
  1. Download the server runtime for your GPU from the releases page and extract
     it into a "runtime" folder next to this app:
       trellis-cuda-<os>-x64      (NVIDIA)
       trellis-rocm-<os>-x64      (AMD; needs the matching ROCm runtime on PATH)
       trellis-vulkan-<os>-x64    (AMD / Intel / universal fallback)
  2. Put the 10 GGUF files from huggingface.co/ilintar/trellis2-gguf into a
     "models" folder next to this app.
  3. Launch the app (trellis-studio / trellis-studio.exe) and generate.

Or just launch the app and point it at an existing server binary + models
directory in Settings.

To uninstall: delete this folder.

Linux note: the app needs the system webkit2gtk-4.1 runtime installed.
