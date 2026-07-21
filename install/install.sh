#!/usr/bin/env bash
# Trellis Studio — one-command setup for Linux (x86-64).
#
#   curl -fsSL https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.sh | bash
#
# Detects the GPU runtime (CUDA / ROCm / Vulkan), downloads the matching
# trellis-server bundle + the TRELLIS.2 weights (~16.5 GB), installs the
# Trellis Studio desktop app, and writes the config the app reads on launch.
set -euo pipefail

REPO="pwilkin/trellis.cpp"
REL_BASE="https://github.com/${REPO}/releases/latest/download"
HF_BASE="https://huggingface.co/ilintar/trellis2-gguf/resolve/main"
MODELS=(birefnet.gguf dinov3.gguf ss_flow.gguf ss_dec.gguf \
        shape_flow_512.gguf shape_flow_1024.gguf shape_dec.gguf \
        tex_flow_512.gguf tex_flow_1024.gguf tex_dec.gguf)
# ROCm gfx arches covered by the published rocm bundle (see .github/workflows/release.yml).
ROCM_GFX="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1200 gfx1201"

# ---- defaults / args -------------------------------------------------------
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/trellis-studio"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/trellis-studio"
BACKEND=""; GPU=0; PORT=8080; MODELS_DIR=""; SKIP_MODELS=0; SKIP_APP=0; ASSUME_YES=0

usage() {
  cat <<EOF
Trellis Studio installer (Linux)

  --backend cuda|rocm|vulkan   force a runtime (default: auto-detect)
  --gpu N                      GPU index (default 0; <0 = CPU)
  --port P                     server port (default 8080)
  --dest DIR                   install location (default $DEST)
  --models-dir DIR             where to put weights (default <dest>/models)
  --skip-models                don't download the ~16.5 GB weights
  --skip-app                   don't download the desktop app
  -y, --yes                    don't prompt for confirmation
  -h, --help                   this help
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --dest) DEST="$2"; shift 2;;
    --models-dir) MODELS_DIR="$2"; shift 2;;
    --skip-models) SKIP_MODELS=1; shift;;
    --skip-app) SKIP_APP=1; shift;;
    -y|--yes) ASSUME_YES=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 1;;
  esac
done
MODELS_DIR="${MODELS_DIR:-$DEST/models}"
RUNTIME_DIR="$DEST/runtime"

# ---- logging ---------------------------------------------------------------
# Status messages go to stderr so command substitution (e.g. $(detect_backend))
# only ever captures the intended stdout value.
c_g=$'\e[32m'; c_y=$'\e[33m'; c_r=$'\e[31m'; c_b=$'\e[36m'; c_0=$'\e[0m'
log()  { echo "${c_g}==>${c_0} $*" >&2; }
info() { echo "${c_b} - ${c_0} $*" >&2; }
warn() { echo "${c_y}warn:${c_0} $*" >&2; }
die()  { echo "${c_r}error:${c_0} $*" >&2; exit 1; }

for t in curl tar; do command -v "$t" >/dev/null || die "'$t' is required"; done

# ---- backend detection -----------------------------------------------------
detect_backend() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -qi gpu; then
    echo cuda; return
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    local gfx; gfx="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -1 || true)"
    if [ -n "$gfx" ] && echo "$ROCM_GFX" | grep -qw "$gfx"; then
      # ROCm is available, but the published rocm bundle needs a matching TheRock
      # runtime; Vulkan is self-contained and often faster on these iGPUs, so it's
      # the safe auto default. Pass --backend rocm to force ROCm.
      info "detected AMD $gfx — ROCm-capable (use --backend rocm to force it)"
    fi
  fi
  echo vulkan
}

if [ -z "$BACKEND" ]; then
  BACKEND="$(detect_backend)"
  log "auto-detected backend: ${c_b}${BACKEND}${c_0}"
else
  log "backend (forced): ${c_b}${BACKEND}${c_0}"
fi
case "$BACKEND" in cuda|rocm|vulkan) ;; *) die "invalid backend: $BACKEND";; esac

echo
info "install dir : $DEST"
info "models dir  : $MODELS_DIR $([ "$SKIP_MODELS" = 1 ] && echo '(skipped)')"
info "backend/gpu : $BACKEND / $GPU     port: $PORT"
echo
if [ "$ASSUME_YES" != 1 ] && [ -t 0 ]; then
  read -r -p "Proceed? [Y/n] " ans; case "${ans:-y}" in [nN]*) exit 0;; esac
fi

# ---- download helper (resumable) ------------------------------------------
download() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  info "↓ $(basename "$dest")"
  curl -fL --retry 3 --retry-delay 2 --progress-bar -C - -o "$dest" "$url" \
    || die "download failed: $url"
}

# ---- 1. server runtime bundle ---------------------------------------------
log "downloading trellis-server ($BACKEND) runtime"
mkdir -p "$RUNTIME_DIR"
BUNDLE="trellis-${BACKEND}-linux-x64.tar.gz"
TMP_BUNDLE="$(mktemp)"
download "${REL_BASE}/${BUNDLE}" "$TMP_BUNDLE"
tar -C "$RUNTIME_DIR" -xzf "$TMP_BUNDLE"
rm -f "$TMP_BUNDLE"
SERVER_BIN="$RUNTIME_DIR/trellis-server"
[ -f "$SERVER_BIN" ] || die "trellis-server not found after extract"
chmod +x "$SERVER_BIN" || true

if [ "$BACKEND" = "rocm" ]; then
  warn "ROCm bundle needs a TheRock ROCm runtime on LD_LIBRARY_PATH."
  warn "See docs/getting-started.md; if the server fails to start, install ROCm 7.x"
  warn "(TheRock, gfx-matched) or re-run with --backend vulkan."
fi

# ---- 2. weights ------------------------------------------------------------
if [ "$SKIP_MODELS" = 1 ]; then
  warn "skipping model download (--skip-models); set them in the app's Settings."
else
  log "downloading TRELLIS.2 weights (~16.5 GB, resumable) -> $MODELS_DIR"
  mkdir -p "$MODELS_DIR"
  for m in "${MODELS[@]}"; do download "${HF_BASE}/${m}" "$MODELS_DIR/${m}"; done
fi

# ---- 3. desktop app --------------------------------------------------------
if [ "$SKIP_APP" = 1 ]; then
  warn "skipping desktop app download (--skip-app)."
else
  log "downloading Trellis Studio desktop app"
  APP="$DEST/Trellis Studio.AppImage"
  if curl -fL --retry 2 --progress-bar -o "$APP" "${REL_BASE}/trellis-studio-linux-x86_64.AppImage"; then
    chmod +x "$APP"
    info "app installed: $APP"
  else
    warn "app AppImage not available on the latest release yet — skipping."
    warn "You can still run the UI in a browser against trellis-server (see docs)."
    rm -f "$APP"
  fi
fi

# ---- 4. config -------------------------------------------------------------
log "writing config"
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.json" <<JSON
{
  "serverBin": "$SERVER_BIN",
  "modelsDir": "$MODELS_DIR",
  "backend": "$BACKEND",
  "gpu": $GPU,
  "host": "127.0.0.1",
  "port": $PORT,
  "outputDir": "$DEST/output"
}
JSON
info "config: $CONFIG_DIR/config.json"

echo
log "${c_g}done${c_0} — launch Trellis Studio${SKIP_MODELS:+ (add your models dir in Settings)}."
[ -f "$DEST/Trellis Studio.AppImage" ] && info "run: \"$DEST/Trellis Studio.AppImage\""
