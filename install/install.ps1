<#
.SYNOPSIS
  Trellis Studio — one-command setup for Windows (x64).

.DESCRIPTION
  Detects the GPU runtime (CUDA / ROCm / Vulkan), downloads the matching
  trellis-server bundle + the TRELLIS.2 weights (~16.5 GB), installs the
  Trellis Studio desktop app, and writes the config the app reads on launch.

.EXAMPLE
  irm https://raw.githubusercontent.com/pwilkin/trellis.cpp/main/install/install.ps1 | iex
  # or, with options:
  ./install.ps1 -Backend vulkan -SkipModels
#>
[CmdletBinding()]
param(
  [ValidateSet("cuda", "rocm", "vulkan")] [string]$Backend = "",
  [int]$Gpu = 0,
  [int]$Port = 8080,
  [string]$Dest = "$env:LOCALAPPDATA\trellis-studio",
  [string]$ModelsDir = "",
  [switch]$SkipModels,
  [switch]$SkipApp,
  [switch]$Yes
)

$ErrorActionPreference = "Stop"
$Repo = "pwilkin/trellis.cpp"
$RelBase = "https://github.com/$Repo/releases/latest/download"
$HfBase = "https://huggingface.co/ilintar/trellis2-gguf/resolve/main"
$Models = @("birefnet.gguf", "dinov3.gguf", "ss_flow.gguf", "ss_dec.gguf",
  "shape_flow_512.gguf", "shape_flow_1024.gguf", "shape_dec.gguf",
  "tex_flow_512.gguf", "tex_flow_1024.gguf", "tex_dec.gguf")

if (-not $ModelsDir) { $ModelsDir = Join-Path $Dest "models" }
$RuntimeDir = Join-Path $Dest "runtime"
$ConfigDir = Join-Path $env:APPDATA "trellis-studio"

function Log($m)  { Write-Host "==> $m" -ForegroundColor Green }
function Info($m) { Write-Host " -  $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "warn: $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "error: $m" -ForegroundColor Red; exit 1 }

# ---- backend detection -----------------------------------------------------
function Detect-Backend {
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try { if (nvidia-smi -L 2>$null) { return "cuda" } } catch {}
  }
  try {
    $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    if ($gpus | Where-Object { $_.Name -match "AMD|Radeon" }) {
      # ROCm is possible on AMD, but the published bundle needs a matching TheRock
      # runtime; Vulkan is self-contained and robust, so it's the safe auto default.
      Info "detected an AMD GPU — ROCm-capable (use -Backend rocm to force it)"
    }
  } catch {}
  return "vulkan"
}

if (-not $Backend) {
  $Backend = Detect-Backend
  Log "auto-detected backend: $Backend"
} else {
  Log "backend (forced): $Backend"
}

Write-Host ""
Info "install dir : $Dest"
Info ("models dir  : {0}{1}" -f $ModelsDir, $(if ($SkipModels) { " (skipped)" } else { "" }))
Info "backend/gpu : $Backend / $Gpu     port: $Port"
Write-Host ""
if (-not $Yes) {
  $ans = Read-Host "Proceed? [Y/n]"
  if ($ans -match "^[nN]") { exit 0 }
}

# ---- download helper (resumable via BITS, IWR fallback) --------------------
function Download($url, $dest) {
  New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
  Info "down $(Split-Path $dest -Leaf)"
  try {
    Start-BitsTransfer -Source $url -Destination $dest -DisplayName (Split-Path $dest -Leaf)
  } catch {
    Warn "BITS failed, falling back to Invoke-WebRequest"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
  }
}

# ---- 1. server runtime bundle ---------------------------------------------
Log "downloading trellis-server ($Backend) runtime"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
$bundle = "trellis-$Backend-windows-x64.zip"
$tmp = Join-Path $env:TEMP $bundle
Download "$RelBase/$bundle" $tmp
Expand-Archive -Path $tmp -DestinationPath $RuntimeDir -Force
Remove-Item $tmp -Force
$ServerBin = Join-Path $RuntimeDir "trellis-server.exe"
if (-not (Test-Path $ServerBin)) { Die "trellis-server.exe not found after extract" }

if ($Backend -eq "rocm") {
  Warn "ROCm bundle needs a TheRock ROCm runtime on PATH; see docs/getting-started.md."
  Warn "If the server fails to start, re-run with -Backend vulkan."
}

# ---- 2. weights ------------------------------------------------------------
if ($SkipModels) {
  Warn "skipping model download (-SkipModels); set the models dir in the app's Settings."
} else {
  Log "downloading TRELLIS.2 weights (~16.5 GB, resumable) -> $ModelsDir"
  New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
  foreach ($m in $Models) { Download "$HfBase/$m" (Join-Path $ModelsDir $m) }
}

# ---- 3. desktop app --------------------------------------------------------
if ($SkipApp) {
  Warn "skipping desktop app download (-SkipApp)."
} else {
  Log "downloading Trellis Studio desktop app"
  $setup = Join-Path $env:TEMP "trellis-studio-windows-x64-setup.exe"
  try {
    Download "$RelBase/trellis-studio-windows-x64-setup.exe" $setup
    Info "launching installer (silent, per-user)"
    Start-Process $setup -ArgumentList "/S" -Wait
  } catch {
    Warn "app installer not available on the latest release yet — skipping."
    Warn "You can still run the UI in a browser against trellis-server (see docs)."
  }
}

# ---- 4. config -------------------------------------------------------------
Log "writing config"
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
$cfg = [ordered]@{
  serverBin = $ServerBin
  modelsDir = $ModelsDir
  backend   = $Backend
  gpu       = $Gpu
  host      = "127.0.0.1"
  port      = $Port
}
$cfg | ConvertTo-Json | Set-Content -Path (Join-Path $ConfigDir "config.json") -Encoding UTF8
Info "config: $(Join-Path $ConfigDir 'config.json')"

Write-Host ""
Log "done — launch Trellis Studio from the Start menu."
