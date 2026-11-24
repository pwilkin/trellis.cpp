<#
.SYNOPSIS
  Download the TRELLIS.2 GGUF weights.

.DESCRIPTION
  install/install.ps1 fetches weights, but it is bundled with an x64-only runtime
  download. This does just the weights, so it works on ARM64 where we build the
  runtime ourselves.

  Downloads are resumable: an interrupted run can simply be re-run, and files that
  already match the published size are skipped.

.EXAMPLE
  scripts\fetch-models.ps1 -Quant q4
  scripts\fetch-models.ps1 -Quant q8 -Dest D:\trellis-models
#>
[CmdletBinding()]
param(
    # "" = f16 (~16.5 GB), q8 (~9.5 GB), q4 (~6 GB)
    [ValidateSet('', 'q8', 'q4')]
    [string]$Quant = 'q4',

    [string]$Dest = ''
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
if (-not $Dest) { $Dest = Join-Path $repo "models\$(if ($Quant) { $Quant } else { 'f16' })" }

$base = 'https://huggingface.co/ilintar/trellis2-gguf/resolve/main'
$path = if ($Quant) { "$Quant/" } else { '' }

$models = @(
    # shape_enc.gguf (--refine-mesh --refine-coords encoder) is optional and not published here;
    # convert it with tools/convert.py shape_enc. Without it --refine-mesh falls back to voxel coords.
    'birefnet.gguf', 'dinov3.gguf',
    'ss_flow.gguf', 'ss_dec.gguf',
    'shape_flow_512.gguf', 'shape_flow_1024.gguf', 'shape_dec.gguf',
    'tex_flow_512.gguf', 'tex_flow_1024.gguf', 'tex_dec.gguf'
)

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Write-Host "[fetch] quant: $(if ($Quant) { $Quant } else { 'f16' })   dest: $Dest" -ForegroundColor Cyan

# HF serves these LFS files from a CDN redirect, and the redirect response's own
# Content-Length (a few dozen bytes) is NOT the file size. The real size comes from
# HF's X-Linked-Size header. Returns -1 when the size cannot be established, in which
# case the caller skips size verification rather than guessing.
function Get-RemoteSize([string]$url) {
    try {
        $r = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -MaximumRedirection 5
        foreach ($h in @('X-Linked-Size', 'Content-Length')) {
            if ($r.Headers.Keys -contains $h) {
                $v = [int64](@($r.Headers[$h])[0])
                if ($v -gt 1MB) { return $v }     # anything smaller is a redirect/pointer body
            }
        }
    } catch { }
    return -1
}

$total = 0L
foreach ($m in $models) {
    $url = "$base/$path$m"
    $out = Join-Path $Dest $m

    $remote = Get-RemoteSize $url
    if (Test-Path $out) {
        $have = (Get-Item $out).Length
        if ($remote -gt 0 -and $have -eq $remote) {
            Write-Host "[fetch] skip $m ($([math]::Round($have/1MB)) MB, already complete)" -ForegroundColor DarkGray
            $total += $have
            continue
        }
        if ($remote -lt 0 -and $have -gt 1MB) {
            Write-Warning "cannot confirm remote size for $m; keeping existing $([math]::Round($have/1MB)) MB file (delete it to force a re-download)"
            $total += $have
            continue
        }
        Write-Host "[fetch] re-downloading $m (have $have, want $remote)" -ForegroundColor Yellow
        Remove-Item $out -Force
    }

    $sizeNote = if ($remote -gt 0) { "$([math]::Round($remote/1MB)) MB" } else { 'unknown size' }
    Write-Host "[fetch] $m ($sizeNote)" -ForegroundColor Cyan

    # BITS is resumable and does not buffer the whole body in memory the way
    # Invoke-WebRequest does -- which matters for the 2 GB+ flow models.
    try {
        Start-BitsTransfer -Source $url -Destination $out -Description "trellis $m" -ErrorAction Stop
    } catch {
        Write-Warning "BITS failed for $m ($($_.Exception.Message)); falling back to Invoke-WebRequest"
        Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
    }

    $got = (Get-Item $out).Length
    if ($remote -gt 0 -and $got -ne $remote) {
        throw "size mismatch for ${m}: got $got, expected $remote"
    }
    $total += $got
}

Write-Host "[fetch] done -- $([math]::Round($total/1GB, 2)) GB in $Dest" -ForegroundColor Green
