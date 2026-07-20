// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health   -> "ok"
//   POST /generate  multipart: image file + seed/resolution/bg_removal/uv fields
//                   -> model/gltf-binary, or JSON {"error": "..."} on failure.

import { apiBase, loadConfig } from "./config";
import type { GenParams } from "./types";

async function base(): Promise<string> {
  return apiBase(await loadConfig());
}

export async function health(timeoutMs = 2000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${await base()}/health`, { signal: ctrl.signal });
    return res.ok && (await res.text()).trim() === "ok";
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

/** trellis-server maps bg_removal: "auto" keeps the server default (no field). */
function toForm(image: Blob, p: GenParams): FormData {
  const fd = new FormData();
  fd.append("image", image, "input.png");
  fd.append("seed", String(p.seed));
  fd.append("resolution", String(p.resolution));
  if (p.bgRemoval !== "auto") fd.append("bg_removal", p.bgRemoval);
  fd.append("uv", p.uv);
  return fd;
}

export interface GenerateResult {
  glb: Blob;
}

export async function generate(
  image: Blob,
  params: GenParams,
  signal?: AbortSignal,
): Promise<GenerateResult> {
  const res = await fetch(`${await base()}/generate`, {
    method: "POST",
    body: toForm(image, params),
    signal,
  });
  if (!res.ok) {
    let msg = `generation failed (HTTP ${res.status})`;
    try {
      const j = await res.json();
      if (j && typeof j.error === "string") msg = j.error;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(msg);
  }
  const glb = await res.blob();
  if (glb.size === 0) throw new Error("server returned an empty model");
  return { glb };
}
