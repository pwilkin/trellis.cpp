// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health   -> "ok"
//   POST /jobs      same form as /generate -> 202 {"id"}; GET /jobs/{id} -> status;
//                   GET /jobs/{id}/result -> GLB; DELETE /jobs/{id} cancels / frees.
//   POST /generate  multipart: image file + seed/resolution/bg_removal/uv/texture fields;
//                   multi-view: one "image" part per view (front first) + "azimuths" and
//                   "elevations" (comma lists, one entry per image) + blend_temp/front_axis
//                   -> model/gltf-binary, or JSON {"error": "..."} on failure.

import { apiBase, loadConfig } from "./config";
import type { ExtraView, GenParams } from "./types";

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

/**
 * EXIF orientation of a JPEG (1 = upright; 0 = no tag / not parseable). Only the
 * APP1 segment near the file start is read.
 */
async function jpegOrientation(image: Blob): Promise<number> {
  const b = new DataView(await image.slice(0, 128 * 1024).arrayBuffer());
  let off = 2;
  while (off + 4 <= b.byteLength && b.getUint8(off) === 0xff) {
    const marker = b.getUint8(off + 1);
    const len = b.getUint16(off + 2);
    if (marker === 0xda) break; // start of scan: no more metadata
    if (marker === 0xe1 && off + 10 <= b.byteLength && b.getUint32(off + 4) === 0x45786966) {
      const tiff = off + 10; // "Exif\0\0" then the TIFF header
      if (tiff + 8 > b.byteLength) return 0;
      const le = b.getUint16(tiff) === 0x4949;
      const ifd = tiff + b.getUint32(tiff + 4, le);
      if (ifd + 2 > b.byteLength) return 0;
      const n = b.getUint16(ifd, le);
      for (let i = 0; i < n; i++) {
        const e = ifd + 2 + i * 12;
        if (e + 12 > b.byteLength) return 0;
        if (b.getUint16(e, le) === 0x0112) return b.getUint16(e + 8, le);
      }
      return 0;
    }
    off += 2 + len;
  }
  return 0;
}

/**
 * Can trellis-server read these bytes as-is? It decodes with stb_image (PNG,
 * JPEG, BMP, GIF, TGA, PSD, HDR, PNM). Those are sent untouched: re-encoding
 * would premultiply alpha on cut-out edges, apply ICC profiles the CLI ignores,
 * and inflate uploads; TGA/PSD/HDR/PNM can't even be decoded by a browser.
 */
async function serverReadable(image: Blob): Promise<boolean> {
  const h = new Uint8Array(await image.slice(0, 16).arrayBuffer());
  const ascii = String.fromCharCode(...h);
  if (h[0] === 0x89 && ascii.slice(1, 4) === "PNG") return true;
  if (h[0] === 0xff && h[1] === 0xd8 && h[2] === 0xff) {
    // stb_image ignores EXIF rotation: let the browser apply it to rotated photos.
    return (await jpegOrientation(image)) <= 1;
  }
  if (ascii.startsWith("GIF8") || ascii.startsWith("BM") || ascii.startsWith("8BPS")) return true;
  if (ascii.startsWith("#?RADIANCE") || ascii.startsWith("#?RGBE")) return true;
  if (/^P[56]\s/.test(ascii)) return true;
  // TGA has no magic number: trust the file name / type.
  const name = image instanceof File ? image.name : "";
  return /\.(tga|tpic)$/i.test(name) || /tga|targa/i.test(image.type);
}

/** The image as the server should receive it: untouched when readable, else PNG. */
export async function prepareImage(image: Blob): Promise<Blob> {
  return (await serverReadable(image)) ? image : toPng(image);
}

/**
 * Re-encode an image as PNG in the browser, for formats stb_image can't read
 * (WebP, AVIF, SVG...) and for EXIF-rotated JPEGs (<img> applies the rotation).
 */
export async function toPng(image: Blob): Promise<Blob> {
  const url = URL.createObjectURL(image);
  try {
    const img = new Image();
    img.src = url;
    try {
      await img.decode();
    } catch {
      throw new Error("this image format can't be decoded — use a PNG, JPEG or WebP file");
    }
    // An SVG without width/height has no intrinsic size.
    const w = img.naturalWidth || 1024;
    const h = img.naturalHeight || 1024;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas unavailable");
    ctx.drawImage(img, 0, 0, w, h);
    const png = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!png) throw new Error("could not convert the image to PNG");
    return png;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** trellis-server maps bg_removal: "auto" keeps the server default (no field). */
function toForm(image: Blob, p: GenParams, views: ExtraView[]): FormData {
  const fd = new FormData();
  fd.append("image", image, "input.png");
  if (views.length) {
    views.forEach((v, i) => fd.append("image", v.image, `view${i + 1}.png`));
    fd.append("azimuths", [p.mainAz ?? 0, ...views.map((v) => v.az)].map(String).join(","));
    fd.append("elevations", [p.mainEl ?? 0, ...views.map((v) => v.el)].map(String).join(","));
    if (p.blendTempOverride !== undefined) fd.append("blend_temp", String(p.blendTempOverride));
    if (p.frontAxisOverride) fd.append("front_axis", p.frontAxisOverride);
  }
  fd.append("seed", String(p.seed));
  fd.append("resolution", String(p.resolution));
  if (p.bgRemoval !== "auto") fd.append("bg_removal", p.bgRemoval);
  fd.append("uv", p.uv);
  if (!p.texture) fd.append("texture", "off");
  return fd;
}

export interface GenerateResult {
  glb: Blob;
}

/** GET /jobs/{id} on trellis-server. */
export interface JobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  position?: number;
  elapsed?: number;
  error?: string;
}

export interface GenerateHooks {
  /** The server accepted the job (not called on servers without the job API). */
  onJob?: (id: string) => void;
  /** Every poll answer, plus `null` while the server is unreachable (still retrying). */
  onStatus?: (s: JobStatus | null) => void;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("aborted", "AbortError"));
    }, { once: true });
  });

/** The server's error message: JSON {"error"} or, for field validation, plain text. */
async function errorOf(res: Response, fallback: string): Promise<string> {
  try {
    const body = (await res.text()).trim();
    try {
      const j = JSON.parse(body);
      if (j && typeof j.error === "string" && j.error) return j.error;
    } catch {
      if (body && body.length < 500 && !body.startsWith("<")) return body;
    }
  } catch {
    /* unreadable body */
  }
  return fallback;
}

/**
 * Generate a model. Uses the server's job API (POST /jobs, then short polls) so a
 * long run never depends on one HTTP request staying open: a 1024 run can take
 * 20 minutes, longer than browsers and proxies wait for a response (Firefox gives
 * up after 300 s). Servers without /jobs get the old synchronous POST /generate.
 */
export async function generate(
  image: Blob,
  params: GenParams,
  views: ExtraView[] = [],
  signal?: AbortSignal,
  hooks: GenerateHooks = {},
): Promise<GenerateResult> {
  const url = await base();
  if (views.length) {
    const res = await fetch(`${url}/capabilities`, { signal, cache: "no-store" });
    if (!res.ok && res.status !== 404 && res.status !== 405) {
      throw new Error(await errorOf(res, `Could not check server capabilities (HTTP ${res.status}).`));
    }
    const capabilities = res.ok ? await res.json().catch(() => null) : null;
    if (capabilities?.multiview !== true) {
      throw new Error("Update trellis-server for multi-view generation, or use a single image.");
    }
  }
  const pngs = await Promise.all([image, ...views.map((v) => v.image)].map(prepareImage));
  const form = () => toForm(pngs[0], params, views.map((v, i) => ({ ...v, image: pngs[i + 1] })));

  const sub = await fetch(`${url}/jobs`, { method: "POST", body: form(), signal });
  if (!views.length && (sub.status === 404 || sub.status === 405)) {
    // Server predates the job API: one synchronous request.
    const res = await fetch(`${url}/generate`, { method: "POST", body: form(), signal });
    if (!res.ok) throw new Error(await errorOf(res, `generation failed (HTTP ${res.status})`));
    const glb = await res.blob();
    if (glb.size === 0) throw new Error("server returned an empty model");
    return { glb };
  }
  if (!sub.ok) throw new Error(await errorOf(sub, `generation failed (HTTP ${sub.status})`));
  const { id } = (await sub.json()) as JobStatus;
  hooks.onJob?.(id);
  return waitForJob(id, signal, hooks);
}

/**
 * Poll a job until it finishes, then download its GLB and free it on the server.
 * Network errors and gateway failures are retried with backoff indefinitely: the
 * job keeps running server-side, so only the server saying the job failed or is
 * unknown ends the wait (or the caller aborting, which cancels a queued job).
 */
export async function waitForJob(
  id: string,
  signal?: AbortSignal,
  hooks: GenerateHooks = {},
): Promise<GenerateResult> {
  const url = `${await base()}/jobs/${encodeURIComponent(id)}`;
  signal?.addEventListener("abort", () => {
    // Best effort: cancels a queued job, frees a finished one; a running one can't be stopped.
    fetch(url, { method: "DELETE" }).catch(() => {});
  }, { once: true });

  // Retries a request through network errors and 5xx-without-JSON (proxy) answers.
  let delay = 1000;
  const retrying = async (path: string): Promise<Response> => {
    for (;;) {
      try {
        const res = await fetch(url + path, { signal, cache: "no-store" });
        const isJson = (res.headers.get("content-type") || "").includes("json");
        if (res.status < 500 || isJson) return res;
      } catch (e) {
        if (signal?.aborted) throw e;
      }
      hooks.onStatus?.(null);
      await sleep(delay, signal);
      delay = Math.min(delay * 2, 15000);
    }
  };

  for (;;) {
    const res = await retrying("");
    if (res.status === 404) throw new Error("the server no longer knows this job (was it restarted?)");
    if (!res.ok) throw new Error(await errorOf(res, `job status failed (HTTP ${res.status})`));
    const st = (await res.json()) as JobStatus;
    delay = 1000;
    hooks.onStatus?.(st);
    if (st.status === "error") throw new Error(st.error || "3D reconstruction failed");
    if (st.status === "cancelled") throw new Error("the job was cancelled");
    if (st.status === "done") break;
    await sleep(2000, signal);
  }

  const res = await retrying("/result");
  if (!res.ok) throw new Error(await errorOf(res, `downloading the model failed (HTTP ${res.status})`));
  const glb = await res.blob();
  if (glb.size === 0) throw new Error("server returned an empty model");
  fetch(url, { method: "DELETE" }).catch(() => {});
  return { glb };
}
