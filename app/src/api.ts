// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health   -> "ok"
//   POST /jobs      same form as /generate -> 202 {"id"}; GET /jobs/{id} -> status;
//                   GET /jobs/{id}/result -> GLB; DELETE /jobs/{id} cancels / frees.
//   POST /generate  multipart: image file + seed/resolution/bg_removal/uv/texture fields
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
  signal?: AbortSignal,
  hooks: GenerateHooks = {},
): Promise<GenerateResult> {
  const form = () => toForm(image, params);
  const url = await base();

  const sub = await fetch(`${url}/jobs`, { method: "POST", body: form(), signal });
  if (sub.status === 404 || sub.status === 405) {
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
