import "./ui.css";
import { generate, health, waitForJob, type GenerateHooks, type GenerateResult, type JobStatus } from "./api";
import { loadConfig } from "./config";
import { renderSettings } from "./settings";
import { all, clear as clearStore, del as removeRecord, get as getRecord, isEphemeral, newId, put } from "./store";
import { isTauri, listen, saveBytes, saveToOutputDir } from "./tauri";
import { Viewer } from "./viewer";
import { DEFAULT_PARAMS, type GenParams, type GenRecord } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- element refs ----
const dropzone = $("dropzone");
const fileInput = $<HTMLInputElement>("file-input");
const inputPreview = $<HTMLImageElement>("input-preview");
const dropHint = $("dropzone-hint");
const generateBtn = $<HTMLButtonElement>("generate-btn");
const progress = $("progress");
const progressStage = $("progress-stage");
const progressElapsed = $("progress-elapsed");
const cancelBtn = $<HTMLButtonElement>("cancel-btn");
const resetViewBtn = $<HTMLButtonElement>("reset-view");
const saveGlbBtn = $<HTMLButtonElement>("save-glb");
const viewerCaption = $("viewer-caption");
const galleryEl = $("gallery");
const clearGalleryBtn = $("clear-gallery");
const backendBadge = $("backend-badge");
const serverDot = $("server-dot");
const serverLabel = $("server-label");
const setupBanner = $("setup-banner");

// ---- state ----
const viewer = new Viewer($("viewer-mount"));
let inputImage: Blob | null = null;
let inputName = "input.png";
let currentGlb: Blob | null = null;
let currentName = "model"; // file name stem for "Save GLB…" of the model in the viewer
// Set when the user opens a gallery item during a run: the result then goes to the
// gallery without taking over the viewer.
let browsedAway = false;
let activeId: string | null = null;
let serverOnline = false;
let generating = false;
let abort: AbortController | null = null;
let elapsedTimer: number | null = null;
let galleryUrls: string[] = [];
let warnedEphemeral = false;

// ---- controls -> params ----
function readParams(): GenParams {
  const res = parseInt(($("ctl-res") as HTMLSelectElement).value, 10);
  const seed = parseInt(($("ctl-seed") as HTMLInputElement).value, 10);
  return {
    resolution: (res === 512 || res === 1536 ? res : 1024) as GenParams["resolution"],
    seed: isNaN(seed) ? DEFAULT_PARAMS.seed : seed,
    bgRemoval: ($("ctl-bg") as HTMLSelectElement).value as GenParams["bgRemoval"],
    uv: ($("ctl-uv") as HTMLSelectElement).value as GenParams["uv"],
    texture: ($("ctl-tex") as HTMLSelectElement).value !== "off",
  };
}

function applyParams(p: GenParams): void {
  ($("ctl-res") as HTMLSelectElement).value = String(p.resolution);
  ($("ctl-seed") as HTMLInputElement).value = String(p.seed);
  ($("ctl-bg") as HTMLSelectElement).value = p.bgRemoval;
  ($("ctl-uv") as HTMLSelectElement).value = p.uv;
  // records saved before this option existed have no field: treat as textured
  ($("ctl-tex") as HTMLSelectElement).value = p.texture === false ? "off" : "on";
}

// ---- toasts ----
function toast(msg: string, kind: "" | "ok" | "err" = ""): void {
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 8000 : 4000);
}

// ---- input image ----
function setInput(blob: Blob, name: string): void {
  inputImage = blob;
  inputName = name || "input.png";
  inputPreview.src = URL.createObjectURL(blob);
  inputPreview.classList.remove("hidden");
  dropHint.classList.add("hidden");
  updateGenerateEnabled();
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") fileInput.click();
});
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) setInput(f, f.name);
});
// HTML5 drag-and-drop. In Tauri this needs `dragDropEnabled: false` on the
// window (tauri.conf.json): otherwise the webview's native file-drop handler
// swallows the drop on Windows/WebView2 and the page never sees the files.

// A file dropped anywhere outside the drop zone must not make the webview
// navigate to it (which would replace the whole app with the image).
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i;
const isImage = (f: File) => f.type.startsWith("image/") || IMAGE_EXT.test(f.name);

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  }),
);
// dragleave also fires when the pointer moves onto a child (the preview <img>),
// which made the highlight flicker; only clear it when actually leaving the zone.
dropzone.addEventListener("dragleave", (e) => {
  if (!dropzone.contains((e as DragEvent).relatedTarget as Node | null)) dropzone.classList.remove("drag");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  const f = (e as DragEvent).dataTransfer?.files?.[0];
  if (!f) return;
  if (isImage(f)) setInput(f, f.name);
  else toast(`"${f.name}" isn't an image — drop a PNG, JPEG or WebP.`, "err");
});
window.addEventListener("paste", async (e: ClipboardEvent) => {
  for (const item of e.clipboardData?.items ?? []) {
    const image = item.type.startsWith("image/") && item.getAsFile();
    if (image) return setInput(image, image.name || "pasted.png");
  }
  // WebKitGTK can omit image data from ClipboardEvent. Since paste was
  // user-triggered, try reading the clipboard item list for an image.
  const items = await navigator.clipboard.read().catch(() => []);
  for (const item of items) {
    const type = item.types.find((type) => type.startsWith("image/"));
    if (type) {
      const image = await item.getType(type).catch(() => null);
      if (image) return setInput(image, "pasted.png");
    }
  }
});

// ---- generate ----
function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function updateGenerateEnabled(): void {
  generateBtn.disabled = !(serverOnline && inputImage && !generating);
}

function describeJob(s: JobStatus | null): string {
  if (!s) return "server unreachable — retrying (the job keeps running on the server)…";
  if (s.status === "queued") {
    const ahead = (s.position ?? 1) - 1;
    return ahead > 0 ? `queued — ${ahead} job${ahead > 1 ? "s" : ""} ahead` : "queued — starts next";
  }
  if (s.status === "running") return "running on the server…";
  return s.status;
}

/**
 * Run one generation (new, or resumed after a reload) with the progress UI.
 * `rec` holds everything captured when it started, so browsing the gallery or
 * changing inputs meanwhile can't leak into the saved result.
 */
async function runJob(
  rec: GenRecord,
  work: (signal: AbortSignal, hooks: GenerateHooks) => Promise<GenerateResult>,
): Promise<void> {
  generating = true;
  browsedAway = false;
  updateGenerateEnabled();
  progress.classList.remove("hidden");
  progressStage.textContent = "sending…";
  const started = Date.now();
  progressElapsed.textContent = "0:00";
  elapsedTimer = window.setInterval(() => {
    progressElapsed.textContent = fmtElapsed(Date.now() - started);
  }, 1000);
  abort = new AbortController();
  let running = false;
  try {
    const { glb } = await work(abort.signal, {
      onJob: async (id) => {
        // A pending gallery entry: it survives a reload, which resumes the job.
        rec.jobId = id;
        rec.pending = true;
        await put(rec);
        await refreshGallery();
      },
      onStatus: (st) => {
        if (st?.status === "running") running = true;
        progressStage.textContent = describeJob(st);
      },
    });
    await finishRecord(rec, glb);
  } catch (e) {
    await removeRecord(rec.id);
    await refreshGallery();
    if (abort?.signal.aborted) {
      toast(running
        ? "Cancelled — the server finishes the run already in progress, but its result is discarded."
        : "Generation cancelled");
    } else toast((e as Error).message || "generation failed", "err");
  } finally {
    generating = false;
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    progress.classList.add("hidden");
    updateGenerateEnabled();
  }
}

/** Store a finished generation; show it unless the user is browsing another item. */
async function finishRecord(rec: GenRecord, glb: Blob): Promise<void> {
  rec.glb = glb;
  rec.pending = false;
  delete rec.jobId;
  const stem = rec.name.replace(/\.[^.]+$/, "") || "model";

  // 1) Write the GLB to the output folder FIRST. This on-disk file is the real
  // deliverable and must survive any later failure — a dead gallery DB (some
  // WebKitGTK builds can't open IndexedDB) or a WebGL/model-viewer crash must
  // never cost the user a successful generation.
  let savedPath: string | null = null;
  if (isTauri()) {
    try {
      const bytes = new Uint8Array(await glb.arrayBuffer());
      const fname = `${stem}_${rec.params.resolution}_seed${rec.params.seed}_${rec.id}.glb`;
      savedPath = await saveToOutputDir(fname, bytes);
    } catch (e) {
      toast(`Auto-save to output folder failed: ${(e as Error).message}`, "err");
    }
  }

  // 2) Add it to the gallery. The store falls back to in-memory if IndexedDB
  // is unavailable, so this never throws and never blocks the save above.
  await put(rec);
  if (isEphemeral() && !warnedEphemeral) {
    warnedEphemeral = true;
    toast(
      "Gallery won't persist across restarts (IndexedDB unavailable on this system) — " +
        "but every generation is still saved to your output folder.",
      "err",
    );
  }
  if (browsedAway) {
    toast(`"${rec.name}" finished — it's in the gallery${savedPath ? ` and saved to ${savedPath}` : ""}.`, "ok");
    await refreshGallery();
    return;
  }
  toast(savedPath ? `Saved to ${savedPath}` : "Generation complete", "ok");
  currentGlb = glb;
  currentName = stem;
  activeId = rec.id;
  setViewerTools(true);
  viewerCaption.textContent = `${rec.params.resolution} · seed ${rec.params.seed} · ${(glb.size / 1e6).toFixed(1)} MB`;
  await refreshGallery();

  // 3) Best-effort 3D preview + gallery thumbnail — never blocks the save above.
  try {
    await viewer.load(glb);
    const thumb = await viewer.thumbnail();
    if (thumb) {
      rec.thumb = thumb;
      await put(rec);
      await refreshGallery();
    }
  } catch (e) {
    toast(`3D preview couldn't render (your result is still saved): ${(e as Error).message}`, "err");
  }
}

async function doGenerate(): Promise<void> {
  if (!inputImage || generating) return;
  const params = readParams();
  const rec: GenRecord = {
    id: newId(),
    ts: Date.now(),
    name: inputName,
    params,
    input: inputImage,
    glb: null,
    thumb: null,
  };
  await runJob(rec, (signal, hooks) => generate(rec.input, params, signal, hooks));
}

/** Pick up jobs that were still running on the server when the page was closed. */
async function resumePending(): Promise<void> {
  for (const r of (await all()).filter((x) => x.pending)) {
    if (!r.jobId || generating) {
      if (!r.jobId) await removeRecord(r.id);
      continue;
    }
    toast(`Resuming "${r.name}" — still on the server`);
    const id = r.jobId;
    await runJob(r, (signal, hooks) => waitForJob(id, signal, hooks));
  }
  await refreshGallery();
}

generateBtn.addEventListener("click", doGenerate);
cancelBtn.addEventListener("click", () => abort?.abort());

// ---- viewer tools ----
function setViewerTools(on: boolean): void {
  resetViewBtn.disabled = !on;
  saveGlbBtn.disabled = !on;
}
resetViewBtn.addEventListener("click", () => viewer.resetView());
saveGlbBtn.addEventListener("click", async () => {
  if (!currentGlb) return;
  const bytes = new Uint8Array(await currentGlb.arrayBuffer());
  const ok = await saveBytes(`${currentName}.glb`, bytes);
  if (ok) toast("Saved", "ok");
});

// ---- gallery ----
async function loadRecord(id: string): Promise<void> {
  const rec = await getRecord(id);
  if (!rec) return;
  if (rec.pending || !rec.glb) {
    toast("Still generating on the server — it opens here when it's done.");
    return;
  }
  if (generating) browsedAway = true;
  try {
    await viewer.load(rec.glb);
  } catch (e) {
    toast((e as Error).message, "err");
    return;
  }
  inputImage = rec.input;
  inputName = rec.name;
  inputPreview.src = URL.createObjectURL(rec.input);
  inputPreview.classList.remove("hidden");
  dropHint.classList.add("hidden");
  applyParams(rec.params);
  currentGlb = rec.glb;
  currentName = rec.name.replace(/\.[^.]+$/, "") || "model";
  activeId = rec.id;
  setViewerTools(true);
  viewerCaption.textContent = `${rec.name} · ${rec.params.resolution} · seed ${rec.params.seed}`;
  updateGenerateEnabled();
  await refreshGallery();
}

async function refreshGallery(): Promise<void> {
  galleryUrls.forEach((u) => URL.revokeObjectURL(u));
  galleryUrls = [];
  const recs = await all();
  galleryEl.innerHTML = "";
  if (!recs.length) {
    galleryEl.innerHTML = `<div class="gallery-empty">No generations yet.</div>`;
    return;
  }
  for (const r of recs) {
    const item = document.createElement("div");
    item.className = "gitem" + (r.id === activeId ? " active" : "") + (r.pending ? " pending" : "");
    item.title = `${r.name} · ${new Date(r.ts).toLocaleString()}`;

    const img = document.createElement("img");
    const src = r.thumb ?? r.input;
    const url = URL.createObjectURL(src);
    galleryUrls.push(url);
    img.src = url;
    item.appendChild(img);

    const meta = document.createElement("div");
    meta.className = "gmeta";
    meta.textContent = r.pending ? "generating…" : `${r.params.resolution}`;
    item.appendChild(meta);

    const delBtn = document.createElement("button");
    delBtn.className = "gdel";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await removeRecord(r.id);
      if (activeId === r.id) activeId = null;
      await refreshGallery();
    });
    if (!r.pending) item.appendChild(delBtn);

    item.addEventListener("click", () => loadRecord(r.id));
    galleryEl.appendChild(item);
  }
}

clearGalleryBtn.addEventListener("click", async () => {
  if (!confirm("Delete all saved generations?")) return;
  await clearStore();
  activeId = null;
  await refreshGallery();
});

// ---- settings modal ----
const modal = $("settings-modal");
async function openSettings(): Promise<void> {
  await renderSettings($("settings-body"), () => {
    pollHealth();
    modal.classList.add("hidden");
    toast("Settings applied");
  });
  modal.classList.remove("hidden");
}
$("settings-btn").addEventListener("click", openSettings);
$("banner-settings").addEventListener("click", openSettings);
$("settings-close").addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (e) => {
  if (e.target === modal) modal.classList.add("hidden");
});

// ---- server status ----
async function pollHealth(): Promise<void> {
  const cfg = await loadConfig(true);
  backendBadge.textContent = cfg.backend !== "unknown" ? cfg.backend : "—";
  const ok = await health();
  serverOnline = ok;
  serverDot.className = "dot " + (ok ? "ok" : "err");
  serverLabel.textContent = ok ? "ready" : cfg.configured ? "offline" : "setup needed";
  const needSetup = !ok && !cfg.configured;
  setupBanner.classList.toggle("hidden", !needSetup);
  if (needSetup) {
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Trellis Studio isn't set up yet — run the installer or point it at your models directory.";
  } else if (!ok && cfg.configured) {
    setupBanner.classList.remove("hidden");
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Server is offline — it may still be loading, or check the models directory in settings.";
  }
  updateGenerateEnabled();
}

// ---- server log -> progress (Tauri only) ----
listen<string>("server-log", (line) => {
  const t = String(line).trim();
  if (generating && t) progressStage.textContent = t;
});

// ---- boot ----
async function boot(): Promise<void> {
  setViewerTools(false);
  await refreshGallery();
  await pollHealth();
  window.setInterval(pollHealth, 4000);
  if (!isTauri()) {
    // Browser mode: no shell to report a backend.
    backendBadge.textContent = "browser";
  }
  void resumePending();
}
boot();
