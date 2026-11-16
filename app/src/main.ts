import "./ui.css";
import { generate, health, waitForJob, type GenerateHooks, type GenerateResult, type JobStatus } from "./api";
import { loadConfig } from "./config";
import { renderSettings } from "./settings";
import { all, clear as clearStore, del as removeRecord, get as getRecord, isEphemeral, newId, put } from "./store";
import { isTauri, listen, saveBytes, saveToOutputDir } from "./tauri";
import { Viewer } from "./viewer";
import { DEFAULT_PARAMS, MAX_VIEWS, type ExtraView, type GenParams, type GenRecord } from "./types";

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
const frontTag = $("front-tag");
const viewsEl = $("views");
const viewsList = $("views-list");
const addViewBtn = $<HTMLButtonElement>("add-view");
const viewInput = $<HTMLInputElement>("view-input");
const clearViewsBtn = $("clear-views");
const mvAdv = $("mv-adv");

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
let extraViews: ExtraView[] = [];
// Camera direction of the main image (sent only with extra views).
let mainAz = 0;
let mainEl = 0;
let viewUrls: string[] = [];

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
    blendTemp: readBlendTemp(),
    frontAxis: ($("ctl-front") as HTMLSelectElement).value as GenParams["frontAxis"],
    mainAz,
    mainEl,
  };
}

function readBlendTemp(): number {
  const t = parseFloat(($("ctl-blend") as HTMLInputElement).value);
  return isFinite(t) && t > 0 ? t : DEFAULT_PARAMS.blendTemp;
}

function applyParams(p: GenParams): void {
  ($("ctl-res") as HTMLSelectElement).value = String(p.resolution);
  ($("ctl-seed") as HTMLInputElement).value = String(p.seed);
  ($("ctl-bg") as HTMLSelectElement).value = p.bgRemoval;
  ($("ctl-uv") as HTMLSelectElement).value = p.uv;
  // records saved before this option existed have no field: treat as textured
  ($("ctl-tex") as HTMLSelectElement).value = p.texture === false ? "off" : "on";
  // records saved before multi-view existed have neither field
  ($("ctl-blend") as HTMLInputElement).value = String(p.blendTemp ?? DEFAULT_PARAMS.blendTemp);
  ($("ctl-front") as HTMLSelectElement).value = p.frontAxis ?? DEFAULT_PARAMS.frontAxis;
  mainAz = p.mainAz ?? 0;
  mainEl = p.mainEl ?? 0;
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
  renderViews(); // the main-image row shows the new thumbnail
  updateGenerateEnabled();
}

// ---- extra views (multi-view conditioning) ----
// The main image is always the front view (azimuth 0, elevation 0); each extra
// view is another photo of the same object with its own camera direction.
const AZIMUTHS: [number, string][] = [
  [0, "front"], [45, "front-right"], [90, "right"], [135, "back-right"],
  [180, "back"], [225, "back-left"], [270, "left"], [315, "front-left"],
];

/** Guess a view's direction from its file name, else take the next unused one. */
function guessDirection(name: string): { az: number; el: number } {
  const n = name.toLowerCase();
  if (/\btop\b|_top|top_/.test(n)) return { az: 0, el: 90 };
  if (/bottom/.test(n)) return { az: 0, el: -90 };
  if (/back/.test(n)) return { az: 180, el: 0 };
  if (/left/.test(n)) return { az: 270, el: 0 };
  if (/right/.test(n)) return { az: 90, el: 0 };
  const used = new Set([mainAz, ...extraViews.map((v) => v.az)]);
  const az = [180, 90, 270, 45, 135, 225, 315].find((a) => !used.has(a)) ?? 180;
  return { az, el: 0 };
}

function addViews(files: File[]): void {
  for (const f of files) {
    if (!isImage(f)) {
      toast(`"${f.name}" isn't an image — skipped.`, "err");
      continue;
    }
    if (extraViews.length + 1 >= MAX_VIEWS) {
      toast(`At most ${MAX_VIEWS} images (front + ${MAX_VIEWS - 1} views) per generation.`, "err");
      break;
    }
    extraViews.push({ image: f, name: f.name || "view.png", ...guessDirection(f.name || "") });
  }
  renderViews();
}

/** Direction select + elevation input for one image; calls back with clamped values. */
function directionControls(
  az0: number,
  el0: number,
  onAz: (az: number) => void,
  onEl: (el: number) => void,
): [HTMLSelectElement, HTMLInputElement] {
  const az = document.createElement("select");
  az.title = "Azimuth: where the camera was, going around the object";
  for (const [deg, label] of AZIMUTHS) az.add(new Option(`${label} · ${deg}°`, String(deg)));
  if (!AZIMUTHS.some(([deg]) => deg === az0)) az.add(new Option(`${az0}°`, String(az0)));
  az.value = String(az0);
  az.addEventListener("change", () => onAz(parseFloat(az.value)));

  const el = document.createElement("input");
  el.type = "number";
  el.min = "-90";
  el.max = "90";
  el.step = "15";
  el.value = String(el0);
  el.title = "Elevation in degrees (90 = from above, -90 = from below)";
  el.addEventListener("change", () => {
    const e = Math.max(-90, Math.min(90, parseFloat(el.value) || 0));
    el.value = String(e);
    onEl(e);
  });
  return [az, el];
}

function azLabel(az: number): string {
  return `${AZIMUTHS.find(([deg]) => deg === az)?.[1] ?? "view"} · ${az}°`;
}

function updateFrontTag(): void {
  frontTag.textContent = `main · ${azLabel(mainAz)}${mainEl ? `, ${mainEl}° up` : ""}`;
  frontTag.classList.toggle("hidden", extraViews.length === 0 || !inputImage);
}

function renderViews(): void {
  viewUrls.forEach((u) => URL.revokeObjectURL(u));
  viewUrls = [];
  viewsList.innerHTML = "";
  if (extraViews.length) {
    viewsList.innerHTML = `<div class="vhead"><span></span><span>direction</span><span>elev °</span><span></span></div>`;
    // The main image's own direction (default front, level): the other views are
    // placed relative to the object the same way.
    const row = document.createElement("div");
    row.className = "vrow vmain";
    row.title = `${inputName} (main image)`;
    const img = document.createElement("img");
    if (inputImage) {
      const url = URL.createObjectURL(inputImage);
      viewUrls.push(url);
      img.src = url;
    }
    img.alt = "main image";
    row.appendChild(img);
    const [az, el] = directionControls(mainAz, mainEl,
      (a) => { mainAz = a; updateFrontTag(); },
      (e) => { mainEl = e; updateFrontTag(); });
    row.append(az, el);
    const tag = document.createElement("span");
    tag.className = "vmain-tag";
    tag.textContent = "main";
    row.appendChild(tag);
    viewsList.appendChild(row);
  }
  extraViews.forEach((v, i) => {
    const row = document.createElement("div");
    row.className = "vrow";
    row.title = v.name;

    const img = document.createElement("img");
    const url = URL.createObjectURL(v.image);
    viewUrls.push(url);
    img.src = url;
    img.alt = v.name;
    row.appendChild(img);

    const [az, el] = directionControls(v.az, v.el,
      (a) => (extraViews[i].az = a),
      (e) => (extraViews[i].el = e));
    row.append(az, el);

    const del = document.createElement("button");
    del.className = "vdel";
    del.type = "button";
    del.textContent = "✕";
    del.title = "Remove this view";
    del.addEventListener("click", () => {
      extraViews.splice(i, 1);
      renderViews();
    });
    row.appendChild(del);

    viewsList.appendChild(row);
  });
  const multi = extraViews.length > 0;
  updateFrontTag();
  clearViewsBtn.classList.toggle("hidden", !multi);
  mvAdv.classList.toggle("hidden", !multi);
  addViewBtn.disabled = extraViews.length + 1 >= MAX_VIEWS;
  generateBtn.textContent = multi ? `Generate 3D · ${extraViews.length + 1} views` : "Generate 3D";
}

addViewBtn.addEventListener("click", () => viewInput.click());
viewInput.addEventListener("change", () => {
  addViews(Array.from(viewInput.files ?? []));
  viewInput.value = "";
});
clearViewsBtn.addEventListener("click", () => {
  extraViews = [];
  renderViews();
});
["dragenter", "dragover"].forEach((ev) =>
  viewsEl.addEventListener(ev, (e) => {
    e.preventDefault();
    viewsEl.classList.add("drag");
  }),
);
viewsEl.addEventListener("dragleave", (e) => {
  if (!viewsEl.contains((e as DragEvent).relatedTarget as Node | null)) viewsEl.classList.remove("drag");
});
viewsEl.addEventListener("drop", (e) => {
  e.preventDefault();
  viewsEl.classList.remove("drag");
  addViews(Array.from((e as DragEvent).dataTransfer?.files ?? []));
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") fileInput.click();
});
fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files ?? []);
  if (files[0]) setInput(files[0], files[0].name);
  // several files picked at once: the first is the front, the rest extra views
  if (files.length > 1) addViews(files.slice(1));
  fileInput.value = "";
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
  const files = Array.from((e as DragEvent).dataTransfer?.files ?? []);
  const f = files[0];
  if (!f) return;
  if (isImage(f)) setInput(f, f.name);
  else toast(`"${f.name}" isn't an image — drop a PNG, JPEG or WebP.`, "err");
  // several files dropped at once: the first is the front, the rest extra views
  if (files.length > 1) addViews(files.slice(1));
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
  const nViews = rec.views?.length ?? 0;
  const stem = rec.name.replace(/\.[^.]+$/, "") || "model";

  // 1) Write the GLB to the output folder FIRST. This on-disk file is the real
  // deliverable and must survive any later failure — a dead gallery DB (some
  // WebKitGTK builds can't open IndexedDB) or a WebGL/model-viewer crash must
  // never cost the user a successful generation.
  let savedPath: string | null = null;
  if (isTauri()) {
    try {
      const bytes = new Uint8Array(await glb.arrayBuffer());
      const mv = nViews ? `_mv${nViews + 1}` : "";
      const fname = `${stem}_${rec.params.resolution}${mv}_seed${rec.params.seed}_${rec.id}.glb`;
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
  viewerCaption.textContent =
    `${rec.params.resolution}${viewsLabel(nViews)} · seed ${rec.params.seed} · ${(glb.size / 1e6).toFixed(1)} MB`;
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
  const views = extraViews.map((v) => ({ ...v }));
  const rec: GenRecord = {
    id: newId(),
    ts: Date.now(),
    name: inputName,
    params,
    input: inputImage,
    views: views.length ? views : undefined,
    glb: null,
    thumb: null,
  };
  await runJob(rec, (signal, hooks) => generate(rec.input, params, views, signal, hooks));
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

/** " · N views" for a generation with extra views, "" for a single image. */
function viewsLabel(extra: number): string {
  return extra ? ` · ${extra + 1} views` : "";
}

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
  extraViews = (rec.views ?? []).map((v) => ({ ...v }));
  renderViews();
  currentGlb = rec.glb;
  currentName = rec.name.replace(/\.[^.]+$/, "") || "model";
  activeId = rec.id;
  setViewerTools(true);
  viewerCaption.textContent =
    `${rec.name} · ${rec.params.resolution}${viewsLabel(rec.views?.length ?? 0)} · seed ${rec.params.seed}`;
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
    meta.textContent = r.pending
      ? "generating…"
      : `${r.params.resolution}${r.views?.length ? ` · ${r.views.length + 1}v` : ""}`;
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
