// Trellis Studio UI: wires the page (index.html) to the server, the queue, the
// viewer and the gallery. Layout follows the design handoff "Trellis Studio —
// Queue, Output Presets & Viewer". The queue keeps the stage-then-Generate
// flow: the first image waits for Generate, later drops queue behind it.

import "./ui.css";
import { generate, health } from "./api";
import { loadConfig } from "./config";
import { createSettingsEditor } from "./controls";
import { inspectGlb, type GlbInfo } from "./glb";
import { buildHandoff } from "./handoff";
import { fovOf, paramChips, tileLabel } from "./options";
import { getPrefs } from "./prefs";
import { GenerationQueue } from "./queue";
import { renderSettings } from "./settings";
import { StageTracker } from "./stages";
import { initTitlebar } from "./titlebar";
import { all, clear as clearStore, del as removeRecord, get as getRecord, isEphemeral, newId, put } from "./store";
import { isTauri, listen, revealInOutput, saveBytes, saveToOutputDir, writeHandoffFolder } from "./tauri";
import { DEFAULT_PARAMS, formatFaces, randomSeed, type GenParams, type GenRecord } from "./types";
import { Viewer } from "./viewer";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- element refs ----
const dropzone = $("dropzone");
const dropzoneTitle = $("dropzone-title");
const fileInput = $<HTMLInputElement>("file-input");
const inputPreview = $<HTMLImageElement>("input-preview");
const dropHint = $("dropzone-hint");
const generateBtn = $<HTMLButtonElement>("generate-btn");
const queueCount = $("queue-count");
const progress = $("progress");
const progressThumb = $<HTMLImageElement>("progress-thumb");
const progressName = $("progress-name");
const progressChips = $("progress-chips");
const progressStage = $("progress-stage");
const progressStageRow = $("progress-stage-row");
const progressPct = $("progress-pct");
const progressBar = $("progress-bar");
const progressLog = $("progress-log");
const progressElapsed = $("progress-elapsed");
const cancelBtn = $<HTMLButtonElement>("cancel-btn");
const viewerEmpty = $("viewer-empty");
const viewerInfo = $("viewer-info");
const viewerTag = $("viewer-tag");
const viewerName = $("viewer-name");
const viewerChips = $("viewer-chips");
const viewerStats = $("viewer-stats");
const infoBtn = $<HTMLButtonElement>("info-btn");
const infoPanel = $("info-panel");
const sourceBtn = $<HTMLButtonElement>("source-btn");
const sourcePanel = $("source-panel");
const sourceImg = $<HTMLImageElement>("source-img");
const shadeSeg = $("shade-seg");
const resetViewBtn = $<HTMLButtonElement>("reset-view");
const saveGlbBtn = $<HTMLButtonElement>("save-glb");
const exportBtn = $<HTMLButtonElement>("export-handoff");
const galleryEl = $("gallery");
const galleryCount = $("gallery-count");
const clearGalleryBtn = $("clear-gallery");
const backendBadge = $("backend-badge");
const serverDot = $("server-dot");
const serverLabel = $("server-label");
const setupBanner = $("setup-banner");

// ---- state ----
initTitlebar();
const viewer = new Viewer($("viewer-mount"));
let inputImage: Blob | null = null;
let inputName = "input.png";
let currentGlb: Blob | null = null;
let currentName = "model"; // input name of the result in the viewer (not of the running job)
let activeId: string | null = null;
let serverOnline = false;
let configured = false;
let generating = false;
/** Settings of the job currently running; null when idle. */
let runningParams: GenParams | null = null;
let abort: AbortController | null = null;
let elapsedTimer: number | null = null;
let galleryUrls: string[] = [];
let progressThumbUrl: string | null = null;
/** Stage + percent of the running job, from the server log (stages.ts); null when idle. */
let tracker: StageTracker | null = null;
let warnedEphemeral = false;

/** Images waiting behind the running job (see queue.ts). */
const queue = new GenerationQueue(
  {
    wrap: $("queue"),
    label: $("queue-label"),
    list: $("queue-list"),
    clearBtn: $("clear-queue"),
  },
  () => refreshQueueUi(),
);

// ---- settings editor (controls.ts) ----
/** Settings for the next generation. New images start from the defaults with a random seed. */
const editor = createSettingsEditor({ ...DEFAULT_PARAMS, seed: randomSeed() });
$("settings-editor").append(editor.el);

const readParams = (): GenParams => editor.get();

// ---- small helpers ----
function toast(msg: string, kind: "" | "ok" | "err" = ""): void {
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 8000 : 4000);
}

function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Fill a .chips container with one chip per setting. */
function renderChips(container: HTMLElement, chips: string[]): void {
  container.replaceChildren(
    ...chips.map((c) => {
      const span = document.createElement("span");
      span.className = "chip";
      span.textContent = c;
      return span;
    }),
  );
}

// ---- queue panel state ----
/** Count pill, drop zone title/size and the Generate button, from the current state. */
function refreshQueueUi(): void {
  const jobs = queue.size + (generating ? 1 : 0) + (!generating && inputImage ? 1 : 0);
  queueCount.textContent = String(jobs);
  const empty = jobs === 0;
  dropzone.classList.toggle("empty", empty && !inputImage);
  dropzoneTitle.textContent = empty ? "Drop an image to start" : "Drop images to add to queue";
  updateGenerateEnabled();
}

function updateGenerateEnabled(): void {
  generateBtn.disabled = !(serverOnline && (inputImage || queue.size > 0) && !generating);
}

// ---- input image ----
function setInput(blob: Blob, name: string): void {
  inputImage = blob;
  inputName = name || "input.png";
  inputPreview.src = URL.createObjectURL(blob);
  inputPreview.classList.remove("hidden");
  dropHint.classList.add("hidden");
  dropzone.classList.add("has-input");
  refreshQueueUi();
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") fileInput.click();
});
fileInput.addEventListener("change", () => {
  acceptFiles([...(fileInput.files ?? [])]);
  fileInput.value = ""; // let the same file be picked again
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
const isGlb = (f: File) => f.type === "model/gltf-binary" || /\.glb$/i.test(f.name);

/**
 * Single entry point for every new image (browse, drop, paste).
 * Idle: the image becomes the input, ready for Generate.
 * Generating: it joins the queue and runs after the current job.
 */
function acceptImage(image: Blob, name: string): void {
  if (generating) enqueue(image, name);
  else setInput(image, name);
}

/**
 * Browse / drop entry point for one or more files. A .glb opens in the viewer
 * (see openGlb); other non-images are skipped with a toast. With several
 * images, only the first can become the input; the rest always queue behind it.
 */
function acceptFiles(files: File[]): void {
  const glbs = files.filter(isGlb);
  const images = files.filter(isImage);
  const skipped = files.length - images.length - glbs.length;
  if (skipped) {
    toast(
      skipped === 1 && files.length === 1
        ? `"${files[0].name}" isn't an image or a .glb — drop a PNG, JPEG, WebP or GLB.`
        : `Skipped ${skipped} unsupported file${skipped > 1 ? "s" : ""}.`,
      "err",
    );
  }
  if (glbs.length) {
    void openGlb(glbs[0]);
    if (glbs.length > 1) toast(`Opened "${glbs[0].name}": the viewer shows one model at a time.`);
  }
  // Only the first can become the input; the rest always queue (when idle they'd
  // otherwise just replace each other in the preview).
  images.forEach((f, i) => (i === 0 ? acceptImage(f, f.name) : enqueue(f, f.name)));
}

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
  acceptFiles([...((e as DragEvent).dataTransfer?.files ?? [])]);
});
// The viewer is a drop target too: a .glb opens there, an image goes to the queue.
const viewerPanel = document.querySelector<HTMLElement>(".viewer-panel")!;
["dragenter", "dragover"].forEach((ev) =>
  viewerPanel.addEventListener(ev, (e) => {
    e.preventDefault();
    viewerPanel.classList.add("drag");
  }),
);
viewerPanel.addEventListener("dragleave", (e) => {
  if (!viewerPanel.contains((e as DragEvent).relatedTarget as Node | null)) viewerPanel.classList.remove("drag");
});
viewerPanel.addEventListener("drop", (e) => {
  e.preventDefault();
  viewerPanel.classList.remove("drag");
  acceptFiles([...((e as DragEvent).dataTransfer?.files ?? [])]);
});
window.addEventListener("paste", async (e: ClipboardEvent) => {
  for (const item of e.clipboardData?.items ?? []) {
    const image = item.type.startsWith("image/") && item.getAsFile();
    if (image) return acceptImage(image, image.name || "pasted.png");
  }
  // WebKitGTK can omit image data from ClipboardEvent. Since paste was
  // user-triggered, try reading the clipboard item list for an image.
  const items = await navigator.clipboard.read().catch(() => []);
  for (const item of items) {
    const type = item.types.find((type) => type.startsWith("image/"));
    if (type) {
      const image = await item.getType(type).catch(() => null);
      if (image) return acceptImage(image, "pasted.png");
    }
  }
});

// ---- queue ----
/**
 * Queue an image. Its starting settings come from one of two sources
 * (Settings → "Queued images use the same settings…", prefs.ts), and its
 * card in "Up next" can change them until it starts (queue.ts):
 *   - on, while a job runs: a copy of the running job's settings, so a batch
 *     dropped onto a running generation all comes out the same;
 *   - otherwise: whatever the settings panel shows right now.
 */
function enqueue(image: Blob, name: string): void {
  const params = getPrefs().queueSameSettings && runningParams ? { ...runningParams } : readParams();
  queue.add(image, name, params);
}

/**
 * Start the next queued image, if any, with the settings it was queued with.
 *
 * Note: if the server is offline this pauses instead, keeping the queue
 * intact. Otherwise every remaining item would fail instantly in turn and the
 * whole batch would be lost to error toasts. Generate resumes it.
 */
function startNextQueued(): void {
  if (generating || queue.size === 0) return;
  if (!serverOnline) {
    toast(`Server offline — queue paused (${queue.size} waiting). Press Generate to resume.`, "err");
    return;
  }
  const next = queue.take()!;
  setInput(next.image, next.name);
  void doGenerate(next.params);
}

// ---- viewer chrome ----
/**
 * Open or close the model details card behind the (i) button. It starts
 * closed so the model has the whole viewer, and stays as the user left it
 * when another model loads.
 */
function setInfoOpen(open: boolean): void {
  infoPanel.classList.toggle("hidden", !open);
  infoBtn.setAttribute("aria-expanded", String(open));
  infoBtn.title = open ? "Hide model details" : "Model details";
}
infoBtn.addEventListener("click", () => setInfoOpen(infoPanel.classList.contains("hidden")));
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!infoPanel.classList.contains("hidden")) setInfoOpen(false);
  if (!sourcePanel.classList.contains("hidden")) setSourceOpen(false);
});

/**
 * Open or close the source image card behind the image button. Like the
 * details card, it stays as the user left it when another model loads.
 */
function setSourceOpen(open: boolean): void {
  sourcePanel.classList.toggle("hidden", !open);
  sourceBtn.setAttribute("aria-expanded", String(open));
  sourceBtn.title = open ? "Hide source image" : "Source image";
}
sourceBtn.addEventListener("click", () => setSourceOpen(sourcePanel.classList.contains("hidden")));

let sourceUrl: string | null = null;
/**
 * The image the shown model was generated from, or null (a .glb opened from
 * disk has none): that hides the image button and its card.
 */
function setSource(image: Blob | null): void {
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = image ? URL.createObjectURL(image) : null;
  if (sourceUrl) sourceImg.src = sourceUrl;
  else sourceImg.removeAttribute("src");
  sourceBtn.classList.toggle("hidden", !image);
  if (!image) sourcePanel.classList.add("hidden");
  else if (sourceBtn.getAttribute("aria-expanded") === "true") sourcePanel.classList.remove("hidden");
}

/**
 * What a GLB actually contains, as stats-pill parts: face count, atlas size
 * (or "no texture" for a geometry-only model) and file size.
 */
async function glbStats(glb: Blob): Promise<{ info: GlbInfo | null; parts: string[] }> {
  const info = await inspectGlb(glb).catch(() => null);
  const base = info?.images.find((i) => i.role === "basecolor");
  const bmp = base ? await createImageBitmap(base.data).catch(() => null) : null;
  const parts: string[] = [];
  if (info) parts.push(`${formatFaces(info.faces)} faces`);
  if (bmp) parts.push(`${bmp.width} px atlas`);
  else if (info && !base) parts.push("no texture");
  bmp?.close();
  parts.push(`${(glb.size / 1e6).toFixed(1)} MB`);
  return { info, parts };
}

/** Fill the viewer's info card: status tag, name, chips and stats. */
function setViewerInfo(tag: string, name: string, chips: string[], stats: string[]): void {
  viewerEmpty.classList.add("hidden");
  viewerInfo.classList.remove("hidden");
  viewerTag.classList.remove("live");
  viewerTag.textContent = tag;
  viewerName.textContent = name;
  renderChips(viewerChips, chips);
  viewerStats.textContent = stats.join(" · ");
}

/**
 * Show a finished result's info around the viewer: status tag, name, chips
 * and the stats pill with the values the GLB actually contains.
 *
 * @returns what the GLB contains, or null if it couldn't be read.
 */
async function showResultInfo(name: string, params: GenParams, glb: Blob, durationMs?: number): Promise<GlbInfo | null> {
  const { info, parts } = await glbStats(glb);
  setViewerInfo(
    durationMs != null ? `Completed · ${fmtElapsed(durationMs)}` : "Completed",
    name,
    paramChips(params),
    [`${params.resolution}`, `seed ${params.seed}`, ...parts],
  );
  return info;
}

/**
 * Open a .glb from disk in the viewer: a trellis-cli output, an older export,
 * a model from another tool. Shading, Reset view, Model details and Save GLB
 * all work. It isn't added to the gallery (that holds generations, with their
 * source image and settings), and Export handoff stays off because it needs both.
 */
async function openGlb(file: File): Promise<void> {
  try {
    await viewer.load(file);
  } catch (e) {
    toast(`Couldn't open "${file.name}": ${(e as Error).message}`, "err");
    return;
  }
  currentGlb = file;
  currentName = file.name;
  activeId = null;
  setSource(null);
  setViewerTools(true);
  exportBtn.disabled = true;
  const { parts } = await glbStats(file);
  setViewerInfo("Opened from file", file.name, [], parts);
  await refreshGallery(); // drop the gallery highlight: the viewer no longer shows a record
}

/**
 * Warn when the result shows the server ignored part of the request.
 *
 * Note: trellis-server ignores form fields it doesn't know, so a build that
 * predates `faces`/`atlas` returns its defaults with no error. Checks:
 *   - faces: QEM can stop slightly above its target on hard geometry, so only
 *     a clear miss (more than 1.5x) counts;
 *   - atlas: the baked textures are exactly atlas x atlas (uv_bake.cpp), so any
 *     difference counts;
 *   - PNG: any WebP texture means `webp=off` was not applied.
 */
async function checkServerHonored(params: GenParams, info: GlbInfo | null): Promise<void> {
  if (!info) return;
  const missed: string[] = [];
  if (params.faces && info.faces > params.faces * 1.5) {
    missed.push(`${formatFaces(params.faces)} faces (got ${formatFaces(info.faces)})`);
  }
  const base = info.images.find((i) => i.role === "basecolor");
  if (params.atlas && base) {
    const bmp = await createImageBitmap(base.data).catch(() => null);
    if (bmp && bmp.width !== params.atlas) missed.push(`${params.atlas} px atlas (got ${bmp.width})`);
    bmp?.close();
  }
  if (params.textures === "png" && info.images.some((i) => i.mimeType === "image/webp")) {
    missed.push("PNG textures (got WebP)");
  }
  if (missed.length) {
    toast(
      `The server ignored: ${missed.join(", ")}. This trellis-server probably predates ` +
        "these settings; point Settings at a newer build.",
      "err",
    );
  }
}

// ---- generate ----
/** Show the running-job card for a job that's starting. */
function showRunningCard(image: Blob, name: string, params: GenParams): void {
  if (progressThumbUrl) URL.revokeObjectURL(progressThumbUrl);
  progressThumbUrl = URL.createObjectURL(image);
  progressThumb.src = progressThumbUrl;
  progressName.textContent = name;
  renderChips(progressChips, paramChips(params));
  progressStage.textContent = "Starting…";
  progressPct.textContent = "";
  progressBar.classList.add("indeterminate");
  progressBar.style.width = "";
  progressLog.textContent = "";
  progressElapsed.textContent = "0:00";
  progress.classList.remove("hidden");
}

/**
 * Show the tracker's stage and percent: "Shape latent · 4 of 8", "37%".
 * Before the first stage header (and always in a plain browser, which gets no
 * server log) the bar stays indeterminate.
 */
function renderStage(): void {
  const p = tracker?.progress();
  if (!p) return;
  progressStage.textContent = `${p.name} · ${p.index + 1} of ${p.total}`;
  progressPct.textContent = `${p.percent}%`;
  progressBar.classList.remove("indeterminate");
  progressBar.style.width = `${p.percent}%`;
}

/** The raw server log line under the stage, hidden by default; click the stage row to toggle. */
if (isTauri()) {
  progressStageRow.classList.add("has-log");
  progressStageRow.title = "Show the server log line";
  progressStageRow.addEventListener("click", () => {
    const show = progressLog.classList.toggle("hidden") === false;
    progressStageRow.title = show ? "Hide the server log line" : "Show the server log line";
  });
}

/**
 * Generate a model from the current input, then hand off to the queue.
 *
 * @param params Settings for this run. Defaults to the settings panel (the
 *               Generate button); queued jobs pass the settings they were
 *               queued with.
 *
 * Note: the input image and name are captured at the start. Drops and pastes
 * during the run go to the queue, but `inputImage` still changes when the
 * queue advances, so the result must be saved under this job's own input.
 */
async function doGenerate(params: GenParams = readParams()): Promise<void> {
  if (generating) return;
  if (!inputImage) return startNextQueued(); // Generate with only a queue: start it
  const image = inputImage;
  const name = inputName;
  generating = true;
  runningParams = params;
  dropzone.classList.add("queueing");
  tracker = new StageTracker(params.resolution);
  showRunningCard(image, name, params);
  // The staged image is now the running job: clear the drop zone for new images.
  inputImage = null;
  inputPreview.classList.add("hidden");
  dropHint.classList.remove("hidden");
  dropzone.classList.remove("has-input");
  refreshQueueUi();
  refreshStatus();
  const started = Date.now();
  elapsedTimer = window.setInterval(() => {
    progressElapsed.textContent = fmtElapsed(Date.now() - started);
    renderStage(); // time-based stages advance between log lines
  }, 1000);
  abort = new AbortController();
  try {
    const { glb } = await generate(image, params, abort.signal);
    const durationMs = Date.now() - started;
    // The model is done; what's left (save, gallery, preview) is local. Freeze the
    // card's timer at the real generation time instead of letting it run on.
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    progressElapsed.textContent = fmtElapsed(durationMs);
    tracker?.finish();
    progressStage.textContent = "Loading preview…";
    progressPct.textContent = "100%";
    progressBar.classList.remove("indeterminate");
    progressBar.style.width = "100%";
    currentGlb = glb;
    currentName = name;
    setSource(image);

    const rec: GenRecord = {
      id: newId(),
      ts: Date.now(),
      name,
      params,
      input: image,
      glb,
      thumb: null,
      durationMs,
    };

    // 1) Write the GLB to the output folder FIRST. This on-disk file is the real
    // deliverable and must survive any later failure — a dead gallery DB (some
    // WebKitGTK builds can't open IndexedDB) or a WebGL/model-viewer crash must
    // never cost the user a successful generation.
    let savedPath: string | null = null;
    if (isTauri()) {
      try {
        const bytes = new Uint8Array(await glb.arrayBuffer());
        const base = name.replace(/\.[^.]+$/, "") || "model";
        const fam = params.model === "pixal3d" ? `_pixal3d-fov${+fovOf(params).toFixed(2)}` : "";
        const fname = `${base}${fam}_${params.resolution}_seed${params.seed}_${rec.id}.glb`;
        savedPath = await saveToOutputDir(fname, bytes);
      } catch (e) {
        toast(`Auto-save to output folder failed: ${(e as Error).message}`, "err");
      }
    }
    const took = fmtElapsed(durationMs);
    toast(savedPath ? `Done in ${took} — saved to ${savedPath}` : `Generation complete in ${took}`, "ok");

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
    activeId = rec.id;
    setViewerTools(true);
    void checkServerHonored(params, await showResultInfo(name, params, glb, durationMs));
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
  } catch (e) {
    if (abort?.signal.aborted) {
      toast(queue.size ? `Generation cancelled — queue paused (${queue.size} waiting)` : "Generation cancelled");
    } else toast(`${name}: ${(e as Error).message || "generation failed"}`, "err");
  } finally {
    generating = false;
    runningParams = null;
    tracker = null;
    dropzone.classList.remove("queueing");
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    progress.classList.add("hidden");
    refreshQueueUi();
    refreshStatus();
  }
  // Success or failure, move on to the next queued image: one bad input
  // shouldn't stall a batch. Cancel pauses instead, since the user asked to stop.
  if (!abort?.signal.aborted) startNextQueued();
}

generateBtn.addEventListener("click", () => void doGenerate());
cancelBtn.addEventListener("click", () => abort?.abort());

// ---- viewer tools ----
function setViewerTools(on: boolean): void {
  resetViewBtn.disabled = !on;
  saveGlbBtn.disabled = !on;
  // The handoff export writes a folder on disk, so it needs the desktop app.
  exportBtn.disabled = !on || !isTauri();
}
resetViewBtn.addEventListener("click", () => viewer.resetView());

type Shade = "textured" | "clay" | "wireframe" | "colorwire";

/**
 * Shading (Textured | Clay | Wireframe | Color wire), one exclusive choice.
 * Wireframe draws dark triangles over clay, which reads best for density;
 * Color wire draws the triangles alone in the texture's colors, which shows
 * where each color sits, hidden parts included. Both rely on model-viewer
 * internals (viewer.ts), so if they're unavailable those options disable
 * themselves and the view falls back to clay.
 */
function setShade(shade: Shade): void {
  viewer.setClay(shade === "clay" || shade === "wireframe");
  const wire = shade === "wireframe" || shade === "colorwire";
  if (!viewer.setWireframe(wire, shade === "colorwire" ? "textured" : "overlay") && wire) {
    shadeSeg.querySelectorAll<HTMLButtonElement>('[data-shade="wireframe"], [data-shade="colorwire"]').forEach((b) => {
      b.disabled = true;
      b.title = "Wireframe isn't available with this version of the 3D viewer";
    });
    toast("Wireframe isn't available with this version of the 3D viewer.", "err");
    shade = "clay";
  }
  shadeSeg.querySelectorAll<HTMLButtonElement>("[data-shade]").forEach((b) => {
    const on = b.dataset.shade === shade;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
  });
}
shadeSeg.querySelectorAll<HTMLButtonElement>("[data-shade]").forEach((b) =>
  b.addEventListener("click", () => setShade(b.dataset.shade as Shade)),
);

/**
 * Export a modeler handoff package (handoff.ts) for the result in the viewer,
 * then open its folder. The turnaround sheets are rendered from the live
 * viewer, so the camera moves briefly while it runs.
 */
exportBtn.addEventListener("click", async () => {
  const rec = activeId ? await getRecord(activeId) : undefined;
  if (!rec) return;
  exportBtn.disabled = true;
  try {
    const pkg = await buildHandoff(rec, viewer);
    const dir = await writeHandoffFolder(pkg.folder, pkg.files);
    toast(`Handoff exported (${pkg.files.length} files) to ${dir}`, "ok");
    if (pkg.webpTextures) {
      toast(
        "This model has WebP textures, which some 3D tools can't open. It was generated " +
          "before Studio switched to PNG; regenerate it for the modeler.",
        "err",
      );
    }
    await revealInOutput(dir).catch(() => {});
  } catch (e) {
    toast(`Handoff export failed: ${(e as Error).message ?? e}`, "err");
  } finally {
    exportBtn.disabled = false;
  }
});

saveGlbBtn.addEventListener("click", async () => {
  if (!currentGlb) return;
  const bytes = new Uint8Array(await currentGlb.arrayBuffer());
  const base = currentName.replace(/\.[^.]+$/, "") || "model";
  const ok = await saveBytes(`${base}.glb`, bytes);
  if (ok) toast("Saved", "ok");
});

// ---- gallery ----
async function loadRecord(id: string): Promise<void> {
  const rec = await getRecord(id);
  if (!rec) return;
  try {
    await viewer.load(rec.glb);
  } catch (e) {
    toast((e as Error).message, "err");
    return;
  }
  // Only the viewer changes: the "New images use" settings stay as they are.
  currentGlb = rec.glb;
  currentName = rec.name;
  activeId = rec.id;
  setSource(rec.input);
  setViewerTools(true);
  await showResultInfo(rec.name, rec.params, rec.glb, rec.durationMs);
  await refreshGallery();
}

/** Rebuild the gallery strip: 128x88 tiles with "res · preset" and duration, name below. */
async function refreshGallery(): Promise<void> {
  galleryUrls.forEach((u) => URL.revokeObjectURL(u));
  galleryUrls = [];
  const recs = await all();
  galleryCount.textContent = String(recs.length);
  galleryEl.innerHTML = "";
  if (!recs.length) {
    galleryEl.innerHTML = `<div class="gallery-empty">No generations yet.</div>`;
    return;
  }
  for (const r of recs) {
    const tile = document.createElement("div");
    tile.className = "gtile" + (r.id === activeId ? " active" : "");
    tile.title = `${r.name} · ${new Date(r.ts).toLocaleString()}`;

    const frame = document.createElement("div");
    frame.className = "gtile-img";
    const img = document.createElement("img");
    const url = URL.createObjectURL(r.thumb ?? r.input);
    galleryUrls.push(url);
    img.src = url;
    img.alt = r.name;
    frame.appendChild(img);

    const left = document.createElement("span");
    left.className = "gtile-tag left";
    left.textContent = tileLabel(r.params);
    frame.appendChild(left);
    if (r.durationMs != null) {
      const right = document.createElement("span");
      right.className = "gtile-tag right";
      right.textContent = fmtElapsed(r.durationMs);
      frame.appendChild(right);
    }

    const delBtn = document.createElement("button");
    delBtn.className = "gdel";
    delBtn.textContent = "✕";
    delBtn.title = "Delete from gallery";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await removeRecord(r.id);
      if (activeId === r.id) activeId = null;
      await refreshGallery();
    });
    frame.appendChild(delBtn);

    const nameEl = document.createElement("span");
    nameEl.className = "gtile-name";
    nameEl.textContent = r.name;

    tile.append(frame, nameEl);
    tile.addEventListener("click", () => loadRecord(r.id));
    galleryEl.appendChild(tile);
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
/** Header status: dot + "ready" / "generating" / "offline" / "setup needed". */
function refreshStatus(): void {
  const state = generating ? "busy" : serverOnline ? "ok" : "err";
  serverDot.className = `status-dot ${state}`;
  serverLabel.textContent = generating
    ? "generating"
    : serverOnline
      ? "ready"
      : configured
        ? "offline"
        : "setup needed";
}

async function pollHealth(): Promise<void> {
  const cfg = await loadConfig(true);
  configured = cfg.configured;
  if (isTauri()) backendBadge.textContent = cfg.backend !== "unknown" ? cfg.backend.toUpperCase() : "—";
  serverOnline = await health();
  refreshStatus();
  const needSetup = !serverOnline && !cfg.configured;
  setupBanner.classList.toggle("hidden", serverOnline);
  if (needSetup) {
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Trellis Studio isn't set up yet — run the installer or point it at your models directory.";
  } else if (!serverOnline) {
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Server is offline — it may still be loading, or check the models directory in settings.";
  }
  updateGenerateEnabled();
}

// ---- server log -> running card (Tauri only) ----
listen<string>("server-log", (line) => {
  const t = String(line).trim();
  if (!generating || !t) return;
  progressLog.textContent = t;
  if (tracker?.feed(t)) renderStage();
});

// ---- boot ----
async function boot(): Promise<void> {
  setViewerTools(false);
  refreshQueueUi();
  await refreshGallery();
  await pollHealth();
  window.setInterval(pollHealth, 4000);
  if (!isTauri()) backendBadge.textContent = "BROWSER"; // no shell to report a backend
}
boot();
