// Shared types for Trellis Studio.

export type Resolution = 512 | 1024 | 1536;
export type BgRemoval = "auto" | "birefnet" | "threshold";
export type Uv = "xatlas" | "box";
export type FrontAxis = "-y" | "+y" | "+x" | "-x" | "ref-z" | "ref-x";

/** The server accepts at most this many images per request (front + extra views). */
export const MAX_VIEWS = 12;

/** An additional photo of the same object, taken from another direction.
 *  az: 0 front, 90 right, 180 back, 270 left; el: +90 top, -90 bottom. */
export interface ExtraView {
  image: Blob;
  name: string;
  az: number;
  el: number;
}

/** Generation knobs mapped onto trellis-server's POST /generate form fields. */
export interface GenParams {
  resolution: Resolution;
  seed: number;
  bgRemoval: BgRemoval;
  uv: Uv;
  /** false => geometry only (server field texture=off). */
  texture: boolean;
  /** Multi-view blend sharpness (server field blend_temp; only sent with extra views). */
  blendTempOverride?: number;
  /** Multi-view front axis (server field front_axis; only sent with extra views). */
  frontAxisOverride?: FrontAxis;
  /** Camera direction of the main image (only sent with extra views; default 0, 0). */
  mainAz?: number;
  mainEl?: number;
}

export const DEFAULT_PARAMS: GenParams = {
  resolution: 1024,
  seed: 42,
  bgRemoval: "auto",
  uv: "xatlas",
  texture: true,
};

/** Written by the installer, read by the Tauri shell + surfaced to the UI. */
export interface AppConfig {
  serverBin: string;
  modelsDir: string;
  backend: string;
  gpu: number;
  host: string;
  port: number;
  /** where generated GLBs are auto-saved (Tauri only). */
  outputDir: string;
  /** true once the shell has a usable config.json; false => "setup needed". */
  configured: boolean;
}

/** One persisted generation (IndexedDB record). */
export interface GenRecord {
  id: string;
  ts: number;
  name: string;
  // Older records stored 2 and -y even when the request inherited server defaults.
  params: GenParams & { blendTemp?: number; frontAxis?: FrontAxis };
  input: Blob; // source image (the front view)
  views?: ExtraView[]; // extra views (multi-view); absent on single-image records
  glb: Blob | null; // resulting model/gltf-binary; null while pending
  /** true while the server is still working on it (the entry survives reloads). */
  pending?: boolean;
  /** trellis-server job id of a pending entry, to resume polling after a reload. */
  jobId?: string;
  thumb: Blob | null; // model-viewer snapshot for the gallery
}
