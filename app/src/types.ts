// Shared types for Trellis Studio.

export type Resolution = 512 | 1024 | 1536;
export type BgRemoval = "auto" | "birefnet" | "threshold";
export type Uv = "xatlas" | "box";
/** GLB texture encoding: WebP is smaller, PNG opens in every DCC tool / importer. */
export type TextureFormat = "webp" | "png";
/** Which flow weights generate the model (server field `model`, CLI `--model`). */
export type ModelFamily = "trellis" | "pixal3d";

/** Generation knobs mapped onto trellis-server's POST /generate form fields. */
export interface GenParams {
  resolution: Resolution;
  seed: number;
  bgRemoval: BgRemoval;
  uv: Uv;
  /**
   * QEM decimation face target (server field `faces`, CLI `--faces`). Studio
   * always sets it (options.ts presets). 0 or absent = the server's default
   * (300K at 1024/1536, 150K at 512): only gallery records from older builds.
   */
  faces?: number;
  /** UV atlas size in px (server field `atlas`, CLI `--atlas`). 0 or absent = server default, as for faces. */
  atlas?: number;
  /** Texture encoding (server field `webp=off` for PNG). Absent = WebP, the server default. */
  textures?: TextureFormat;
  /** false => geometry only (server field texture=off). */
  texture: boolean;
  /** Absent = "trellis" (gallery records from before the Model option). */
  model?: ModelFamily;
  /**
   * Pixal3D only: horizontal field of view of the input image in degrees
   * (server field `fov`, CLI `--fov`). Pixal3D projects the image into the
   * grid through a camera with this lens; a wrong value tilts straight image
   * edges (e.g. a bust's cut-off base) and drifts geometry off the silhouette.
   * Absent = PIXAL3D_DEFAULT_FOV (options.ts). Ignored for TRELLIS.2.
   */
  fov?: number;
}

/**
 * Texture format Studio always requests. It only changes how the colors are
 * stored in the GLB, not the model. The engine's default WebP (glTF extension
 * EXT_texture_webp) is far smaller, but a GLB that requires it won't open at
 * all in tools without that extension; PNG opens in every DCC tool and in
 * Unreal, which is where Studio's output goes. Not a user control.
 */
export const TEXTURE_FORMAT: TextureFormat = "png";

/** Starting settings for new images: the "Game" preset (options.ts) at res 1024. */
export const DEFAULT_PARAMS: GenParams = {
  resolution: 1024,
  seed: 42,
  bgRemoval: "auto",
  uv: "xatlas",
  faces: 50_000,
  atlas: 2048,
  textures: TEXTURE_FORMAT,
  texture: true,
  model: "trellis",
};

/** A random seed in the range the server accepts (uint32), for "new images get a random seed". */
export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31);
}

/** Compact face count: 49812 -> "49.8K", 300000 -> "300K", 1250000 -> "1.25M". */
export function formatFaces(n: number): string {
  if (n >= 1e6) return `${+(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${+(n / 1e3).toFixed(1)}K`;
  return String(n);
}

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
  params: GenParams;
  input: Blob; // source image
  glb: Blob; // resulting model/gltf-binary
  thumb: Blob | null; // model-viewer snapshot for the gallery
  /** Wall-clock generation time (request sent -> GLB received); absent on older records. */
  durationMs?: number;
}
