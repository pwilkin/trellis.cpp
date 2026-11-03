// Wrapper around Google's <model-viewer> web component (vendored, three.js-based).
// It renders the WebP-textured PBR GLBs that trellis produces, and `camera-controls`
// gives the rotate + zoom the previewer needs.
//
// Beyond plain viewing it offers:
//   - Clay mode: every material shown as flat matte grey, to judge the form
//     without the paint (uses model-viewer's materials API, so nothing is
//     reloaded and the original textures are restored exactly);
//   - Wireframe: the mesh's triangles drawn over the shaded surface, to see
//     the face density a face target produces (works together with clay);
//   - Color wire: the model's own textured materials drawn as lines only, so
//     every edge takes the texture's color and parts hidden behind other
//     surfaces (e.g. eyes under a lid) show through;
//   - Turnaround sheets: four views around the model composited into one
//     labelled PNG, for the modeler handoff package.

// Minimal structural types for the parts of model-viewer's materials API we use
// (model-viewer 2.x: `model.materials[i].pbrMetallicRoughness`).
type RGBA = [number, number, number, number];
interface MvTextureInfo {
  texture: unknown | null;
  setTexture(texture: unknown | null): void;
}
interface MvMaterial {
  pbrMetallicRoughness: {
    baseColorFactor: RGBA;
    metallicFactor: number;
    roughnessFactor: number;
    baseColorTexture: MvTextureInfo;
    metallicRoughnessTexture: MvTextureInfo;
    setBaseColorFactor(rgba: RGBA): void;
    setMetallicFactor(v: number): void;
    setRoughnessFactor(v: number): void;
  };
}

// Minimal structural types for the three.js objects inside model-viewer's
// scene that the wireframe overlay touches. model-viewer bundles its own
// three.js, so new objects are built from existing ones' constructors.
interface ThreeMaterial {
  isMeshStandardMaterial?: boolean;
  wireframe: boolean;
  transparent: boolean;
  opacity: number;
  depthWrite: boolean;
  polygonOffset: boolean;
  polygonOffsetFactor: number;
  polygonOffsetUnits: number;
  metalness: number;
  roughness: number;
  color: { setRGB(r: number, g: number, b: number): void };
  dispose(): void;
}
interface ThreeObject {
  isMesh?: boolean;
  userData: Record<string, unknown>;
  geometry?: unknown;
  material?: ThreeMaterial | ThreeMaterial[];
  parent: ThreeObject | null;
  add(child: ThreeObject): void;
  remove(child: ThreeObject): void;
}
interface ThreeScene {
  traverse(fn: (o: ThreeObject) => void): void;
  queueRender(): void;
}

// Minimal structural type for the bits of <model-viewer> we touch.
interface ModelViewerEl extends HTMLElement {
  model?: { materials: MvMaterial[] };
  src: string;
  cameraOrbit: string;
  cameraTarget: string;
  fieldOfView: string;
  jumpCameraToGoal(): void;
  resetTurntableRotation(theta?: number): void;
  toBlob(options?: {
    mimeType?: string;
    qualityArgument?: number;
    idealAspect?: boolean;
  }): Promise<Blob>;
}

const DEFAULT_ORBIT = "0deg 75deg 105%";

/** Clay look: mid warm grey (lighter clips to near-white under the neutral environment), rough, non-metallic. */
const CLAY_COLOR: RGBA = [0.42, 0.42, 0.4, 1];
const CLAY_ROUGHNESS = 0.9;

/** Wireframe lines: dark, semi-transparent, drawn over the surface. */
const WIRE_RGB: [number, number, number] = [0.06, 0.07, 0.08];
const WIRE_OPACITY = 0.5;
const WIRE_TAG = "trellisWireframe"; // userData flag marking our overlay meshes

/**
 * How the wireframe is drawn: "overlay" = dark lines over the surface (pairs
 * with clay); "textured" = the model's own materials switched to lines, in
 * the texture's colors, with no surface.
 */
export type WireStyle = "overlay" | "textured";

/**
 * model-viewer's three.js scene. NOT public API: model-viewer stores it on the
 * element under a module-private `Symbol("scene")`, and its `queueRender()`
 * asks for a redraw. Both exist in the vendored build (public/vendor); if an
 * update removes them this returns null and the wireframe toggle reports
 * itself unsupported instead of failing.
 */
function threeSceneOf(el: HTMLElement): ThreeScene | null {
  const key = Object.getOwnPropertySymbols(el).find((s) => s.description === "scene");
  const scene = key ? (el as unknown as Record<symbol, unknown>)[key] : null;
  const ok =
    !!scene &&
    typeof (scene as ThreeScene).traverse === "function" &&
    typeof (scene as ThreeScene).queueRender === "function";
  return ok ? (scene as ThreeScene) : null;
}

/**
 * Turnaround views: azimuth around the model, at the same 75° elevation the
 * repo's showcase sheets use (tools/mv_preview). 0° looks at the glTF front (+Z).
 */
const SHEET_VIEWS = [
  { theta: 0, label: "0° front" },
  { theta: 90, label: "90° side" },
  { theta: 180, label: "180° back" },
  { theta: 270, label: "270° side" },
] as const;

/** A material's original look, saved while clay mode is on. */
interface SavedMaterial {
  material: MvMaterial;
  baseColorTexture: unknown | null;
  metallicRoughnessTexture: unknown | null;
  baseColorFactor: RGBA;
  metallicFactor: number;
  roughnessFactor: number;
}

/** Wait for model-viewer to draw at least one fresh frame after a change. */
async function settle(ms = 250): Promise<void> {
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  await new Promise((r) => setTimeout(r, ms));
}

// Load the vendored model-viewer module once, at runtime, from public/vendor
// (resolved relative to the document so it works in both vite dev and the Tauri
// bundle). Returns a promise that settles when the <model-viewer> element is
// registered, so callers can safely await before using its methods.
let mvReady: Promise<void> | null = null;
function ensureModelViewer(): Promise<void> {
  if (mvReady) return mvReady;
  mvReady = new Promise<void>((resolve, reject) => {
    const s = document.createElement("script");
    s.type = "module";
    s.src = new URL("vendor/model-viewer.min.js", document.baseURI).href;
    s.onerror = () => reject(new Error("failed to load model-viewer"));
    document.head.appendChild(s);
    customElements.whenDefined("model-viewer").then(() => resolve());
  });
  return mvReady;
}
// Kick off loading as soon as the module is imported.
ensureModelViewer().catch(() => {});

export class Viewer {
  private el: ModelViewerEl;
  private objectUrl: string | null = null;
  /** Whether clay mode is wanted; it stays on across model loads. */
  private clay = false;
  /** Original materials of the current model while clay is applied; null otherwise. */
  private saved: SavedMaterial[] | null = null;
  /** Whether the wireframe overlay is wanted; it stays on across model loads. */
  private wire = false;
  /** Overlay meshes added for the current model, and the surface materials we offset. */
  private overlays: ThreeObject[] = [];
  private offsetMaterials: ThreeMaterial[] = [];
  /** Which wireframe look is wanted; it stays across model loads. */
  private wireStyle: WireStyle = "overlay";
  /** Model materials switched to wireframe by the "textured" style. */
  private wiredMaterials: ThreeMaterial[] = [];

  constructor(mount: HTMLElement) {
    const mv = document.createElement("model-viewer") as ModelViewerEl;
    mv.setAttribute("camera-controls", "");
    mv.setAttribute("interaction-prompt", "none");
    mv.setAttribute("shadow-intensity", "1");
    mv.setAttribute("exposure", "1");
    mv.setAttribute("camera-orbit", DEFAULT_ORBIT);
    mv.setAttribute("min-field-of-view", "5deg");
    mv.setAttribute("touch-action", "pan-y");
    mv.classList.add("viewer");
    mount.appendChild(mv);
    this.el = mv;
  }

  /** Load a GLB blob; resolves once model-viewer has it rendered. */
  async load(glb: Blob): Promise<void> {
    await ensureModelViewer();
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = URL.createObjectURL(glb);
    return new Promise((resolve, reject) => {
      const onLoad = () => {
        cleanup();
        this.saved = null; // new model, new materials
        if (this.clay) this.applyClay();
        this.removeWireframe(); // the old overlays left with the old model
        if (this.wire) this.applyWireframe();
        resolve();
      };
      const onError = (e: Event) => {
        cleanup();
        const detail = (e as CustomEvent).detail;
        reject(new Error(String(detail?.sourceError ?? "failed to load model")));
      };
      // Guard against model-viewer never firing load/error (e.g. a stalled WebGL
      // init) so the caller's flow can't hang forever.
      const timer = window.setTimeout(() => {
        cleanup();
        reject(new Error("3D preview timed out"));
      }, 30000);
      const cleanup = () => {
        window.clearTimeout(timer);
        this.el.removeEventListener("load", onLoad);
        this.el.removeEventListener("error", onError);
      };
      this.el.addEventListener("load", onLoad);
      this.el.addEventListener("error", onError);
      this.el.src = this.objectUrl!;
    });
  }

  clear(): void {
    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }
    this.el.src = "";
  }

  resetView(): void {
    this.el.cameraOrbit = DEFAULT_ORBIT;
    this.el.cameraTarget = "auto auto auto";
    this.el.fieldOfView = "auto";
    this.el.resetTurntableRotation(0);
    this.el.jumpCameraToGoal();
  }

  /**
   * A PNG snapshot of the current view for the gallery thumbnail. Always the
   * textured look, so thumbnails stay recognisable even while clay is on.
   */
  async thumbnail(): Promise<Blob | null> {
    const wasClay = this.clay;
    const wasWire = this.wire;
    try {
      if (wasClay) this.setClay(false);
      if (wasWire) this.setWireframe(false);
      // model-viewer's 'load' fires before its auto-framing camera animation has
      // settled, so an immediate toBlob can capture a mid-frame view; give the
      // framing a beat to finish before snapshotting.
      await new Promise((r) => setTimeout(r, 400));
      return await this.el.toBlob({ mimeType: "image/png", idealAspect: true });
    } catch {
      return null;
    } finally {
      if (wasClay) this.setClay(true);
      if (wasWire) this.setWireframe(true);
    }
  }

  // ---- clay mode ----

  /** Whether clay mode is on. */
  get isClay(): boolean {
    return this.clay;
  }

  /** Show the model as untextured matte grey (true) or with its textures (false). */
  setClay(on: boolean): void {
    this.clay = on;
    if (on) this.applyClay();
    else this.restoreMaterials();
  }

  private applyClay(): void {
    const materials = this.el.model?.materials;
    if (!materials || this.saved) return; // no model yet, or already clay
    this.saved = materials.map((material) => {
      const pbr = material.pbrMetallicRoughness;
      return {
        material,
        baseColorTexture: pbr.baseColorTexture.texture,
        metallicRoughnessTexture: pbr.metallicRoughnessTexture.texture,
        baseColorFactor: [...pbr.baseColorFactor] as RGBA,
        metallicFactor: pbr.metallicFactor,
        roughnessFactor: pbr.roughnessFactor,
      };
    });
    for (const { material } of this.saved) {
      const pbr = material.pbrMetallicRoughness;
      pbr.baseColorTexture.setTexture(null);
      pbr.metallicRoughnessTexture.setTexture(null);
      pbr.setBaseColorFactor(CLAY_COLOR);
      pbr.setMetallicFactor(0);
      pbr.setRoughnessFactor(CLAY_ROUGHNESS);
    }
  }

  private restoreMaterials(): void {
    for (const s of this.saved ?? []) {
      const pbr = s.material.pbrMetallicRoughness;
      pbr.baseColorTexture.setTexture(s.baseColorTexture);
      pbr.metallicRoughnessTexture.setTexture(s.metallicRoughnessTexture);
      pbr.setBaseColorFactor(s.baseColorFactor);
      pbr.setMetallicFactor(s.metallicFactor);
      pbr.setRoughnessFactor(s.roughnessFactor);
    }
    this.saved = null;
  }

  // ---- wireframe ----

  /** Whether the wireframe overlay is on. */
  get isWireframe(): boolean {
    return this.wire;
  }

  /** Whether this model-viewer build lets us draw a wireframe (see threeSceneOf). */
  get supportsWireframe(): boolean {
    return threeSceneOf(this.el) !== null;
  }

  /**
   * Draw the triangles (true) or remove them (false), in the given style
   * (default: the current one, so callers that toggle it off and on keep it).
   * @returns false if unsupported by this model-viewer build (nothing changes).
   */
  setWireframe(on: boolean, style: WireStyle = this.wireStyle): boolean {
    if (!this.supportsWireframe) return false;
    this.wire = on;
    this.wireStyle = style;
    this.removeWireframe(); // also clears the other style when switching
    if (on) this.applyWireframe();
    return true;
  }

  /**
   * Add, under every model mesh, a second mesh that shares its geometry but
   * draws it as dark semi-transparent lines. The surface is pushed back with a
   * polygon offset so the lines don't z-fight with it. Being children of the
   * model's meshes, the overlays follow the model and leave with it on reload.
   */
  private applyWireframe(): void {
    const scene = threeSceneOf(this.el);
    if (!scene || this.overlays.length || this.wiredMaterials.length) return;
    const targets: ThreeObject[] = [];
    scene.traverse((o) => {
      const mat = o.material;
      // Model meshes only: skip our overlays and model-viewer's own helpers
      // (shadow / ground planes don't use MeshStandardMaterial).
      if (o.isMesh && !o.userData[WIRE_TAG] && mat && !Array.isArray(mat) && mat.isMeshStandardMaterial) {
        targets.push(o);
      }
    });
    if (this.wireStyle === "textured") {
      // Switch the model's own materials to lines: three.js draws each edge
      // with the material's texture, and with no filled faces nothing hides
      // the edges behind other surfaces.
      for (const mesh of targets) {
        const surface = mesh.material as ThreeMaterial;
        if (surface.wireframe) continue;
        surface.wireframe = true;
        this.wiredMaterials.push(surface);
      }
      scene.queueRender();
      return;
    }
    for (const mesh of targets) {
      const surface = mesh.material as ThreeMaterial;
      const MaterialCtor = surface.constructor as new () => ThreeMaterial;
      const MeshCtor = mesh.constructor as new (geometry: unknown, material: ThreeMaterial) => ThreeObject;
      const lines = new MaterialCtor(); // a fresh MeshStandardMaterial: no textures
      lines.wireframe = true;
      lines.color.setRGB(...WIRE_RGB);
      lines.metalness = 0;
      lines.roughness = 1;
      lines.transparent = true;
      lines.opacity = WIRE_OPACITY;
      lines.depthWrite = false;
      const overlay = new MeshCtor(mesh.geometry, lines);
      overlay.userData[WIRE_TAG] = true;
      mesh.add(overlay);
      this.overlays.push(overlay);
      if (!surface.polygonOffset) {
        surface.polygonOffset = true;
        surface.polygonOffsetFactor = 1;
        surface.polygonOffsetUnits = 1;
        this.offsetMaterials.push(surface);
      }
    }
    scene.queueRender();
  }

  private removeWireframe(): void {
    for (const overlay of this.overlays) {
      overlay.parent?.remove(overlay);
      (overlay.material as ThreeMaterial).dispose();
    }
    for (const m of this.offsetMaterials) m.polygonOffset = false;
    for (const m of this.wiredMaterials) m.wireframe = false;
    this.overlays = [];
    this.offsetMaterials = [];
    this.wiredMaterials = [];
    threeSceneOf(this.el)?.queueRender();
  }

  // ---- turnaround sheet ----

  /**
   * Render the four SHEET_VIEWS and composite them into one 2x2 PNG with a
   * label on each view.
   *
   * Each view is center-cropped to a square: model-viewer frames the model to
   * the shorter side, so the crop keeps all of it. Resolution follows the
   * viewer's on-screen size (for 4K sheets, use tools/mv_preview/render_quad.js).
   * The camera and the clay setting are restored afterwards.
   *
   * Note: model-viewer only repaints while visible (tools/mv_preview/README.md),
   * so call this with the viewer on screen.
   */
  async captureSheet(options: { clay: boolean }): Promise<Blob> {
    const camera = {
      orbit: this.el.cameraOrbit,
      target: this.el.cameraTarget,
      fov: this.el.fieldOfView,
    };
    const wasClay = this.clay;
    const wasWire = this.wire;
    try {
      if (options.clay !== wasClay) this.setClay(options.clay);
      if (wasWire) this.setWireframe(false); // sheets show the surface only
      const tiles: ImageBitmap[] = [];
      for (const view of SHEET_VIEWS) {
        this.el.cameraOrbit = `${view.theta}deg 75deg auto`;
        this.el.cameraTarget = "auto auto auto";
        this.el.fieldOfView = "auto";
        this.el.resetTurntableRotation(0);
        this.el.jumpCameraToGoal();
        await settle();
        const shot = await createImageBitmap(await this.el.toBlob({ mimeType: "image/png" }));
        const side = Math.min(shot.width, shot.height);
        tiles.push(
          await createImageBitmap(shot, (shot.width - side) / 2, (shot.height - side) / 2, side, side),
        );
        shot.close();
      }
      return await composeSheet(tiles, SHEET_VIEWS.map((v) => v.label));
    } finally {
      if (this.clay !== wasClay) this.setClay(wasClay);
      if (wasWire) this.setWireframe(true);
      this.el.cameraOrbit = camera.orbit;
      this.el.cameraTarget = camera.target;
      this.el.fieldOfView = camera.fov;
      this.el.jumpCameraToGoal();
    }
  }
}

/** Lay out square tiles 2x2 on a dark background, each with a label in its top-left corner. */
async function composeSheet(tiles: ImageBitmap[], labels: string[]): Promise<Blob> {
  const side = tiles[0].width;
  const gap = Math.round(side * 0.01);
  const canvas = document.createElement("canvas");
  canvas.width = side * 2 + gap;
  canvas.height = side * 2 + gap;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#0e1014";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const fontPx = Math.max(12, Math.round(side * 0.035));
  ctx.font = `600 ${fontPx}px system-ui, sans-serif`;
  tiles.forEach((tile, i) => {
    const x = (i % 2) * (side + gap);
    const y = Math.floor(i / 2) * (side + gap);
    ctx.drawImage(tile, x, y);
    tile.close();
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    const w = ctx.measureText(labels[i]).width + fontPx;
    ctx.fillRect(x, y, w, fontPx * 1.8);
    ctx.fillStyle = "#e6e8ee";
    ctx.fillText(labels[i], x + fontPx / 2, y + fontPx * 1.25);
  });
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("could not encode sheet"))), "image/png"),
  );
}
