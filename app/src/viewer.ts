// Wrapper around Google's <model-viewer> web component (vendored, three.js-based).
// It renders the WebP-textured PBR GLBs that trellis produces, and `camera-controls`
// gives the rotate + zoom the previewer needs.

// Minimal structural type for the bits of <model-viewer> we touch.
interface ModelViewerEl extends HTMLElement {
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

  /** A PNG snapshot of the current view for the gallery thumbnail. */
  async thumbnail(): Promise<Blob | null> {
    try {
      // model-viewer's 'load' fires before its auto-framing camera animation has
      // settled, so an immediate toBlob can capture a mid-frame view; give the
      // framing a beat to finish before snapshotting.
      await new Promise((r) => setTimeout(r, 400));
      return await this.el.toBlob({ mimeType: "image/png", idealAspect: true });
    } catch {
      return null;
    }
  }
}
