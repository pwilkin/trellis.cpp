// Thin bridge over the Tauri v2 runtime so the same UI also runs in a plain
// browser (vite dev / served by trellis-server). Every Tauri call is guarded by
// `isTauri()`; in the browser we fall back to sensible web behaviour.

import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { listen as tauriListen, type UnlistenFn } from "@tauri-apps/api/event";

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** Invoke a Rust command; throws in the browser (callers must guard). */
export function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  return tauriInvoke<T>(cmd, args);
}

/** Subscribe to a Tauri event; no-op (returns a noop unlisten) in the browser. */
export async function listen<T>(
  event: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  if (!isTauri()) return () => {};
  return tauriListen<T>(event, (e) => handler(e.payload as T));
}

/**
 * Save bytes to disk. In Tauri, open a native "Save as" dialog then write via
 * the fs plugin. In the browser, trigger a normal download.
 */
export async function saveBytes(defaultName: string, bytes: Uint8Array): Promise<boolean> {
  if (isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const { writeFile } = await import("@tauri-apps/plugin-fs");
    const path = await save({
      defaultPath: defaultName,
      filters: [{ name: "glTF binary", extensions: ["glb"] }],
    });
    if (!path) return false;
    await writeFile(path, bytes);
    return true;
  }
  const blob = new Blob([bytes as BlobPart], { type: "model/gltf-binary" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = defaultName;
  a.click();
  URL.revokeObjectURL(url);
  return true;
}

/** Native folder picker (Tauri only); returns the chosen path or null. */
export async function pickDirectory(current?: string): Promise<string | null> {
  if (!isTauri()) return null;
  const { open } = await import("@tauri-apps/plugin-dialog");
  const res = await open({ directory: true, defaultPath: current || undefined });
  return typeof res === "string" ? res : null;
}

/** Open the configured output directory in the OS file browser (Tauri only). */
export async function openOutputDir(): Promise<void> {
  if (isTauri()) await invoke("open_output_dir");
}

/**
 * Auto-save GLB bytes into the configured output directory (Tauri only).
 * The Rust side resolves + creates the dir and returns the full path; the fs
 * plugin then writes the bytes there. Returns the path, or null in the browser.
 */
export async function saveToOutputDir(name: string, bytes: Uint8Array): Promise<string | null> {
  if (!isTauri()) return null;
  const { writeFile } = await import("@tauri-apps/plugin-fs");
  const path = await invoke<string>("output_path", { name });
  await writeFile(path, bytes);
  return path;
}
