// UI preferences: how the app behaves, as opposed to how the server runs.
//
// These deliberately do NOT live in config.json. That file is the contract
// between the installer and the Rust shell (see src-tauri/src/config.rs), and
// its serde struct would silently drop any field it doesn't know on the next
// save. Preferences only matter to this web UI, so they are kept in
// localStorage, which persists across restarts in both the Tauri webview and a
// plain browser.
//
// Storage can be unavailable (private mode, a locked-down webview), so every
// access is wrapped: a failed read yields the defaults and a failed write
// keeps the value in memory for this session only.

/** All user-adjustable UI preferences. Add new ones here with a default below. */
export interface Prefs {
  /**
   * When an image is queued while a generation is running, give it the same
   * settings as that running job (true), or the settings shown in the panel
   * at the moment it was added (false).
   *
   * On keeps a dropped batch consistent; off lets you change resolution or
   * seed between drops to vary each queued image.
   */
  queueSameSettings: boolean;
}

const LS_KEY = "trellis.prefs";

const DEFAULTS: Prefs = {
  queueSameSettings: true,
};

/** In-memory copy; the source of truth once loaded. */
let current: Prefs | null = null;

/** Current preferences (defaults merged with anything stored). */
export function getPrefs(): Prefs {
  if (current) return current;
  let stored: Partial<Prefs> = {};
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) stored = JSON.parse(raw) as Partial<Prefs>;
  } catch {
    /* unreadable or corrupt: fall back to defaults */
  }
  current = { ...DEFAULTS, ...stored };
  return current;
}

/** Update one preference and persist it (best-effort). */
export function setPref<K extends keyof Prefs>(key: K, value: Prefs[K]): void {
  current = { ...getPrefs(), [key]: value };
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(current));
  } catch {
    /* storage unavailable: keep the in-memory value for this session */
  }
}
