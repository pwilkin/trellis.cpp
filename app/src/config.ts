// Resolves the app config (server host/port, models dir, backend). In Tauri the
// shell owns config.json and exposes it via the `get_config` command; in the
// browser we fall back to defaults, overridable via localStorage.

import { invoke, isTauri } from "./tauri";
import type { AppConfig } from "./types";

const LS_KEY = "trellis.config";

const DEFAULTS: AppConfig = {
  serverBin: "",
  modelsDir: "",
  backend: "unknown",
  gpu: 0,
  host: "127.0.0.1",
  port: 8080,
  outputDir: "",
  configured: false,
};

let cached: AppConfig | null = null;

export async function loadConfig(force = false): Promise<AppConfig> {
  if (cached && !force) return cached;
  if (isTauri()) {
    try {
      const c = await invoke<AppConfig | null>("get_config");
      cached = c ? { ...DEFAULTS, ...c, configured: true } : { ...DEFAULTS };
    } catch {
      cached = { ...DEFAULTS };
    }
  } else {
    const raw = localStorage.getItem(LS_KEY);
    cached = raw
      ? { ...DEFAULTS, ...(JSON.parse(raw) as Partial<AppConfig>), configured: true }
      : { ...DEFAULTS, configured: true }; // browser: assume a server is up somewhere
  }
  return cached;
}

/** Persist config. In Tauri, hand it to the shell (which may restart the server). */
export async function saveConfig(patch: Partial<AppConfig>): Promise<AppConfig> {
  const next: AppConfig = { ...(cached ?? DEFAULTS), ...patch };
  if (isTauri()) {
    await invoke("save_config", { config: next });
  } else {
    localStorage.setItem(LS_KEY, JSON.stringify(next));
  }
  cached = next;
  return next;
}

export function apiBase(cfg: AppConfig): string {
  return `http://${cfg.host}:${cfg.port}`;
}
