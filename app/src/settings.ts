// Settings modal: shows the resolved config and lets the user adjust the bits
// that make sense per environment. In Tauri, saving hands the config to the shell
// (which restarts the server); in the browser we only expose host/port.
//
// The "Queue" section is different: those are UI preferences (prefs.ts), not
// server config, so they save the moment they change and need no restart.

import { loadConfig, saveConfig } from "./config";
import { getPrefs, setPref } from "./prefs";
import { invoke, isTauri, logsDir, openLogsDir, openOutputDir, pickDirectory } from "./tauri";

function field(label: string, id: string, value: string, type = "text"): string {
  return `<label class="ctl"><span>${label}</span>
    <input id="${id}" type="${type}" value="${value.replace(/"/g, "&quot;")}" /></label>`;
}

/** Text field with a "Browse…" folder picker and an "Open" button. */
function dirField(label: string, id: string, value: string): string {
  return `<label class="ctl"><span>${label}</span>
    <div class="dir-row">
      <input id="${id}" type="text" value="${value.replace(/"/g, "&quot;")}" />
      <button id="${id}-browse" class="tool-btn" type="button">Browse…</button>
      <button id="${id}-open" class="tool-btn" type="button">Open</button>
    </div></label>`;
}

function ro(label: string, value: string): string {
  return `<div class="kv">${label}: <b>${value || "—"}</b></div>`;
}

/** Checkbox row with a one-line hint underneath. */
function check(label: string, id: string, checked: boolean, hint: string): string {
  return `<label class="check">
    <input id="${id}" type="checkbox" ${checked ? "checked" : ""} />
    <span>${label}<small class="muted">${hint}</small></span></label>`;
}

/** The Queue section: UI preferences, shown in both Tauri and browser mode. */
function queuePrefsSection(): string {
  return `<div class="section-head">Queue</div>
    ${check(
      "Queued images use the same settings as the generation in progress",
      "pref-queue-same",
      getPrefs().queueSameSettings,
      "Off: each image dropped mid-run uses the settings panel as it is when you drop it.",
    )}`;
}

/** Wire the Queue section's controls. Changes are saved immediately. */
function bindQueuePrefs(body: HTMLElement): void {
  const box = body.querySelector("#pref-queue-same") as HTMLInputElement;
  box.addEventListener("change", () => setPref("queueSameSettings", box.checked));
}

export async function renderSettings(
  body: HTMLElement,
  onSaved: () => void,
): Promise<void> {
  const cfg = await loadConfig(true);

  if (isTauri()) {
    // get_config already fills the default output dir; fall back to it when unset
    // (e.g. no config.json yet).
    let outputDir = cfg.outputDir;
    if (!outputDir) {
      try {
        outputDir = await invoke<string>("default_output_dir");
      } catch {
        /* leave blank */
      }
    }
    const logDir = await logsDir();

    body.innerHTML = `
      ${ro("Backend", cfg.backend)}
      ${field("Server binary (trellis-server)", "set-bin", cfg.serverBin)}
      ${field("Models directory", "set-models", cfg.modelsDir)}
      ${dirField("Output folder (generated GLBs are saved here)", "set-output", outputDir)}
      ${field("GPU index (&lt;0 = CPU)", "set-gpu", String(cfg.gpu), "number")}
      ${field("Port", "set-port", String(cfg.port), "number")}
      <label class="ctl"><span>Server logs (each launch is saved here — attach these to bug reports)</span>
        <div class="dir-row">
          <input id="set-logs" type="text" value="${logDir.replace(/"/g, "&quot;")}" readonly />
          <button id="set-logs-open" class="tool-btn" type="button">Open</button>
        </div></label>
      ${queuePrefsSection()}
      <div class="modal-actions">
        <button id="set-restart" class="tool-btn">Restart server</button>
        <button id="set-save" class="primary">Save &amp; restart</button>
      </div>`;

    bindQueuePrefs(body);

    (body.querySelector("#set-logs-open") as HTMLButtonElement).onclick = async () => {
      try {
        await openLogsDir();
      } catch (e) {
        alert(`Could not open the logs folder: ${(e as Error).message ?? e}`);
      }
    };

    const outputInput = body.querySelector("#set-output") as HTMLInputElement;
    (body.querySelector("#set-output-browse") as HTMLButtonElement).onclick = async () => {
      const picked = await pickDirectory(outputInput.value.trim());
      if (picked) outputInput.value = picked;
    };
    (body.querySelector("#set-output-open") as HTMLButtonElement).onclick = async () => {
      // persist the (possibly-edited) path first so the shell opens what's shown
      await saveConfig({ outputDir: outputInput.value.trim() });
      try {
        await openOutputDir();
      } catch (e) {
        alert(`Could not open the output folder: ${(e as Error).message ?? e}`);
      }
    };

    // Report a failed (re)start instead of swallowing it: with a manual setup the
    // shell has no server binary yet, so restart_server errors and the app just
    // sits at "offline" looking like it ignored the new models directory (#21).
    const restart = async () => {
      try {
        await invoke("restart_server");
      } catch (e) {
        alert(`Could not start the server: ${(e as Error).message ?? e}`);
      }
    };

    const save = async () => {
      const serverBin = (body.querySelector("#set-bin") as HTMLInputElement).value.trim();
      const modelsDir = (body.querySelector("#set-models") as HTMLInputElement).value.trim();
      const gpu = parseInt((body.querySelector("#set-gpu") as HTMLInputElement).value, 10);
      const port = parseInt((body.querySelector("#set-port") as HTMLInputElement).value, 10);
      await saveConfig({
        serverBin,
        modelsDir,
        gpu: isNaN(gpu) ? 0 : gpu,
        port: isNaN(port) ? 8080 : port,
        outputDir: outputInput.value.trim(),
      });
      await restart();
      onSaved();
    };
    (body.querySelector("#set-save") as HTMLButtonElement).onclick = save;
    (body.querySelector("#set-restart") as HTMLButtonElement).onclick = async () => {
      await restart();
      onSaved();
    };
  } else {
    body.innerHTML = `
      <div class="kv">Running in a browser — connecting to a trellis-server you launched.</div>
      ${field("Host", "set-host", cfg.host)}
      ${field("Port", "set-port", String(cfg.port), "number")}
      ${queuePrefsSection()}
      <div class="modal-actions">
        <button id="set-save" class="primary">Save</button>
      </div>`;
    bindQueuePrefs(body);
    (body.querySelector("#set-save") as HTMLButtonElement).onclick = async () => {
      const host = (body.querySelector("#set-host") as HTMLInputElement).value.trim() || "127.0.0.1";
      const port = parseInt((body.querySelector("#set-port") as HTMLInputElement).value, 10);
      await saveConfig({ host, port: isNaN(port) ? 8080 : port });
      onSaved();
    };
  }
}
