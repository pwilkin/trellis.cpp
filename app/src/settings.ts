// Settings modal: shows the resolved config and lets the user adjust the bits
// that make sense per environment. In Tauri, saving hands the config to the shell
// (which restarts the server); in the browser we only expose host/port.

import { loadConfig, saveConfig } from "./config";
import { invoke, isTauri } from "./tauri";

function field(label: string, id: string, value: string, type = "text"): string {
  return `<label class="ctl"><span>${label}</span>
    <input id="${id}" type="${type}" value="${value.replace(/"/g, "&quot;")}" /></label>`;
}

function ro(label: string, value: string): string {
  return `<div class="kv">${label}: <b>${value || "—"}</b></div>`;
}

export async function renderSettings(
  body: HTMLElement,
  onSaved: () => void,
): Promise<void> {
  const cfg = await loadConfig(true);

  if (isTauri()) {
    body.innerHTML = `
      ${ro("Backend", cfg.backend)}
      ${ro("Server binary", cfg.serverBin)}
      ${field("Models directory", "set-models", cfg.modelsDir)}
      ${field("GPU index (&lt;0 = CPU)", "set-gpu", String(cfg.gpu), "number")}
      ${field("Port", "set-port", String(cfg.port), "number")}
      <div class="modal-actions">
        <button id="set-restart" class="tool-btn">Restart server</button>
        <button id="set-save" class="primary">Save &amp; restart</button>
      </div>`;

    const save = async () => {
      const modelsDir = (body.querySelector("#set-models") as HTMLInputElement).value.trim();
      const gpu = parseInt((body.querySelector("#set-gpu") as HTMLInputElement).value, 10);
      const port = parseInt((body.querySelector("#set-port") as HTMLInputElement).value, 10);
      await saveConfig({ modelsDir, gpu: isNaN(gpu) ? 0 : gpu, port: isNaN(port) ? 8080 : port });
      try {
        await invoke("restart_server");
      } catch {
        /* shell will surface its own error; status polling reflects it */
      }
      onSaved();
    };
    (body.querySelector("#set-save") as HTMLButtonElement).onclick = save;
    (body.querySelector("#set-restart") as HTMLButtonElement).onclick = async () => {
      try {
        await invoke("restart_server");
      } catch {
        /* ignore */
      }
      onSaved();
    };
  } else {
    body.innerHTML = `
      <div class="kv">Running in a browser — connecting to a trellis-server you launched.</div>
      ${field("Host", "set-host", cfg.host)}
      ${field("Port", "set-port", String(cfg.port), "number")}
      <div class="modal-actions">
        <button id="set-save" class="primary">Save</button>
      </div>`;
    (body.querySelector("#set-save") as HTMLButtonElement).onclick = async () => {
      const host = (body.querySelector("#set-host") as HTMLInputElement).value.trim() || "127.0.0.1";
      const port = parseInt((body.querySelector("#set-port") as HTMLInputElement).value, 10);
      await saveConfig({ host, port: isNaN(port) ? 8080 : port });
      onSaved();
    };
  }
}
