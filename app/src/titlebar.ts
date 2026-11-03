// Studio draws its own minimize / maximize / close buttons in the header.
// The OS title bar is off ("decorations": false in src-tauri/tauri.conf.json), so
// the header is also the drag handle: drag any empty part to move the window,
// double-click it to maximize. In a plain browser (vite dev, or the page served by
// trellis-server) this does nothing.

import { getCurrentWindow } from "@tauri-apps/api/window";
import { isTauri } from "./tauri";

const SVG = (body: string) =>
  `<svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1" aria-hidden="true">${body}</svg>`;
const ICON_MIN = SVG('<path d="M0 5.5h10"/>');
const ICON_MAX = SVG('<rect x="0.5" y="0.5" width="9" height="9"/>');
const ICON_RESTORE = SVG('<rect x="0.5" y="2.5" width="7" height="7"/><path d="M2.5 2.5v-2h7v7h-2"/>');
const ICON_CLOSE = SVG('<path d="M0.5 0.5l9 9M9.5 0.5l-9 9"/>');

export function initTitlebar(): void {
  if (!isTauri()) return;
  const bar = document.querySelector<HTMLElement>(".topbar");
  const right = bar?.querySelector<HTMLElement>(".topbar-right");
  if (!bar || !right) return;

  const win = getCurrentWindow();
  document.documentElement.classList.add("custom-titlebar");

  const button = (cls: string, label: string, icon: string, onClick: () => void) => {
    const b = document.createElement("button");
    b.className = `win-btn ${cls}`;
    b.title = label;
    b.setAttribute("aria-label", label);
    b.innerHTML = icon;
    b.addEventListener("click", onClick);
    return b;
  };
  const maxBtn = button("win-max", "Maximize", ICON_MAX, () => void win.toggleMaximize());
  const controls = document.createElement("div");
  controls.className = "window-controls";
  controls.append(
    button("win-min", "Minimize", ICON_MIN, () => void win.minimize()),
    maxBtn,
    button("win-close", "Close", ICON_CLOSE, () => void win.close()),
  );
  right.append(controls);

  // Swap the maximize icon when the window is maximized or restored, including
  // by the OS (Win+Up, snapping, double-click).
  const syncMax = async () => {
    const max = await win.isMaximized();
    maxBtn.innerHTML = max ? ICON_RESTORE : ICON_MAX;
    maxBtn.title = max ? "Restore" : "Maximize";
    maxBtn.setAttribute("aria-label", maxBtn.title);
  };
  let debounce: ReturnType<typeof setTimeout> | undefined;
  void win.onResized(() => {
    clearTimeout(debounce);
    debounce = setTimeout(() => void syncMax(), 100);
  });
  void syncMax();

  // Drag from the header, except from its controls. Same pattern as Tauri's own
  // data-tauri-drag-region: a double-click (detail 2) toggles maximize.
  bar.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    if ((e.target as Element).closest("button, a, input, select, textarea")) return;
    if (e.detail === 2) void win.toggleMaximize();
    else void win.startDragging();
  });
}
