// Drag to reorder: a small pointer-events sortable list (no library).
//
// Used by the "Up next" queue cards (queue.ts). Press on an item's handle and
// move more than a few pixels to pick it up; it follows the pointer while a
// placeholder of the same height shows where it will land. Release to drop,
// Escape to cancel. A press without movement stays a plain click, so the
// handle can still toggle things (the queue card header opens/closes).
//
// Why not HTML5 drag-and-drop: the window already handles native drags for
// image files (main.ts, and Tauri's dragDropEnabled:false), and HTML5 DnD
// would mix the two and show a browser drag ghost. Pointer events keep the
// reorder separate from file drops and look like the card itself moving.
//
// While dragging, the item is moved to <body> with position:fixed. That keeps
// it above everything and out of the scrolling panel's clipping; moving the
// node (rather than cloning it) keeps any open editor's state.

export interface SortableOptions {
  /** Element whose children are the sortable items. */
  list: HTMLElement;
  /** Matches an item (a direct child of `list`). */
  itemSelector: string;
  /** Matches the part of an item you press to drag it. */
  handleSelector: string;
  /** Called when an item is picked up, before it's measured (e.g. to collapse it). */
  onDragStart?(item: HTMLElement): void;
  /** Called after a drop that changed the order; indexes are among the items. */
  onDrop(from: number, to: number): void;
}

export interface Sortable {
  /** True while an item is being dragged. */
  readonly dragging: boolean;
  /** Put a dragged item back where it came from, without calling onDrop. */
  cancel(): void;
}

/** Pixels the pointer must move before a press becomes a drag. */
const DRAG_THRESHOLD = 5;
/** Distance from the scroll container's edge that starts auto-scroll. */
const SCROLL_EDGE = 40;

/** Nearest ancestor that scrolls vertically, or null. */
function scrollParent(node: HTMLElement): HTMLElement | null {
  for (let p = node.parentElement; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if (oy === "auto" || oy === "scroll") return p;
  }
  return null;
}

export function makeSortable(opts: SortableOptions): Sortable {
  const { list } = opts;
  let pressed: { item: HTMLElement; pointerId: number; x: number; y: number } | null = null;
  let drag: { item: HTMLElement; from: number; grabY: number; placeholder: HTMLElement } | null = null;

  const items = () => [...list.children].filter((c): c is HTMLElement => c instanceof HTMLElement && c.matches(opts.itemSelector));

  list.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || drag) return;
    const handle = (e.target as HTMLElement).closest(opts.handleSelector);
    const item = handle?.closest(opts.itemSelector) as HTMLElement | null;
    if (!item || item.parentElement !== list) return;
    pressed = { item, pointerId: e.pointerId, x: e.clientX, y: e.clientY };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });

  function begin(e: PointerEvent): void {
    const { item } = pressed!;
    const from = items().indexOf(item);
    opts.onDragStart?.(item);
    const rect = item.getBoundingClientRect();
    const placeholder = document.createElement("div");
    placeholder.className = "sort-placeholder";
    placeholder.style.height = `${rect.height}px`;
    item.before(placeholder);

    item.classList.add("sort-dragging");
    Object.assign(item.style, { position: "fixed", left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px` });
    document.body.append(item);
    document.body.classList.add("sorting");
    drag = { item, from, grabY: e.clientY - rect.top, placeholder };
    window.addEventListener("keydown", onKey);
  }

  function onMove(e: PointerEvent): void {
    if (!pressed || e.pointerId !== pressed.pointerId) return;
    if (!drag) {
      if (Math.hypot(e.clientX - pressed.x, e.clientY - pressed.y) < DRAG_THRESHOLD) return;
      begin(e);
    }
    const d = drag!;
    d.item.style.top = `${e.clientY - d.grabY}px`;

    // Land before the first item whose middle is below the pointer.
    const target = items().find((c) => {
      const r = c.getBoundingClientRect();
      return e.clientY < r.top + r.height / 2;
    });
    if (target) {
      if (target.previousElementSibling !== d.placeholder) target.before(d.placeholder);
    } else if (list.lastElementChild !== d.placeholder) list.append(d.placeholder);

    // Near the top or bottom of the scrolling panel, scroll it.
    const scroller = scrollParent(list);
    if (scroller) {
      const s = scroller.getBoundingClientRect();
      if (e.clientY < s.top + SCROLL_EDGE) scroller.scrollBy(0, -12);
      else if (e.clientY > s.bottom - SCROLL_EDGE) scroller.scrollBy(0, 12);
    }
  }

  function onUp(e: PointerEvent): void {
    if (pressed && e.pointerId !== pressed.pointerId) return;
    if (drag) {
      const placed = [...list.children].filter((c) => c === drag!.placeholder || c.matches(opts.itemSelector));
      const to = placed.indexOf(drag.placeholder);
      const from = drag.from;
      finish();
      // The pointerup is followed by a click on whatever is under the pointer;
      // it belongs to the drag, not to a header toggle.
      const swallow = (ev: Event) => {
        ev.stopPropagation();
        ev.preventDefault();
      };
      window.addEventListener("click", swallow, { capture: true, once: true });
      setTimeout(() => window.removeEventListener("click", swallow, { capture: true }), 0);
      if (to !== from && to >= 0) opts.onDrop(from, to);
    }
    release();
  }

  function onKey(e: KeyboardEvent): void {
    if (e.key === "Escape") cancel();
  }

  /** Put the dragged item where the placeholder is and clear drag styles. */
  function finish(): void {
    if (!drag) return;
    const { item, placeholder } = drag;
    placeholder.replaceWith(item);
    item.classList.remove("sort-dragging");
    Object.assign(item.style, { position: "", left: "", top: "", width: "" });
    document.body.classList.remove("sorting");
    window.removeEventListener("keydown", onKey);
    drag = null;
  }

  function release(): void {
    pressed = null;
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
  }

  function cancel(): void {
    if (drag) {
      // Back to its original slot: move the placeholder there first.
      const others = items();
      const at = others[drag.from];
      if (at) at.before(drag.placeholder);
      else list.append(drag.placeholder);
      finish();
    }
    release();
  }

  return {
    get dragging() {
      return drag !== null;
    },
    cancel,
  };
}
