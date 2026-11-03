// Generation queue: images waiting to run after the current job.
//
// trellis-server runs one generation at a time (a mutex around trellis_run in
// trellis-server.cpp), so the queue lives entirely in the UI: main.ts starts
// the next item when a run finishes, and the server only ever sees one
// request in flight. The queue is in-memory and does not survive a restart.
//
// Each item gets its settings (GenParams) when it is added; which settings is
// decided by the caller (see `queueSameSettings` in prefs.ts). After that the
// item's own card can change them until the job starts: the "Up next" list
// shows one collapsible card per item (design handoff, Phase 2):
//
//   collapsed   #  [thumb]  name                                 ˅
//                           1024  bg auto  xatlas  Game 50K/2048  seed 42
//   expanded    the same header, then the settings editor (controls.ts)
//               and Remove
//
// Drag a card's header to change the order (sortable.ts); a press without
// movement opens/closes it. Alt+Up / Alt+Down on a focused header does the
// same from the keyboard.
//
// Only one card is open at a time, and a newly added image opens. Cards are
// built once per item and reused across renders, so an open editor keeps its
// state (and focus) while the list around it changes.
//
// This module owns the items and their cards. Starting jobs, and deciding
// when to, stays in main.ts.

import { createSettingsEditor, type SettingsEditor } from "./controls";
import { paramChips } from "./options";
import { makeSortable, type Sortable } from "./sortable";
import { newId } from "./store";
import type { GenParams } from "./types";

/** One image waiting to be generated. */
export interface QueueItem {
  id: string;
  image: Blob;
  /** Original file name, used for the output file and the gallery entry. */
  name: string;
  /** Settings this item will run with; editable on its card until it starts. */
  params: GenParams;
  /** Object URL for the thumbnail; revoked when the item leaves the queue. */
  thumbUrl: string;
}

/** DOM elements the queue renders into (see #queue in index.html). */
export interface QueueElements {
  /** Container; hidden while the queue is empty. */
  wrap: HTMLElement;
  /** Heading text, e.g. "Queued · 3" (kept for screen readers / tooltips). */
  label: HTMLElement;
  /** Vertical list of job cards. */
  list: HTMLElement;
  /** "Clear" button. */
  clearBtn: HTMLElement;
}

/** The DOM of one card, kept for the item's lifetime. */
interface Card {
  root: HTMLElement;
  head: HTMLButtonElement;
  pos: HTMLElement;
  chips: HTMLElement;
  /** Built on first open (most queued items are never edited). */
  editor: SettingsEditor | null;
  body: HTMLElement;
}

/** Six-dot grip, shown in place of the position number on hover (the header drags). */
const GRIP_SVG =
  '<svg width="10" height="14" viewBox="0 0 10 14" fill="currentColor" aria-hidden="true">' +
  '<circle cx="2" cy="2" r="1.3"></circle><circle cx="8" cy="2" r="1.3"></circle>' +
  '<circle cx="2" cy="7" r="1.3"></circle><circle cx="8" cy="7" r="1.3"></circle>' +
  '<circle cx="2" cy="12" r="1.3"></circle><circle cx="8" cy="12" r="1.3"></circle></svg>';

const CHEVRON_SVG =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" ' +
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"></path></svg>';

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function fillChips(container: HTMLElement, p: GenParams): void {
  container.replaceChildren(...paramChips(p).map((c) => el("span", "chip", c)));
}

/** FIFO queue of pending generations, with its own editable cards. */
export class GenerationQueue {
  private items: QueueItem[] = [];
  private readonly cards = new Map<string, Card>();
  /** The expanded card, if any. */
  private openId: string | null = null;
  private readonly sortable: Sortable;

  /**
   * @param els      Where to render the queue.
   * @param onChange Called after every add/remove/move/clear/take, so the
   *                 caller can refresh things that depend on the queue (e.g.
   *                 the count pill, whether Generate is enabled).
   */
  constructor(
    private readonly els: QueueElements,
    private readonly onChange: () => void = () => {},
  ) {
    els.clearBtn.addEventListener("click", () => this.clear());
    this.sortable = makeSortable({
      list: els.list,
      itemSelector: ".qcard",
      handleSelector: ".qcard-head",
      // A dragged editor would be a tall, awkward thing to carry: close it first.
      onDragStart: (cardEl) => {
        const id = cardEl.dataset.id;
        if (id && id === this.openId) {
          this.openId = null;
          this.render();
        }
      },
      onDrop: (from, to) => this.move(from, to),
    });
    this.render();
  }

  /** Number of images waiting. */
  get size(): number {
    return this.items.length;
  }

  /** Append an image to the end of the queue with the given settings, and open its card. */
  add(image: Blob, name: string, params: GenParams): void {
    const item: QueueItem = { id: newId(), image, name, params: { ...params }, thumbUrl: URL.createObjectURL(image) };
    this.items.push(item);
    this.openId = item.id;
    this.changed();
  }

  /**
   * Remove and return the next item, or undefined if the queue is empty.
   * It runs with the settings its card shows right now. Its thumbnail URL is
   * revoked; the image Blob itself stays valid.
   */
  take(): QueueItem | undefined {
    const item = this.items.shift();
    if (!item) return undefined;
    this.drop(item);
    this.changed();
    return item;
  }

  /** Remove one item by id (Remove on its card). */
  remove(id: string): void {
    const i = this.items.findIndex((q) => q.id === id);
    if (i < 0) return;
    this.drop(this.items[i]);
    this.items.splice(i, 1);
    this.changed();
  }

  /** Move the item at index `from` to index `to` (a drag, or Alt+Up/Down). */
  move(from: number, to: number): void {
    if (from === to || from < 0 || to < 0 || from >= this.items.length || to >= this.items.length) return;
    const [item] = this.items.splice(from, 1);
    this.items.splice(to, 0, item);
    this.changed();
  }

  /** Drop everything that's waiting. Does not affect a job already running. */
  clear(): void {
    this.items.forEach((q) => this.drop(q));
    this.items = [];
    this.changed();
  }

  /** Open one card (closing any other), or close it if it's already open. */
  toggle(id: string): void {
    this.openId = this.openId === id ? null : id;
    this.render();
    // An opened editor is taller than the panel's visible area; bring it into view.
    if (this.openId) this.cards.get(id)?.root.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  /** Forget an item's card and thumbnail URL (it's leaving the queue). */
  private drop(item: QueueItem): void {
    URL.revokeObjectURL(item.thumbUrl);
    this.cards.get(item.id)?.root.remove();
    this.cards.delete(item.id);
    if (this.openId === item.id) this.openId = null;
  }

  private changed(): void {
    this.render();
    this.onChange();
  }

  /** Build the card for one item (once; render() reuses it). */
  private buildCard(item: QueueItem): Card {
    const root = el("div", "qcard");
    root.dataset.id = item.id;

    const head = el("button", "qcard-head");
    head.type = "button";
    const pos = el("span", "qcard-pos mono");
    const grip = el("span", "qcard-grip");
    grip.innerHTML = GRIP_SVG;
    const thumb = el("img", "qcard-thumb");
    thumb.src = item.thumbUrl;
    thumb.alt = "";
    const info = el("span", "qcard-info");
    const chips = el("span", "chips");
    info.append(el("span", "qcard-name", item.name), chips);
    const chevron = el("span", "qcard-chevron");
    chevron.innerHTML = CHEVRON_SVG;
    head.append(pos, grip, thumb, info, chevron);
    head.addEventListener("click", () => this.toggle(item.id));
    head.addEventListener("keydown", (e) => {
      if (!e.altKey || (e.key !== "ArrowUp" && e.key !== "ArrowDown")) return;
      e.preventDefault();
      const i = this.items.indexOf(item);
      this.move(i, e.key === "ArrowUp" ? i - 1 : i + 1);
      head.focus(); // the card moved in the DOM; keep the keyboard on it
    });

    const body = el("div", "qcard-body");
    const actions = el("div", "qcard-actions");
    const remove = el("button", "ghost-btn small danger", "Remove");
    remove.type = "button";
    remove.addEventListener("click", () => this.remove(item.id));
    actions.append(remove);
    body.append(actions);

    root.append(head, body);
    fillChips(chips, item.params);
    const card: Card = { root, head, pos, chips, editor: null, body };
    this.cards.set(item.id, card);
    return card;
  }

  /** Create the card's settings editor the first time it opens. */
  private ensureEditor(item: QueueItem, card: Card): void {
    if (card.editor) return;
    card.editor = createSettingsEditor(item.params, (p) => {
      item.params = p;
      fillChips(card.chips, p);
    });
    card.body.prepend(card.editor.el);
  }

  /** Put the cards in queue order and sync numbers and open state. */
  private render(): void {
    const { wrap, label, list } = this.els;
    // The list is about to be rebuilt (e.g. the next job started mid-drag):
    // drop the drag rather than leave a card floating outside the list.
    if (this.sortable?.dragging) this.sortable.cancel();
    wrap.classList.toggle("hidden", this.items.length === 0);
    label.textContent = `Queued · ${this.items.length}`;

    this.items.forEach((item, i) => {
      const card = this.cards.get(item.id) ?? this.buildCard(item);
      const open = item.id === this.openId;
      if (open) this.ensureEditor(item, card);
      card.root.classList.toggle("open", open);
      card.head.setAttribute("aria-expanded", String(open));
      card.head.title = open ? "Collapse · drag to reorder" : "Edit settings · drag to reorder";
      card.pos.textContent = String(i + 1);
      // appendChild moves an existing node, so this also applies a new order.
      list.appendChild(card.root);
    });
  }
}

/** Human-readable summary of every setting: "1024 · bg auto · xatlas · Game 50K/2048 · seed 42". */
export function describeParams(p: GenParams): string {
  return paramChips(p).join(" · ");
}
