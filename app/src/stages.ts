// Generation progress, read from trellis-server's own log.
//
// The desktop shell forwards every server stdout/stderr line as a
// `server-log` event (src-tauri/src/server.rs). trellis prints a numbered
// header when each pipeline step starts (trellis_cli.cpp):
//
//   [1/6] preprocess …               [4/7] shape SLAT flow …
//   [2/6] DINOv3 conditioning        [5/7] FlexiDualGrid shape decode …
//   [3/6] sparse-structure flow …    [6/7] texture SLAT flow + PBR decode
//                                    [7/7] write <out>.glb
//
// (The /6 vs /7 totals are inconsistent in trellis itself; only the step
// number is used.) Step 7 is the long CPU post-process: remesh, QEM
// simplification, then UV unwrap and texture bake, which logs `uv_bake:`
// (or `uv_box_project:` / `uv_chart_project:`). That split is shown as its own
// stage, so the card names 8 stages.
//
// The flow steps also print one line per sampling step when stdout isn't a
// terminal (flow_runner.cpp):  [flow] [#####...........]  5/12  23.1s ~30s left
//
// Percent: each stage has an expected duration. A stage's share of the bar
// fills from its flow step counter when it has one, otherwise from time spent
// in it. Expected durations start from rough RTX 5090 figures and are replaced
// by what this machine actually took (stored per resolution after every
// successful run), so the bar gets more accurate with use. It never moves
// backwards and stops at 99% until the model arrives.
//
// In a plain browser there is no server log: the tracker never sees a stage
// and the card keeps its indeterminate bar.

/** One stage of the pipeline as shown on the running card. */
export interface StageDef {
  name: string;
  /** Matches the (trimmed) log line that starts this stage. */
  start: RegExp;
}

export const STAGES: readonly StageDef[] = [
  { name: "Preprocess", start: /^\[1\/\d\]/ },
  { name: "Image conditioning", start: /^\[2\/\d\]/ },
  { name: "Sparse structure", start: /^\[3\/\d\]/ },
  { name: "Shape latent", start: /^\[4\/\d\]/ },
  { name: "Decode mesh", start: /^\[5\/\d\]/ },
  { name: "Texture latent", start: /^\[6\/\d\]/ },
  { name: "Simplify mesh", start: /^\[7\/\d\]/ },
  { name: "UV unwrap & bake", start: /^uv_(bake|box_project|chart_project):/ },
];

/** `[flow] [####....]  5/12 …` -> done 5 of 12. */
const FLOW_STEP = /^\[flow\] \[[#.]*\]\s*(\d+)\/(\d+)/;

/**
 * Starting guesses in seconds per stage, until this machine has a measured run
 * at that resolution. Rough RTX 5090 figures; 1024 and 1536 run a two-pass
 * cascade in "Shape latent".
 */
const DEFAULT_SECONDS: Record<number, number[]> = {
  512: [3, 3, 7, 8, 6, 12, 20, 15],
  1024: [3, 3, 7, 35, 10, 23, 40, 25],
  1536: [3, 3, 7, 90, 20, 30, 60, 30],
};

const LS_KEY = "trellis.stageSeconds.v1";

/** Measured seconds per stage, by resolution (best-effort localStorage). */
function loadMeasured(): Record<string, number[]> {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? (JSON.parse(raw) as Record<string, number[]>) : {};
  } catch {
    return {};
  }
}

function saveMeasured(all: Record<string, number[]>): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(all));
  } catch {
    /* storage unavailable: estimates just won't improve this session */
  }
}

/** What the running card shows. */
export interface StageProgress {
  /** 0-based stage index into STAGES. */
  index: number;
  total: number;
  name: string;
  /** 0–99 while running. */
  percent: number;
}

/** Follows one generation's log lines. Create a new one per job. */
export class StageTracker {
  private index = -1;
  /** When each stage started (ms), by stage index; missing for skipped stages. */
  private readonly startedAt: number[] = [];
  private readonly expected: number[];
  /** Flow passes seen in the current stage, and the latest step fraction. */
  private flowsSeen = 0;
  private flowFraction: number | null = null;
  /** "Shape latent" at 1024+ runs two flows (LR then HR); only the HR one tracks the stage. */
  private cascade = false;
  private shown = 0;

  constructor(
    private readonly resolution: number,
    private readonly now: () => number = () => Date.now(),
  ) {
    const measured = loadMeasured()[String(resolution)];
    const fallback = DEFAULT_SECONDS[resolution] ?? DEFAULT_SECONDS[1024];
    this.expected = STAGES.map((_, i) => measured?.[i] ?? fallback[i]);
  }

  /**
   * Feed one log line.
   * @returns true if the stage or step changed (worth re-rendering).
   */
  feed(rawLine: string): boolean {
    const line = rawLine.trim();
    // Stages only move forward: a later header can't be undone by an earlier one.
    for (let i = this.index + 1; i < STAGES.length; i++) {
      if (STAGES[i].start.test(line)) {
        this.index = i;
        this.startedAt[i] = this.now();
        this.flowsSeen = 0;
        this.flowFraction = null;
        this.cascade = /cascade/.test(line);
        return true;
      }
    }
    const flow = FLOW_STEP.exec(line);
    if (flow && this.index >= 0) {
      const done = Number(flow[1]);
      const steps = Number(flow[2]) || 1;
      if (done === 0) this.flowsSeen++;
      const tracking = !this.cascade || this.flowsSeen >= 2;
      // The flow is most of its stage; the decode that follows is the rest.
      this.flowFraction = tracking ? (done / steps) * 0.9 : null;
      return true;
    }
    return false;
  }

  /** Current stage and percent, or null before the first stage header. */
  progress(): StageProgress | null {
    if (this.index < 0) return null;
    const total = this.expected.reduce((a, b) => a + b, 0);
    const before = this.expected.slice(0, this.index).reduce((a, b) => a + b, 0);
    const here = this.expected[this.index];
    const inStage = (this.now() - this.startedAt[this.index]) / 1000;
    const byTime = Math.min(inStage / here, 0.95);
    const within = this.flowFraction ?? (this.cascade ? Math.min(byTime, 0.1) : byTime);
    const pct = Math.floor(((before + within * here) / total) * 100);
    this.shown = Math.min(99, Math.max(this.shown, pct));
    return { index: this.index, total: STAGES.length, name: STAGES[this.index].name, percent: this.shown };
  }

  /**
   * The model arrived: record how long each stage took on this machine, so the
   * next run at this resolution estimates from real numbers. Stages that were
   * skipped keep their previous estimate. Blends with the previous measurement
   * so one unusual run doesn't swing it.
   */
  finish(): void {
    if (this.index < 0) return;
    const end = this.now();
    const all = loadMeasured();
    const prev = all[String(this.resolution)];
    const next = this.expected.slice();
    for (let i = 0; i < STAGES.length; i++) {
      const start = this.startedAt[i];
      if (start == null) continue;
      const nextStart = this.startedAt.slice(i + 1).find((t) => t != null) ?? end;
      const secs = Math.max(0.1, (nextStart - start) / 1000);
      next[i] = prev?.[i] != null ? (prev[i] + secs) / 2 : secs;
    }
    all[String(this.resolution)] = next.map((s) => Math.round(s * 10) / 10);
    saveMeasured(all);
  }
}
