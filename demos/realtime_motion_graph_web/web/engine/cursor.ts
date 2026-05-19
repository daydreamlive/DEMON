// Custom cursor: a precise white center dot, plus a palette-colored
// sparkler trail behind motion and a radial confetti burst on
// mousedown. The 4-particle orbital constellation that used to ring
// the pointer was retired — it added visual noise on every cursor
// movement and competed with the trail for attention.
//
// Renders into a single 2D canvas so the whole effect is one composited
// layer in every browser. The previous DOM-divs + box-shadow + CSS
// `mix-blend-mode: plus-lighter` implementation was reliably hardware-
// composited only in Blink; Gecko fell back to software for the blend
// group, which during slider drags (mousemove + pointermove competing
// for the main thread) caused observable cursor lag. We draw plain
// solid discs (no halo, no blend mode). Audio kicks bump the radii via
// `bloom` so the trail still breathes with the music.
//
// Tick is driven from the shared render loop (useRenderLoop) via
// tickActiveCursor(now, bloom); a separate RAF would compete for vsync
// and double the per-frame draw surface. Caller renders
// <canvas class="cursor-canvas"> (see <CustomCursor />).

// Palette shared with the perimeter / halo / start-mark ribbons.
const PALETTE = ["#3db6be", "#c7b566", "#f08a48", "#e84f3d"];

// Spark / confetti tuning. Gravity is per-frame at ~60fps; the loop scales
// physics by dt / 16 so motion stays steady on uneven frame timing.
//
// SPARK_LIFE_MS + SPARK_MIN_SPEED tuned 2026-05 in response to user
// feedback that the sparks felt like "glitter" and were annoying —
// life dropped from 700ms (visible long trails) to 420ms (sparks
// fade before the trail visually extends), and min-speed raised from
// 3.2 to 5 so idle micro-jitter doesn't emit sparks at all.
const GRAVITY = 0.16;
const SPARK_LIFE_MS = 420;
const SPARK_MIN_SPEED = 5;
const SPARK_PER_FRAME_CAP = 1;
const CONFETTI_LIFE_MS = 900;
const CONFETTI_COUNT = 16;
const CONFETTI_MIN_SPEED = 3.2;
const CONFETTI_MAX_SPEED = 7.0;
// Hard cap on live ephemerals. Each is a data object now (no DOM node),
// so this exists only to bound canvas draw cost during frantic motion.
const MAX_EPHEMERAL = 90;

// Solid-disc sizes (no glow). Audio kicks nudge the radii up a touch
// via `bloom`, so the trail still breathes with the music — just
// without a blur halo.
const DOT_RADIUS = 3; // 6 px diameter, matches old CSS
const SPARK_RADIUS = 2;

interface Ephemeral {
  x: number;
  y: number;
  vx: number;
  vy: number;
  age: number; // ms
  life: number; // ms
  color: string;
}

export interface CursorHandle {
  tick: (now: number, bloom: number) => void;
  destroy: () => void;
}

// Module-level singleton: at most one cursor instance is active at a time
// (the Performance page mounts it once). The render loop calls
// tickActiveCursor(now, bloom) each frame; if no cursor is mounted, it's
// a no-op.
let activeCursor: CursorHandle | null = null;

export function tickActiveCursor(now: number, bloom = 0): void {
  activeCursor?.tick(now, bloom);
}

export function initCursor(): CursorHandle {
  const canvasLookup = document.querySelector(
    ".cursor-canvas",
  ) as HTMLCanvasElement | null;
  if (!canvasLookup) return { tick: () => {}, destroy: () => {} };
  // `desynchronized: true` is the spec'd low-latency path for
  // input-following overlays — it lets the browser bypass the standard
  // compositor frame queue when possible, cutting mousemove→photon by
  // ~1 frame in Chromium. No-op fallback elsewhere. Pairs with the GPU
  // layer hint on .cursor-canvas in app/globals.css.
  const ctxLookup = canvasLookup.getContext("2d", { desynchronized: true });
  if (!ctxLookup) return { tick: () => {}, destroy: () => {} };
  // Hold non-null aliases so the closures below (resize, tick, destroy)
  // don't lose narrowing across the function-declaration boundary.
  const canvas: HTMLCanvasElement = canvasLookup;
  const ctx: CanvasRenderingContext2D = ctxLookup;

  // Backing-store size = window inner * DPR. CSS-style size = window
  // inner. setTransform makes 1 unit in our draw calls = 1 CSS px.
  function resize() {
    // Cap DPR at 2 (matches GraphRenderer). On 3×-DPR phones the extra
    // fragment cost — ~2.25× per frame for a full-viewport canvas — is
    // imperceptible on solid 2-3px discs.
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  resize();
  window.addEventListener("resize", resize, { passive: true });

  const ephemerals: Ephemeral[] = [];

  function spawnSpark(
    x: number,
    y: number,
    vx: number,
    vy: number,
    life: number,
    color: string,
  ) {
    if (ephemerals.length >= MAX_EPHEMERAL) ephemerals.shift();
    ephemerals.push({ x, y, vx, vy, age: 0, life, color });
  }

  let mouseX = window.innerWidth / 2;
  let mouseY = window.innerHeight / 2;
  let lastMoveX = mouseX;
  let lastMoveY = mouseY;

  const onMove = (e: MouseEvent) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    const dx = mouseX - lastMoveX;
    const dy = mouseY - lastMoveY;
    const speed = Math.hypot(dx, dy);
    if (speed > SPARK_MIN_SPEED) {
      // One spark per move event, capped. Spark inherits a small slice
      // of the pointer's reverse velocity (so it trails behind motion)
      // plus a touch of jitter.
      for (let i = 0; i < SPARK_PER_FRAME_CAP; i++) {
        const color = PALETTE[Math.floor(Math.random() * PALETTE.length)];
        const trailFactor = 0.06;
        const sparkVx = -dx * trailFactor + (Math.random() - 0.5) * 1.5;
        const sparkVy = -dy * trailFactor + (Math.random() - 0.5) * 1.5;
        spawnSpark(mouseX, mouseY, sparkVx, sparkVy, SPARK_LIFE_MS, color);
      }
    }
    lastMoveX = mouseX;
    lastMoveY = mouseY;
  };

  const onDown = (e: MouseEvent) => {
    // Confetti — a radial cloud at the click site. Each particle gets a
    // random angle and speed, plus a small upward bias so the cloud
    // arcs nicely under gravity instead of just splatting straight.
    for (let i = 0; i < CONFETTI_COUNT; i++) {
      const color = PALETTE[Math.floor(Math.random() * PALETTE.length)];
      const angle = Math.random() * Math.PI * 2;
      const speed =
        CONFETTI_MIN_SPEED +
        Math.random() * (CONFETTI_MAX_SPEED - CONFETTI_MIN_SPEED);
      const vx = Math.cos(angle) * speed;
      const vy = Math.sin(angle) * speed - 1.2;
      spawnSpark(e.clientX, e.clientY, vx, vy, CONFETTI_LIFE_MS, color);
    }
  };

  document.addEventListener("mousemove", onMove, { passive: true });
  document.addEventListener("mousedown", onDown, { passive: true });

  let cancelled = false;
  let lastT = 0;
  function tick(t: number, bloom: number) {
    if (cancelled) return;
    const dt = lastT ? Math.min(50, t - lastT) : 16;
    lastT = t;
    const dtScale = dt / 16;

    // Ephemeral physics + lifetime cull. Walking backwards lets us
    // splice without index issues.
    for (let i = ephemerals.length - 1; i >= 0; i--) {
      const s = ephemerals[i];
      s.age += dt;
      if (s.age >= s.life) {
        ephemerals.splice(i, 1);
        continue;
      }
      s.vy += GRAVITY * dtScale;
      s.x += s.vx * dtScale;
      s.y += s.vy * dtScale;
    }

    // ── Render ────────────────────────────────────────────────────────
    // Solid discs only — no halos, no additive blend. Audio kicks
    // nudge the radii (via `bloom`) so the trail still breathes with
    // the music without smearing out into a glow.
    const w = window.innerWidth;
    const h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);

    const sizeBoost = 1 + bloom * 0.4;

    // Sparks + confetti — fade and shrink by life fraction.
    for (const s of ephemerals) {
      const lifeFrac = s.age / s.life;
      const opacity = 1 - lifeFrac;
      const r = SPARK_RADIUS * sizeBoost * (1 - lifeFrac * 0.7);
      if (r <= 0.1) continue;
      ctx.globalAlpha = opacity;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // White centre dot — drawn last so it sits on top of the cluster
    // and acts as the precise pointer indicator (the OS cursor is
    // hidden by `body.cursor-hidden { cursor: none }`).
    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, DOT_RADIUS * sizeBoost, 0, Math.PI * 2);
    ctx.fill();
  }

  const handle: CursorHandle = {
    tick,
    destroy: () => {
      cancelled = true;
      window.removeEventListener("resize", resize);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mousedown", onDown);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ephemerals.length = 0;
      if (activeCursor === handle) activeCursor = null;
    },
  };
  activeCursor = handle;
  return handle;
}
