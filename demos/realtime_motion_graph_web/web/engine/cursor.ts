// Custom cursor: a TIGHT 4-particle constellation (one per ribbon palette
// color) orbiting close to the pointer, plus a precise white center dot.
// Layered with two ephemeral particle systems for party energy:
//
//   • Sparkler trail — every few px of motion the cursor emits a small
//     palette-colored spark with gravity that fades over ~700ms.
//   • Confetti burst — every mousedown spawns a radial cloud.
//
// Renders into a single 2D canvas so the whole effect is one composited
// layer in every browser. The previous DOM-divs + box-shadow + CSS
// `mix-blend-mode: plus-lighter` implementation was reliably hardware-
// composited only in Blink; Gecko fell back to software for the blend
// group, which during slider drags (mousemove + pointermove competing
// for the main thread) caused observable cursor lag. We draw plain
// solid discs (no halo, no blend mode) — the orbital cluster's
// character is in motion, not glow. Audio kicks bump the radii via
// `bloom` so the cluster still breathes with the music.
//
// Tick is driven from the shared render loop (useRenderLoop) via
// tickActiveCursor(now, bloom); a separate RAF would compete for vsync
// and double the per-frame draw surface. Caller renders
// <canvas class="cursor-canvas"> (see <CustomCursor />).

const ORBIT_RADIUS = 9;
const SPRING_K = 0.18;
const DAMPING = 0.78;
const ROTATION_SPEED = 0.0004;
const ATTRACTOR_RANGE = 140;
const ATTRACTOR_GAIN = 0.4;
const BURST_DECAY = 0.9;
const BURST_GAIN = 1.4;

// Palette shared with the perimeter / halo / start-mark ribbons.
const PALETTE = ["#3db6be", "#c7b566", "#f08a48", "#e84f3d"];

// One entry per orbiting particle. `dir` flips orbit direction (some
// clockwise, some counter-clockwise), `speedMul` shifts the rotation rate,
// `radiusMul` shifts the resting orbital radius. Together: an asymmetric,
// organic cluster instead of four points marching in lockstep.
const PARTICLE_CONFIG: Array<{
  color: string;
  dir: 1 | -1;
  speedMul: number;
  radiusMul: number;
  offset: number;
}> = [
  { color: PALETTE[0], dir: 1, speedMul: 0.7, radiusMul: 1.0, offset: 0.0 },
  { color: PALETTE[1], dir: -1, speedMul: 1.05, radiusMul: 0.86, offset: Math.PI * 0.5 },
  { color: PALETTE[2], dir: 1, speedMul: 1.45, radiusMul: 1.16, offset: Math.PI },
  { color: PALETTE[3], dir: -1, speedMul: 0.9, radiusMul: 0.94, offset: Math.PI * 1.5 },
];

// Spark / confetti tuning. Gravity is per-frame at ~60fps; the loop scales
// physics by dt / 16 so motion stays steady on uneven frame timing.
const GRAVITY = 0.16;
const SPARK_LIFE_MS = 700;
const SPARK_MIN_SPEED = 3.2;
const SPARK_PER_FRAME_CAP = 1;
const CONFETTI_LIFE_MS = 900;
const CONFETTI_COUNT = 16;
const CONFETTI_MIN_SPEED = 3.2;
const CONFETTI_MAX_SPEED = 7.0;
// Hard cap on live ephemerals. Each is a data object now (no DOM node),
// so this exists only to bound canvas draw cost during frantic motion.
const MAX_EPHEMERAL = 90;

// Solid-disc sizes (no glow). The white centre dot is bigger than the
// orbital cluster so the precise pointer position reads cleanly while
// the four palette particles add character around it. Audio kicks
// nudge the radii up a touch via `bloom`, so the cluster still
// breathes with the music — just without a blur halo.
const PARTICLE_RADIUS = 2; // 4 px diameter, matches old CSS
const DOT_RADIUS = 3; // 6 px diameter, matches old CSS
const SPARK_RADIUS = 2;

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  slot: number;
  dir: 1 | -1;
  speedMul: number;
  radiusMul: number;
  color: string;
}

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
  const ctxLookup = canvasLookup.getContext("2d");
  if (!ctxLookup) return { tick: () => {}, destroy: () => {} };
  // Hold non-null aliases so the closures below (resize, tick, destroy)
  // don't lose narrowing across the function-declaration boundary.
  const canvas: HTMLCanvasElement = canvasLookup;
  const ctx: CanvasRenderingContext2D = ctxLookup;

  // Backing-store size = window inner * DPR. CSS-style size = window
  // inner. setTransform makes 1 unit in our draw calls = 1 CSS px.
  function resize() {
    const dpr = window.devicePixelRatio || 1;
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

  const particles: Particle[] = PARTICLE_CONFIG.map((cfg) => ({
    x: window.innerWidth / 2,
    y: window.innerHeight / 2,
    vx: 0,
    vy: 0,
    slot: cfg.offset,
    dir: cfg.dir,
    speedMul: cfg.speedMul,
    radiusMul: cfg.radiusMul,
    color: cfg.color,
  }));

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
  let burst = 0;

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
    burst = 1;
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

  const proximity = (dist: number): number => {
    if (dist <= 0) return 1;
    if (dist >= ATTRACTOR_RANGE) return 0;
    const f = 1 - dist / ATTRACTOR_RANGE;
    return f * f;
  };

  let cancelled = false;
  let lastT = 0;
  function tick(t: number, bloom: number) {
    if (cancelled) return;
    const dt = lastT ? Math.min(50, t - lastT) : 16;
    lastT = t;
    const dtScale = dt / 16;

    burst *= BURST_DECAY;
    const burstScale = 1 + burst * BURST_GAIN;

    const vw = window.innerWidth;
    const stretchX = proximity(mouseY);
    const stretchY = Math.max(proximity(mouseX), proximity(vw - mouseX));

    const rX = ORBIT_RADIUS * burstScale * (1 + stretchX * ATTRACTOR_GAIN);
    const rY = ORBIT_RADIUS * burstScale * (1 + stretchY * ATTRACTOR_GAIN);

    // Spring-physics step for the orbital cluster — unchanged from the
    // DOM implementation, only the side effect (DOM mutation → canvas
    // draw) differs.
    for (const p of particles) {
      p.slot += ROTATION_SPEED * dt * p.dir * p.speedMul;
      const targetX = mouseX + Math.cos(p.slot) * rX * p.radiusMul;
      const targetY = mouseY + Math.sin(p.slot) * rY * p.radiusMul;
      p.vx = (p.vx + (targetX - p.x) * SPRING_K) * DAMPING;
      p.vy = (p.vy + (targetY - p.y) * SPRING_K) * DAMPING;
      p.x += p.vx;
      p.y += p.vy;
    }

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
    // nudge the radii (via `bloom`) so the cluster still breathes
    // with the music without smearing out into a glow.
    const w = window.innerWidth;
    const h = window.innerHeight;
    ctx.clearRect(0, 0, w, h);

    const sizeBoost = 1 + bloom * 0.4;

    // Orbital cluster — 4 px palette dots.
    for (const p of particles) {
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, PARTICLE_RADIUS * sizeBoost, 0, Math.PI * 2);
      ctx.fill();
    }

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
