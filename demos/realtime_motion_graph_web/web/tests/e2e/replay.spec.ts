// Tier C: full-app smoke + browser-side performance against the
// transcript replay server (see playwright.config.ts for the wiring).
//
// Drives the real UI through start-session -> ready -> knob -> swap on
// the swap_fixture recording, asserts the audio buffer actually carries
// signal, that the playhead advances without stalls (worklet underrun
// proxy), that no console errors fire, and writes the browser-side
// streaming stats (decode-worker slice cadence, slice lead over the
// playhead, stale-slice drops at the swap) to
// runs/web-replay-reports/e2e-<scenario>.json.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { expect, test } from "playwright/test";

// Pulls the `declare global { interface Window { __demonTest } }`
// augmentation into scope for the page.evaluate callbacks.
import type {} from "@/engine/testHooks";

import { E2E_SCENARIO } from "../../playwright.config";

const SWAP_TARGET = "prog_rock_loop_60s_enm.wav";

// Playwright transpiles specs to CJS; __dirname over import.meta.
const webRoot = path.resolve(__dirname, "..", "..");
const reportDir = path.join(
  path.resolve(webRoot, "..", "..", ".."),
  "runs",
  "web-replay-reports",
);

// Reference bundle for the replayed scenario (the replay server already
// refused to boot if this cache is missing, so reading it here is safe).
const refsDir = path.join(
  os.homedir(),
  ".cache",
  "demon",
  "test-refs",
  E2E_SCENARIO,
);
const refMetrics = JSON.parse(
  fs.readFileSync(path.join(refsDir, "metrics.json"), "utf-8"),
) as {
  canonical_sha256: string;
  canonical_region: { start_frame: number; end_frame: number };
  n_slices: number;
};

test("start -> ready -> knob -> swap on a replayed session", async ({
  page,
}) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  page.on("pageerror", (err) => consoleErrors.push(String(err)));

  await page.goto("/");
  await page.waitForFunction(() => !!window.__demonTest);

  // ── start session via the real Play affordance ───────────────────────
  await page.getByRole("button", { name: /click to begin/i }).click();
  await expect
    .poll(
      () =>
        page.evaluate(
          () =>
            `${window.__demonTest!.getStatus()} | ${window.__demonTest!.getMessage()}`,
        ),
      { timeout: 60_000, message: "session never reached ready" },
    )
    .toMatch(/^ready/);

  expect(await page.evaluate(() => window.__demonTest!.startProbe())).toBe(
    true,
  );

  // ── streaming health before any interaction ──────────────────────────
  await page.waitForTimeout(3_000);
  const pos1 = await page.evaluate(() =>
    window.__demonTest!.getPositionSec(),
  );
  await page.waitForTimeout(1_000);
  const pos2 = await page.evaluate(() =>
    window.__demonTest!.getPositionSec(),
  );
  expect(pos1, "playhead is live").toBeGreaterThan(0);
  expect(pos2!, "playhead advances").toBeGreaterThan(pos1!);

  const rmsInitial = await page.evaluate(() =>
    window.__demonTest!.bufferRms(),
  );
  expect(rmsInitial, "initial buffer carries audio").toBeGreaterThan(0.01);

  // ── knob: same store action a ribbon drag makes; useParamSync's 8 ms
  //    tick ships it over the live WS as `params` ────────────────────────
  const knobAt = Date.now();
  await page.evaluate(() => window.__demonTest!.setSlider("denoise", 0.7));
  await page.waitForTimeout(1_000);

  // ── swap: the store write the TrackPicker makes; useFixtureSwap turns
  //    it into swap_source, the replay gate answers with the recorded
  //    swap_ready + buffer ─────────────────────────────────────────────
  await page.evaluate(
    (name) => window.__demonTest!.setFixture(name),
    SWAP_TARGET,
  );
  await expect
    .poll(() => page.evaluate(() => window.__demonTest!.getSwapCount()), {
      timeout: 60_000,
      message: "swap never completed",
    })
    .toBe(1);
  await expect
    .poll(
      () => page.evaluate(() => window.__demonTest!.getStatus()),
      { timeout: 30_000 },
    )
    .toBe("ready");

  // Post-swap: buffer replaced (different track), still audible, and the
  // playhead keeps moving (restart_song_on_swap seeks to 0, so just
  // require forward motion from wherever it is now).
  const rmsAfterSwap = await page.evaluate(() =>
    window.__demonTest!.bufferRms(),
  );
  expect(rmsAfterSwap, "post-swap buffer carries audio").toBeGreaterThan(
    0.01,
  );
  const pos3 = await page.evaluate(() =>
    window.__demonTest!.getPositionSec(),
  );
  await page.waitForTimeout(2_000);
  const pos4 = await page.evaluate(() =>
    window.__demonTest!.getPositionSec(),
  );
  expect(pos4!, "playhead advances after swap").toBeGreaterThan(pos3!);

  // ── canonical audio: browser buffer vs the reference bundle ──────────
  // Let the replay stream the rest of the transcript through the REAL
  // worker decode path, then hash the player mirror's canonical region
  // (post-swap position space) and compare against the sha the golden
  // runner recorded for canonical.f32.raw. Bit-exact, same bar as the
  // Node replay tier — this is the browser actually carrying the
  // reference audio in its buffer.
  let lastCount = -1;
  await expect
    .poll(
      async () => {
        const n = (await page.evaluate(
          () => window.__demonTest!.probeStats(),
        ))!.n_slices as number;
        const stable = n === lastCount;
        lastCount = n;
        return stable && n > 0 ? "drained" : `streaming (${n})`;
      },
      {
        timeout: 120_000,
        intervals: [2_000],
        message: "replay transcript never drained",
      },
    )
    .toBe("drained");
  const stalePostDrain = (await page.evaluate(
    () => window.__demonTest!.probeStats(),
  ))!.n_slices as number;
  // Every recorded slice arrived (minus any stale-epoch drops at the
  // swap boundary, which the canonical region is anchored to exclude).
  expect(stalePostDrain).toBeGreaterThan(refMetrics.n_slices * 0.9);

  const browserSha = await page.evaluate(
    ({ start, end }) =>
      window.__demonTest!.bufferRegionSha256(start, end),
    {
      start: refMetrics.canonical_region.start_frame,
      end: refMetrics.canonical_region.end_frame,
    },
  );
  expect(
    browserSha,
    "browser-reconstructed canonical region must match the reference " +
      "bundle bit-exactly (diverged: debug with npm run test:replay, " +
      "which has full diff tooling for the same comparison)",
  ).toBe(refMetrics.canonical_sha256);

  // ── collect browser-side perf stats ──────────────────────────────────
  const stats = (await page.evaluate(() =>
    window.__demonTest!.probeStats(),
  )) as Record<string, unknown> & {
    n_slices: number;
    playhead_stalls: number;
    stale_slices_dropped: number;
  };
  expect(stats).not.toBeNull();
  // Slices flowed through the worker decode path the whole session.
  expect(stats.n_slices).toBeGreaterThan(100);
  // Worklet underrun proxy: the playhead never froze while "ready".
  expect(stats.playhead_stalls).toBe(0);

  const wsTrace = await page.evaluate(() =>
    window.__demonTest!.getWsTrace(),
  );

  fs.mkdirSync(reportDir, { recursive: true });
  const report = {
    scenario: E2E_SCENARIO,
    swap_target: SWAP_TARGET,
    knob_at_epoch_ms: knobAt,
    canonical_sha256: browserSha,
    canonical_matches_reference: browserSha === refMetrics.canonical_sha256,
    stats,
    ws_trace: wsTrace,
    console_errors: consoleErrors,
  };
  const file = path.join(reportDir, `e2e-${E2E_SCENARIO}.json`);
  fs.writeFileSync(file, JSON.stringify(report, null, 2) + "\n");
  console.log(`[e2e] stats -> ${file}`);
  console.log(
    `[e2e] slices=${stats.n_slices} stalls=${stats.playhead_stalls} ` +
      `stale_dropped=${stats.stale_slices_dropped}`,
  );

  // ── console must be clean ─────────────────────────────────────────────
  expect(consoleErrors, consoleErrors.join("\n---\n")).toHaveLength(0);
});
