// Tier C config: boot the real app against the Python transcript-replay
// server (tests/golden/replay_server.py at the repo root) — full browser,
// real AudioWorklet + decode worker, zero GPU.
//
// Two managed servers:
//   1. the replay server, feeding a recorded golden transcript over WS
//      and answering the /api/* capability probes
//   2. `next dev`, with NEXT_PUBLIC_POD_BASE_URL pointed at (1) so the
//      same-origin /api rewrites land on the replay server
//
// Prereqs: refs fetched (`python -m tests.golden.refs_store fetch` from
// the repo root, via the repo venv) and a chromium (`npx playwright
// install chromium`).

import * as path from "node:path";

import { defineConfig } from "playwright/test";

// Playwright transpiles this config to CJS, so __dirname (not
// import.meta) is the portable way to self-locate.
const webRoot = __dirname;
const repoRoot = path.resolve(webRoot, "..", "..", "..");
// Locate the repo venv's interpreter: Windows lays it down under
// Scripts/python.exe, POSIX under bin/python. DEMON_PYTHON overrides
// both (a system python, conda env, or differently-named venv on CI).
const isWin = process.platform === "win32";
const python =
  process.env.DEMON_PYTHON ??
  path.join(
    repoRoot,
    ".venv",
    isWin ? "Scripts" : "bin",
    isWin ? "python.exe" : "python",
  );

export const E2E_SCENARIO = "swap_fixture";
const REPLAY_PORT = 18931;
const WEB_PORT = 3211;
// Recorded gaps divided by 2: fast enough for CI, slow enough that the
// browser-side pipeline (worker decode, worklet writes) sees a realistic
// arrival cadence rather than a burst.
const REPLAY_SPEED = 2;

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 180_000,
  // One worker: both managed servers are single-session resources.
  workers: 1,
  fullyParallel: false,
  reporter: [["list"]],
  use: {
    // localhost, not 127.0.0.1: Next 16's dev server blocks cross-origin
    // access to its dev resources, and it treats the IP form as a
    // different origin from the localhost it binds.
    baseURL: `http://localhost:${WEB_PORT}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        launchOptions: {
          // The worklet needs a running AudioContext without a real
          // user-gesture heuristic (headless click counts, but don't
          // gamble on it).
          args: ["--autoplay-policy=no-user-gesture-required"],
        },
      },
    },
  ],
  webServer: [
    {
      command:
        `"${python}" -m tests.golden.replay_server ` +
        `--scenario ${E2E_SCENARIO} --port ${REPLAY_PORT} ` +
        `--speed ${REPLAY_SPEED}`,
      cwd: repoRoot,
      url: `http://127.0.0.1:${REPLAY_PORT}/api/server-info`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: `npm run dev -- -p ${WEB_PORT}`,
      cwd: webRoot,
      url: `http://localhost:${WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        NEXT_PUBLIC_POD_BASE_URL: `http://127.0.0.1:${REPLAY_PORT}`,
      },
    },
  ],
});
