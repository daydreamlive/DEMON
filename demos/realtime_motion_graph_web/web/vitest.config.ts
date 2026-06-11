import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

const webRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    // Mirror tsconfig's aliases so the tests import the REAL app/SDK
    // modules (sdk/protocol.ts etc.), not copies.
    alias: {
      "@demon/client": path.join(webRoot, "sdk", "index.ts"),
      "@": webRoot,
    },
  },
  test: {
    environment: "node",
    include: [
      "tests/unit/**/*.test.ts",
      "tests/replay/**/*.test.ts",
    ],
    // Replay tests feed thousands of recorded wire frames through the
    // real client decode path; generous ceiling so a slow CI box never
    // flakes on wall-clock.
    testTimeout: 120_000,
  },
});
