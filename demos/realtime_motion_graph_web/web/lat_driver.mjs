// Playwright driver: opens the demo, enables the [lat] client trace, starts
// a session by clicking the start overlay, then stays alive so MCP can turn
// the denoise knob like a human while we read console + server logs.
import { chromium } from "playwright";

const URL = "http://localhost:6660";

const browser = await chromium.launch({
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
});
const ctx = await browser.newContext();
const page = await ctx.newPage();

// Set the trace flag BEFORE app code runs.
await page.addInitScript(() => {
  window.__demonLatTrace = true;
});

page.on("console", (msg) => {
  const t = msg.text();
  if (t.includes("[lat]") || t.includes("[protocol]") || t.includes("RTMG")) {
    console.log(t);
  }
});
page.on("pageerror", (e) => console.log("PAGEERROR", e.message));

await page.goto(URL, { waitUntil: "domcontentloaded" });
console.log("LOADED", URL);
await page.waitForTimeout(2000);

// Click the start-overlay CTA (aria-label "Click to begin").
try {
  const cta = page.locator('button[aria-label="Click to begin"]');
  await cta.waitFor({ state: "visible", timeout: 8000 });
  await cta.click({ force: true });
  console.log("clicked start CTA");
} catch (e) {
  console.log("startclick err", e.message);
}
await page.waitForTimeout(2500);
await page.screenshot({ path: "/tmp/demon_page.png" }).catch(() => {});
console.log("screenshot saved");

// Heartbeat: report the live playhead so latency can be correlated even
// between slices. Reads positionSec off the session store's player.
setInterval(async () => {
  try {
    const pos = await page.evaluate(() => {
      const s = window.__SESSION_STORE__;
      return null; // store not globally exposed; rely on [lat] slice logs
    });
  } catch {}
}, 2000);

console.log("DRIVER_READY");
await new Promise(() => {});
