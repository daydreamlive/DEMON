// Retry a fetch on transient failures: network errors (ECONNREFUSED, etc.)
// and 5xx responses. Used by the catalog-listing helpers so the UI
// self-heals when the operator restarts the Python backend while the
// browser tab is still open. Without this, components mount, hit the
// Next dev proxy's 502 Bad Gateway, and never re-fetch until the page
// is reloaded.
//
// Exponential backoff with an 8 s cap and a 3 min total deadline —
// generous enough to cover a DEMON cold start (model load + TRT warmup
// can run a minute or more) but bounded so a permanently-down backend
// doesn't leave the UI looping forever.

export interface FetchWithRetryOptions {
  /** Total time budget before giving up. Defaults to 3 min. */
  deadlineMs?: number;
  signal?: AbortSignal;
}

export async function fetchWithRetry(
  url: string,
  opts: FetchWithRetryOptions = {},
): Promise<Response> {
  const deadline = Date.now() + (opts.deadlineMs ?? 180_000);
  let delay = 500;
  while (true) {
    try {
      const res = await fetch(url, { signal: opts.signal });
      // 4xx is a real failure (404, auth, etc.) — don't retry. 5xx is
      // typically transient (502 from Next dev proxy while the backend
      // is booting, 503 from a backend warming up), so we retry.
      if (res.ok || res.status < 500) return res;
    } catch (err) {
      if (opts.signal?.aborted) throw err;
      // Network error (TypeError from fetch on ECONNREFUSED, DNS failure,
      // etc.) — fall through to the backoff path and retry.
    }
    if (Date.now() >= deadline) {
      throw new Error(`fetchWithRetry: gave up on ${url} after deadline`);
    }
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        opts.signal?.removeEventListener("abort", onAbort);
        resolve();
      }, Math.min(delay, 8000));
      function onAbort() {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      }
      opts.signal?.addEventListener("abort", onAbort, { once: true });
    });
    delay = Math.min(delay * 2, 8000);
  }
}
