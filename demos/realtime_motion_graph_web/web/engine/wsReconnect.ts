// Generic WebSocket reconnect loop with exponential backoff + jitter.
//
// The DEMON backend has no session-resume protocol — every WS handshake
// builds a fresh per-connection generative session (model load, source
// encode, TRT engine bind). So "reconnect" means "re-run the full init
// handshake against the same fixture + LoRA + prompt state the client
// already holds in its zustand stores," not "restore from server-side
// snapshot." The orchestration lives here; the per-session re-init lives
// in useStartSession.ts.
//
// Production failure modes this is built for:
//   - Code 1006 (abnormal closure) when a RunPod / vast.ai / Cloudflare
//     tunnel drops the TCP connection without a WS close frame (the
//     dominant 1006 source we see).
//   - Brief client-side network blips (wifi handoff, VPN reconnect) that
//     surface to the browser as a 1006.
//   - Tunnel idle-timeout closes where the immediate reconnect succeeds.
//
// What we explicitly DON'T try to recover from:
//   - Pod died / OOM / host eviction. Those produce a fast string of
//     ECONNREFUSED on each attempt; we burn a short backoff window
//     then surface an error so the user knows to refresh. Recovery
//     belongs to the pod orchestrator, not the client.
//   - User-initiated session restart (RemoteBackend.closedByUser=true).
//     The new session start owns the reconnect loop's lifecycle, so
//     starting another session cancels any in-flight backoff.
//   - "Engine not built" / "stem extraction failed" errors raised
//     synchronously from the handshake — those are configuration bugs,
//     not network blips, and retrying without operator input would just
//     flood the server.

export interface ReconnectAttempt {
  /** 1-indexed attempt number (1 == first retry after the initial drop). */
  attempt: number;
  maxAttempts: number;
  /** Delay we waited before this attempt fired. Useful for surfacing a
   *  countdown to the user. */
  delayMs: number;
}

export interface ReconnectHandlers {
  /** Fired at the start of each attempt cycle, *before* the backoff
   *  sleep. The supplied `delayMs` is how long we're about to wait
   *  before invoking `connect`, so this is the right hook for a
   *  countdown surface ("retrying in 2s"). The "we're connecting
   *  now" beat is `onSuccess` (success) or the next `onAttempt`
   *  (failure → backoff for the next try). */
  onAttempt?: (info: ReconnectAttempt) => void;
  /** Fired when an attempt's `connect` resolved successfully. */
  onSuccess?: () => void;
  /** Fired when every retry has been exhausted. Pass the last error
   *  the connect attempts produced so the UI can show something useful. */
  onGiveUp?: (lastError: Error) => void;
}

interface ReconnectOptions {
  maxAttempts?: number;
  /** Base backoff in ms before the first retry (subsequent retries
   *  multiply by 2 and clamp to maxDelayMs). 500 ms gives a near-instant
   *  recovery for transient blips without flooding the server on a real
   *  outage. */
  baseDelayMs?: number;
  maxDelayMs?: number;
}

const DEFAULTS: Required<ReconnectOptions> = {
  // Sized for the targeted failure modes (tunnel blip / brief network
  // drop), not for pod-level outages. Doubling base=500ms each attempt
  // with max=4s gives delays of ~0.25-0.5, 0.5-1, 1-2, 2-4, 2-4 s
  // (full-jitter), summing to a worst-case ~12s window before we hand
  // off to "refresh to retry." Long enough that one transient tunnel
  // hiccup almost always recovers; short enough that a real outage
  // doesn't leave the user staring at "Reconnecting…" wondering if the
  // app is alive. Pod death / OOM is the orchestrator's problem; we
  // shouldn't be the layer that papers over it.
  maxAttempts: 5,
  baseDelayMs: 500,
  maxDelayMs: 4000,
};

export class WsReconnector {
  private cancelled = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private resolveSleep: (() => void) | null = null;
  private readonly opts: Required<ReconnectOptions>;

  constructor(
    /** Async factory that builds a fresh connection from current store
     *  state and rejects on connect failure. The reconnector treats any
     *  rejection as "this attempt failed, try again after backoff." */
    private readonly connect: () => Promise<void>,
    private readonly handlers: ReconnectHandlers = {},
    options: ReconnectOptions = {},
  ) {
    this.opts = { ...DEFAULTS, ...options };
  }

  /** Run the backoff loop. Resolves when an attempt succeeds, when the
   *  caller cancels, or when maxAttempts is reached (after invoking
   *  onGiveUp). Never throws — failures flow through the handlers. */
  async run(): Promise<void> {
    let lastErr: Error = new Error("no attempts ran");
    for (let attempt = 1; attempt <= this.opts.maxAttempts; attempt++) {
      if (this.cancelled) return;
      const baseMs = Math.min(
        this.opts.maxDelayMs,
        this.opts.baseDelayMs * 2 ** (attempt - 1),
      );
      // Full-jitter: pick a random delay in [base/2, base]. Decorrelates
      // retries across many clients piling on after the same pod blip,
      // which would otherwise hammer the server with synchronized waves.
      const jittered = Math.round(baseMs * (0.5 + Math.random() * 0.5));
      this.handlers.onAttempt?.({
        attempt,
        maxAttempts: this.opts.maxAttempts,
        delayMs: jittered,
      });
      await this.sleep(jittered);
      if (this.cancelled) return;
      try {
        await this.connect();
        if (this.cancelled) return;
        this.handlers.onSuccess?.();
        return;
      } catch (e) {
        lastErr = e instanceof Error ? e : new Error(String(e));
      }
    }
    if (!this.cancelled) this.handlers.onGiveUp?.(lastErr);
  }

  /** Cancel any in-flight backoff and stop the loop. Safe to call from
   *  any callback (useSessionStore.reset, page unload, fresh session
   *  start) — idempotent. */
  cancel(): void {
    this.cancelled = true;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.resolveSleep) {
      this.resolveSleep();
      this.resolveSleep = null;
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
      this.resolveSleep = resolve;
      this.timer = setTimeout(() => {
        this.resolveSleep = null;
        this.timer = null;
        resolve();
      }, ms);
    });
  }
}
