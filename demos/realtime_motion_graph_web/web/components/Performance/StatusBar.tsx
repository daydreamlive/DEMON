"use client";

import { useSessionStore } from "@/store/useSessionStore";

// Status placard. Reads from the session store directly so consumers
// don't have to thread props through their performance shell. The
// `data-state` attribute drives all of the visual treatment via
// CSS — see the `.status-bar` rules in styles.css.
//
// State derivation:
//   - `idle` (no session) and `ready` + "Playing" → hidden.
//   - `loading-fixture` / `connecting` → "loading" (accent orange,
//     calm breathing).
//   - `error` / `closed` → "error" (warn yellow, soft inset glow).
//   - everything else with a non-empty message → "info" (white, mid-
//     session swap progress and friends).

type PlacardState = "loading" | "info" | "error";

function deriveState(
  status: string,
  message: string,
): { visible: boolean; state: PlacardState } {
  if (!message || message === "Playing") {
    return { visible: false, state: "info" };
  }
  if (status === "error" || status === "closed") {
    return { visible: true, state: "error" };
  }
  if (status === "loading-fixture" || status === "connecting") {
    return { visible: true, state: "loading" };
  }
  return { visible: true, state: "info" };
}

export function StatusBar() {
  const status = useSessionStore((s) => s.status);
  const message = useSessionStore((s) => s.message);
  const { visible, state } = deriveState(status, message);

  // While `status === "connecting"` the WS handshake + first-session
  // model warm-up can take up to ~60s — surface that expectation under
  // the main message so users don't think the page froze.
  const showConnectingHint = status === "connecting";

  return (
    <div
      className={`status-bar${visible ? " status-bar--visible" : ""}`}
      data-state={state}
      role="status"
      aria-live="polite"
    >
      <span className="status-bar__text">{message}</span>
      {showConnectingHint && (
        <span className="status-bar__subtitle">
          this might take up to a minute
        </span>
      )}
    </div>
  );
}
