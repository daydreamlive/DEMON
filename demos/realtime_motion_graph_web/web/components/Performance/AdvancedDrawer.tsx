"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useIsMobile } from "@/hooks/useIsMobile";
import { useOneShotTooltip } from "@/hooks/useOneShotTooltip";
import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import {
  AdvancedCoachmark,
  advancedCoachmarkStorageKey,
} from "./AdvancedCoachmark";
import { ChannelGainsTile } from "./ChannelGainsTile";
import { ChannelsTile } from "./ChannelsTile";
import { DcwTile } from "./DcwTile";
import { EngineTile } from "./EngineTile";
import { LibraryTile } from "./LibraryTile";
import { LiteControls } from "./LiteControls";
import { MainTile } from "./MainTile";
import { MobileFullSheet } from "./MobileFullSheet";
import { OperatorStrip } from "./OperatorStrip";
import { PromptsTile } from "./PromptsTile";
import { SeedTile } from "./SeedTile";

// Slide-up Advanced Controls drawer. Behavior splits at the mobile
// breakpoint: desktop shows the dense mixer-board layout; mobile shows a
// "Lite" layout (Structure + seed + prompt) with an "All controls" link
// that opens a full-screen tabbed sheet. The handle is disabled while the
// session is idle.

export function AdvancedDrawer() {
  const [open, setOpen] = useState(false);
  const [allOpen, setAllOpen] = useState(false);
  const isMobile = useIsMobile();
  const status = useSessionStore((s) => s.status);
  const showKbdHints = usePerformanceStore((s) => s.showKbdHints);
  const started = status !== "idle";

  // useKeyboardShortcuts dispatches this on Esc / `o`.
  useEffect(() => {
    const handler = () => {
      if (!started) return;
      setOpen((v) => !v);
    };
    document.addEventListener("dd:toggle-drawer", handler);
    return () => document.removeEventListener("dd:toggle-drawer", handler);
  }, [started]);

  // Force-close on any transition back to idle (session reset).
  useEffect(() => {
    if (!started) {
      setOpen(false);
      setAllOpen(false);
    }
  }, [started]);

  // Auto-close when the SCHEDULE CURVES overlay opens. The two are
  // mutually exclusive working modes — drawing curves over the graph
  // vs. dragging sliders in the mixer — and stacking them just shrinks
  // both. When the user opens the curves overlay, hide the drawer
  // (state preserved; reopens on the next dd:toggle-drawer).
  const overlayOpen = useCurveStore((s) => s.overlayOpen);
  useEffect(() => {
    if (overlayOpen) {
      setOpen(false);
      setAllOpen(false);
    }
  }, [overlayOpen]);

  // Mirror open state to body.drawer-open so the existing CSS rule
  // `body[data-mode="graph"].drawer-open #install-stage { bottom: var(--drawer-h); }`
  // shrinks the stage (and the embedded canvases) when the drawer slides up.
  // ResizeObserver inside HUD/Graph fires on the resulting size change.
  useEffect(() => {
    document.body.classList.toggle("drawer-open", open);
    return () => {
      document.body.classList.remove("drawer-open");
    };
  }, [open]);

  // ─── HINT SEQUENCE — Stage D: advanced-controls coachmark ─────────
  // See AdvancedCoachmark.tsx for the full multi-stage hint contract.
  // tl;dr — don't fire on session-ready (would compete with the
  // top-ribbon RemixHint, which is Stage B). Wait until the user
  // clears the per-song remix gate (drags the top ribbon — Stage C
  // signal: usePerformanceStore.remixStarted flips true), THEN delay
  // ~12s so the side-ribbon RemixHints land + get noticed without our
  // coachmark crowding them.
  const remixStarted = usePerformanceStore((s) => s.remixStarted);
  const remixGateClearedAt = useRef<number | null>(null);
  const [coachmarkVisible, setCoachmarkVisible] = useState(false);

  const handleCoachmarkDismiss = useCallback(() => {
    setCoachmarkVisible(false);
    try {
      localStorage.setItem(advancedCoachmarkStorageKey, "1");
    } catch {
      // localStorage may be unavailable (private browsing, quota). The
      // worst case is the user sees the coachmark next session; not
      // worth crashing the drawer over.
    }
  }, []);

  // Capture the timestamp when the remix gate first clears this
  // session. remixStarted resets to false per fixture swap (so each
  // new song re-shows the "drag to start" hint), but the coachmark
  // shouldn't get a second chance — once cleared, leave the
  // timestamp pinned for the lifetime of the page.
  useEffect(() => {
    if (remixStarted && remixGateClearedAt.current === null) {
      remixGateClearedAt.current = Date.now();
    }
  }, [remixStarted]);

  // Schedule the coachmark once the gate + delay are satisfied.
  useEffect(() => {
    if (isMobile) return;
    if (status !== "ready") return;
    if (open) return; // user already discovered the drawer some other way
    if (remixGateClearedAt.current === null) return; // remix gate not yet cleared
    try {
      if (localStorage.getItem(advancedCoachmarkStorageKey) === "1") return;
    } catch {
      // If we can't read localStorage, treat the user as first-run;
      // showing the coachmark once is friendlier than never showing it.
    }
    const DELAY_MS = 12_000;
    const elapsed = Date.now() - remixGateClearedAt.current;
    if (elapsed >= DELAY_MS) {
      setCoachmarkVisible(true);
      return;
    }
    const t = window.setTimeout(() => {
      // Re-check `open` at fire-time — user might have opened the
      // drawer during the delay window.
      if (useSessionStore.getState().status !== "ready") return;
      // Note: we don't have access to `open` here without re-reading,
      // but the next effect (open → dismiss) will catch that race.
      setCoachmarkVisible(true);
    }, DELAY_MS - elapsed);
    return () => window.clearTimeout(t);
  }, [isMobile, status, open, remixStarted]);

  // If the user opens the drawer some other way (keyboard, click on
  // the handle, future custom event), retire the coachmark.
  useEffect(() => {
    if (open && coachmarkVisible) {
      handleCoachmarkDismiss();
    }
  }, [open, coachmarkVisible, handleCoachmarkDismiss]);

  return (
    <>
      <aside
        id="install-sheet"
        className={`install-sheet${open ? " open" : ""}${isMobile ? " install-sheet--mobile" : ""}`}
        aria-hidden={!open}
      >
        <DrawerHandle started={started} open={open} setOpen={setOpen} />

        <div className="install-sheet-body">
          {isMobile ? (
            <LiteControls onOpenAllControls={() => setAllOpen(true)} />
          ) : (
            <>
              <OperatorStrip />
              <div
                className={`mixer-rack${!showKbdHints ? " mixer-rack--no-kbd-hints" : ""}`}
                id="mixer-tiles"
              >
                {/* Single row with every tile, sharing the full width
                    equally, plus a full-width prompts row beneath. CSS
                    on .mixer-rack-row > .mixer-tile stretches each tile
                    to an equal share via `flex: 1 1 0` + `min-width: 0`. */}
                <div className="mixer-rack-row" data-rack-row="all">
                  <MainTile />
                  <EngineTile />
                  <ChannelGainsTile />
                  <ChannelsTile />
                  <DcwTile />
                  <LibraryTile />
                  <SeedTile />
                </div>
                <PromptsTile />
              </div>
            </>
          )}
        </div>
      </aside>

      {isMobile && (
        <MobileFullSheet
          open={allOpen}
          onClose={() => setAllOpen(false)}
        />
      )}

      <AdvancedCoachmark
        visible={coachmarkVisible}
        onDismiss={handleCoachmarkDismiss}
      />
    </>
  );
}

// Extracted so useOneShotTooltip lives in its own component scope —
// keeps AdvancedDrawer's hook list clean (it already runs five effects
// + four store subscriptions) and avoids any chance of conditionally
// calling the hook based on `started` early-returns. The tooltip only
// applies the [data-dd-tooltip] attribute the first time a user hovers
// the handle; afterwards the persistent label + the AdvancedCoachmark
// (Stage D) carry the affordance.
interface DrawerHandleProps {
  started: boolean;
  open: boolean;
  setOpen: (fn: (v: boolean) => boolean) => void;
}
function DrawerHandle({ started, open, setOpen }: DrawerHandleProps) {
  const tipProps = useOneShotTooltip(
    "advanced-drawer",
    started ? "Advanced Controls (o)" : "Press Play to enable",
  );
  void open; // accepted but no longer needed in this body — kept in signature for clarity
  return (
    <button
      id="install-adv-handle"
      className={`install-drawer-handle${started ? "" : " install-drawer-handle--disabled"}`}
      aria-label="Toggle advanced controls drawer"
      aria-disabled={!started}
      {...tipProps}
      onClick={() => {
        if (!started) return;
        setOpen((v) => !v);
      }}
      disabled={!started}
      type="button"
    >
      <span className="install-drawer-handle-grip" aria-hidden="true" />
      <span className="install-drawer-handle-label">Advanced Controls</span>
      <span className="install-drawer-handle-grip" aria-hidden="true" />
    </button>
  );
}
