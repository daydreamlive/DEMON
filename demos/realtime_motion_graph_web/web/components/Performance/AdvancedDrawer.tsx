"use client";

import { useEffect, useState, type ReactNode } from "react";

import { useIsMobile } from "@/hooks/useIsMobile";
import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import { CoreTile } from "./CoreTile";
import { DrawerTabs, useDrawerTab, type DrawerTab } from "./DrawerTabs";
import { LibraryTile } from "./LibraryTile";
import { LiteControls } from "./LiteControls";
import { MobileFullSheet } from "./MobileFullSheet";
import { ModTile } from "./ModTile";
import { OperatorStrip } from "./OperatorStrip";
import { PromptsTile } from "./PromptsTile";
import { VoiceTile } from "./VoiceTile";

// Slide-up Full Controls drawer. Behavior splits at the mobile
// breakpoint: desktop shows the dense mixer-board layout; mobile shows a
// "Lite" layout (Structure + seed + prompt) with an "All controls" link
// that opens a full-screen tabbed sheet. The HeroMacros "Full Controls
// ▸ / Simple Controls ◂" toggle at the bottom of the canvas is the sole
// open/close affordance — it dispatches `dd:toggle-drawer`, which this
// drawer listens for.

interface Props {
  /** Slot for the "Saved" tab body. Mounted by demon-public-demo to
   *  inject its <SessionsTile /> (which depends on auth + /api/sessions
   *  and therefore can't live in DEMON's standalone bundle). When
   *  omitted, the tab shows a small "unavailable" placeholder. */
  savedTab?: ReactNode;
}

export function AdvancedDrawer({ savedTab }: Props = {}) {
  const [open, setOpen] = useState(false);
  const [allOpen, setAllOpen] = useState(false);
  const [activeTab, setActiveTab] = useDrawerTab("core");
  const isMobile = useIsMobile();
  const status = useSessionStore((s) => s.status);
  const showKbdHints = usePerformanceStore((s) => s.showKbdHints);
  const started = status !== "idle";

  // useKeyboardShortcuts dispatches this on Esc / `o`. HeroMacros'
  // toggle button also dispatches the same event.
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

  return (
    <>
      <aside
        id="install-sheet"
        className={`install-sheet${open ? " open" : ""}${isMobile ? " install-sheet--mobile" : ""}`}
        aria-hidden={!open}
      >
        {!isMobile && (
          <button
            type="button"
            className="install-sheet-edge-handle"
            onClick={() => started && setOpen((v) => !v)}
            disabled={!started}
            aria-label={open ? "Close Full Controls" : "Open Full Controls"}
            aria-expanded={open}
          >
            <span className="install-sheet-edge-handle-caret" aria-hidden="true">
              {open ? "◂" : "▸"}
            </span>
          </button>
        )}
        <div className="install-sheet-body">
          {isMobile ? (
            <LiteControls onOpenAllControls={() => setAllOpen(true)} />
          ) : (
            <>
              <div className="install-sheet-topbar">
                <DrawerTabs active={activeTab} onChange={setActiveTab} />
              </div>
              <div
                className={`mixer-rack mixer-rack--tabbed${!showKbdHints ? " mixer-rack--no-kbd-hints" : ""}`}
                id="mixer-tiles"
                data-active-tab={activeTab}
              >
                {renderTabBody(activeTab, savedTab)}
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
    </>
  );
}

// Tab body switch — kept as a plain function (not a component) because
// every tile already runs its own hooks/subscriptions, and a wrapping
// React component would just add another re-render layer with no
// upside.
function renderTabBody(tab: DrawerTab, savedTab?: ReactNode) {
  switch (tab) {
    case "core":
      return <CoreTile />;
    case "mod":
      return <ModTile />;
    case "voice":
      return <VoiceTile />;
    case "prompt":
      return <PromptsTile />;
    case "lib":
      return <LibraryTile />;
    case "saved":
      return savedTab ?? (
        <div className="install-sheet-saved-placeholder">
          Saved sessions are only available in the hosted app.
        </div>
      );
    case "config":
      return <OperatorStrip />;
  }
}
