"use client";

import { useEffect, useState, type ReactNode } from "react";

import { useIsMobile } from "@/hooks/useIsMobile";
import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import { CoreTile } from "./CoreTile";
import { DrawerHelpBar } from "./DrawerHelpBar";
import { DrawerTabs, useDrawerTab, type DrawerTab } from "./DrawerTabs";
import { LibraryTile } from "./LibraryTile";
import { LiteControls } from "./LiteControls";
import { MobileFullSheet } from "./MobileFullSheet";
import { ModTile } from "./ModTile";
import { OperatorStrip } from "./OperatorStrip";
import { PromptsTile } from "./PromptsTile";
import { VoiceTile } from "./VoiceTile";

// Full Controls surface. Two completely different layouts at the
// `useIsMobile` breakpoint:
//
//   Desktop  — LEFT inset floating panel with the 7-tab DrawerTabs body.
//              Edge handle on the right; HeroMacros bay carries the open
//              toggle. Both fire dd:toggle-drawer.
//
//   Mobile   — No slide-up drawer. The LiteControls strip is rendered
//              directly as fixed bottom chrome (no install-sheet wrapper).
//              An "All controls" button on that strip opens the
//              <MobileFullSheet/> portal, which is the full-screen 7-tab
//              equivalent. Progressive disclosure: L0 mini strip → L1
//              full sheet.

interface Props {
  /** Slot for the "Saved" tab body. Mounted by demon-public-demo to
   *  inject its <SessionsTile /> (which depends on auth + /api/sessions
   *  and therefore can't live in DEMON's standalone bundle). When
   *  omitted, the tab shows a small "unavailable" placeholder. */
  savedTab?: ReactNode;
  /** Mobile-only: pulse a dot on the LiteControls "All controls" button
   *  when there are unsaved session tweaks. Wired by demon-public-demo
   *  from its useSavedSessions().dirty signal. */
  unsavedDot?: boolean;
}

export function AdvancedDrawer({ savedTab, unsavedDot }: Props = {}) {
  const [open, setOpen] = useState(false);
  const [allOpen, setAllOpen] = useState(false);
  const [activeTab, setActiveTab] = useDrawerTab("core");
  const isMobile = useIsMobile();
  const status = useSessionStore((s) => s.status);
  const showKbdHints = usePerformanceStore((s) => s.showKbdHints);
  const started = status !== "idle";

  // useKeyboardShortcuts dispatches this on Esc / `o`. HeroMacros'
  // toggle button also dispatches the same event. Desktop only — on
  // mobile there's no slide-up drawer, so the event is a no-op.
  useEffect(() => {
    const handler = () => {
      if (!started || isMobile) return;
      setOpen((v) => !v);
    };
    document.addEventListener("dd:toggle-drawer", handler);
    return () => document.removeEventListener("dd:toggle-drawer", handler);
  }, [started, isMobile]);

  // Force-close on any transition back to idle (session reset).
  useEffect(() => {
    if (!started) {
      setOpen(false);
      setAllOpen(false);
    }
  }, [started]);

  // Auto-close when the SCHEDULE CURVES overlay opens. Mutually
  // exclusive working modes; stacking them just shrinks both.
  const overlayOpen = useCurveStore((s) => s.overlayOpen);
  useEffect(() => {
    if (overlayOpen) {
      setOpen(false);
      setAllOpen(false);
    }
  }, [overlayOpen]);

  // Mirror desktop open state to body.drawer-open so other chrome
  // (graph stage shrink, hero bay style adjustments) can react.
  useEffect(() => {
    document.body.classList.toggle("drawer-open", open);
    return () => {
      document.body.classList.remove("drawer-open");
    };
  }, [open]);

  // ─── Mobile branch ───────────────────────────────────────────────
  // Always-visible LiteControls strip + on-demand full sheet.
  if (isMobile) {
    return (
      <>
        {started && (
          <LiteControls
            onOpenAllControls={() => setAllOpen(true)}
            unsavedDot={unsavedDot}
          />
        )}
        <MobileFullSheet
          open={allOpen}
          onClose={() => setAllOpen(false)}
          savedTab={savedTab}
        />
      </>
    );
  }

  // ─── Desktop branch ──────────────────────────────────────────────
  return (
    <aside
      id="install-sheet"
      className={`install-sheet${open ? " open" : ""}`}
      aria-hidden={!open}
    >
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
      <div className="install-sheet-body">
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
        <DrawerHelpBar />
      </div>
    </aside>
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
    case "styles":
      return (
        <div className="styles-tab">
          <PromptsTile />
          <LibraryTile />
        </div>
      );
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
