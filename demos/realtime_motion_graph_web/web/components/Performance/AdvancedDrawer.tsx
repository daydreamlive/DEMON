"use client";

import { useEffect, useState, type ReactNode } from "react";

import { useCapability } from "@/hooks/useCapability";
import { useIsMobile } from "@/hooks/useIsMobile";
import { useCurveStore } from "@/store/useCurveStore";
import { useMidiStore } from "@/store/useMidiStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import { CollapsibleTile } from "./CollapsibleTile";
import { CoreTile } from "./CoreTile";
import { DrawerHelpBar } from "./DrawerHelpBar";
import { DynamicKnobPanel } from "./DynamicKnobPanel";
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
  // Spread is an open-only mode: closing the drawer always resets it
  // (see effect below). No persistence — the next open starts tabbed.
  const [spread, setSpread] = useState(false);
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

  // Double-tap Esc (useKeyboardShortcuts) is an expand toggle: closed →
  // open straight into the expanded layout; already open (any mode) →
  // close. Setting open+spread in one batch keeps the
  // !open→setSpread(false) reset effect below from firing, since `open`
  // is already true by the time it runs; closing lets that effect reset
  // spread for the next open.
  useEffect(() => {
    const handler = () => {
      if (!started || isMobile) return;
      if (open) {
        setOpen(false);
      } else {
        setOpen(true);
        setSpread(true);
      }
    };
    document.addEventListener("dd:expand-toggle-drawer", handler);
    return () =>
      document.removeEventListener("dd:expand-toggle-drawer", handler);
  }, [started, isMobile, open]);

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

  // Spread mode is an open-drawer-only mode — closing the drawer
  // turns it off, so the next time the drawer opens it comes back in
  // the default tabbed view. localStorage persistence on the toggle
  // button itself is separate (controlled by setSpread); this effect
  // only clears the live state when the drawer is dismissed.
  useEffect(() => {
    if (!open) setSpread(false);
  }, [open]);

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
      className={`install-sheet${open ? " open" : ""}${spread ? " install-sheet--spread" : ""}`}
      aria-hidden={!open}
    >
      <button
        type="button"
        className="install-sheet-edge-handle"
        onClick={() => started && setOpen((v) => !v)}
        disabled={!started}
        aria-label={open ? "Close Full Controls" : "Open Full Controls"}
        aria-expanded={open}
        data-midi-learn="drawer_toggle"
        onContextMenu={(e) => {
          if (!started) return;
          e.preventDefault();
          useMidiStore.getState().openMenu({
            x: e.clientX,
            y: e.clientY,
            kind: "note",
            target: "drawer_toggle",
            el: e.currentTarget,
          });
        }}
      >
        <span className="install-sheet-edge-handle-caret" aria-hidden="true">
          {open ? "◂" : "▸"}
        </span>
      </button>
      <button
        type="button"
        className="install-sheet-spread-toggle"
        onClick={() => {
          if (!started) return;
          // Closed: jump straight into the expanded (spread) layout so
          // the operator can open directly to all-controls. Open: just
          // flip between tabbed and spread.
          if (!open) {
            setOpen(true);
            setSpread(true);
          } else {
            setSpread((v) => !v);
          }
        }}
        disabled={!started}
        aria-label={
          !open
            ? "Open expanded controls"
            : spread
              ? "Tabbed view"
              : "Spread view (all controls)"
        }
        aria-pressed={open && spread}
        title={!open ? "Expanded controls" : spread ? "Tabbed view" : "Spread view"}
        data-midi-learn="drawer_expand_toggle"
        onContextMenu={(e) => {
          if (!started) return;
          e.preventDefault();
          useMidiStore.getState().openMenu({
            x: e.clientX,
            y: e.clientY,
            kind: "note",
            target: "drawer_expand_toggle",
            el: e.currentTarget,
          });
        }}
      >
        <svg
          viewBox="0 0 16 16"
          width={11}
          height={11}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.4}
          aria-hidden="true"
        >
          <rect x="2" y="2" width="5" height="5" rx="0.6" />
          <rect x="9" y="2" width="5" height="5" rx="0.6" />
          <rect x="2" y="9" width="5" height="5" rx="0.6" />
          <rect x="9" y="9" width="5" height="5" rx="0.6" />
        </svg>
      </button>
      <div className="install-sheet-body">
        {!spread && (
          <div className="install-sheet-topbar">
            <DrawerTabs active={activeTab} onChange={setActiveTab} />
          </div>
        )}
        <div
          className={`mixer-rack ${spread ? "mixer-rack--spread" : "mixer-rack--tabbed"}${!showKbdHints ? " mixer-rack--no-kbd-hints" : ""}`}
          id="mixer-tiles"
          data-active-tab={spread ? "all" : activeTab}
        >
          {spread ? renderAllSections(savedTab) : renderTabBody(activeTab, savedTab)}
        </div>
        <DrawerHelpBar spread={spread} />
      </div>
    </aside>
  );
}

// Spread view — render every tab body in sequence with a small section
// header above each. Used when the user toggles spread mode from the
// drawer handle. The rack becomes a CSS grid (see globals.css) so the
// sections lay out as auto-fitting columns and most controls are
// visible without paging.
// Saved sessions are omitted from spread mode on purpose — it's a
// history surface, not a control surface, and dropping it lets the
// five real control sections fit in a 3×2 grid with Voice spanning
// two columns (14 channel faders don't fit in a 1/3 column).
const SPREAD_SECTIONS: Array<{ id: DrawerTab; label: string }> = [
  { id: "core", label: "Core" },
  { id: "styles", label: "Styles" },
  { id: "mod", label: "Mod" },
  { id: "voice", label: "Experimental" },
  { id: "config", label: "Config" },
];

function renderAllSections(savedTab?: ReactNode) {
  return SPREAD_SECTIONS.map((s) => (
    <section key={s.id} className="spread-section" data-section={s.id}>
      <h3 className="spread-section-label">{s.label}</h3>
      {renderTabBody(s.id, savedTab, true)}
    </section>
  ));
}

// Tab body switch — kept as a plain function (not a component) because
// every tile already runs its own hooks/subscriptions, and a wrapping
// React component would just add another re-render layer with no
// upside.
function renderTabBody(tab: DrawerTab, savedTab?: ReactNode, spread = false) {
  switch (tab) {
    case "core":
      return <CoreTile />;
    case "mod":
      return <ModTile />;
    case "voice":
      return <VoiceTile />;
    case "auto":
      return <DynamicKnobPanel />;
    case "styles":
      return <StylesTab spread={spread} />;
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

// Styles tab body. A real component (unlike renderTabBody) so it can
// gate the LoRA library on the backend capability mask: a backend
// family without `lora` (or a lora-disabled session) doesn't get a
// dead library panel. Tabbed view: each panel sits under its own
// collapsed accordion so the operator expands only what they're
// working on. Spread view is a show-everything surface, so the tiles
// render bare.
function StylesTab({ spread }: { spread: boolean }) {
  const canLora = useCapability("lora");
  return (
    <div className="styles-tab">
      {spread ? (
        <>
          <PromptsTile />
          {canLora && <LibraryTile />}
        </>
      ) : (
        <>
          <CollapsibleTile title="Tags">
            <PromptsTile />
          </CollapsibleTile>
          {canLora && (
            <CollapsibleTile title="LoRA Library">
              <LibraryTile />
            </CollapsibleTile>
          )}
        </>
      )}
    </div>
  );
}
