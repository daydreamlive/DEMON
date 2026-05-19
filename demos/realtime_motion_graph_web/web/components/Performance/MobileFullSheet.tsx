"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

import { CoreTile } from "./CoreTile";
import { LibraryTile } from "./LibraryTile";
import { ModTile } from "./ModTile";
import { OperatorStrip } from "./OperatorStrip";
import { PromptsTile } from "./PromptsTile";
import { VoiceTile } from "./VoiceTile";

type Tab = "core" | "mod" | "voice" | "styles" | "saved" | "config";

interface Props {
  open: boolean;
  onClose: () => void;
  /** Slot for the Saved tab body, passed through from the host (the
   *  demo passes <SessionsTile/>). Mirrors AdvancedDrawer's savedTab
   *  prop so the desktop + mobile surfaces share the same component. */
  savedTab?: ReactNode;
}

// Mirrors the desktop DrawerTabs IA: CORE / STYLES (prompts + LoRAs
// together) / MOD / CHANNELS (key=voice) / SAVED / CONFIG.
const TABS: { id: Tab; label: string }[] = [
  { id: "core", label: "Core" },
  { id: "styles", label: "Styles" },
  { id: "mod", label: "Mod" },
  { id: "voice", label: "Channels" },
  { id: "saved", label: "Saved" },
  { id: "config", label: "Config" },
];

// Full-screen tabbed sheet that surfaces the desktop mixer on mobile when
// the user taps "All controls". All four sections live in a horizontal
// scroll-snap track so the user can swipe between them; the tab pills at
// the bottom both reflect and drive the active section. IntersectionObserver
// is the single source of truth — taps scroll into view, the observer
// updates `tab` from whatever's most visible. That way swipe and tap stay
// in sync without setState fighting the scroller.
export function MobileFullSheet({ open, onClose, savedTab }: Props) {
  const [tab, setTab] = useState<Tab>("core");
  const [mounted, setMounted] = useState(false);
  const trackRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Watch which section is currently in view and sync the active tab.
  useEffect(() => {
    if (!open) return;
    const root = trackRef.current;
    if (!root) return;

    const obs = new IntersectionObserver(
      (entries) => {
        let best: IntersectionObserverEntry | null = null;
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          if (!best || e.intersectionRatio > best.intersectionRatio) best = e;
        }
        if (!best) return;
        const id = (best.target as HTMLElement).dataset.section as Tab | undefined;
        if (id) setTab(id);
      },
      { root, threshold: [0.5, 0.75, 1] },
    );
    for (const t of TABS) {
      const el = root.querySelector<HTMLElement>(`[data-section="${t.id}"]`);
      if (el) obs.observe(el);
    }
    return () => obs.disconnect();
  }, [open]);

  function gotoTab(id: Tab) {
    const root = trackRef.current;
    if (!root) return;
    const el = root.querySelector<HTMLElement>(`[data-section="${id}"]`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", inline: "start", block: "nearest" });
  }

  if (!mounted || !open) return null;

  return createPortal(
    <div className="mobile-sheet" role="dialog" aria-modal="true">
      <div className="mobile-sheet-accent" aria-hidden="true" />

      <header className="mobile-sheet-header">
        <button
          type="button"
          className="mobile-sheet-back"
          onClick={onClose}
          aria-label="Back"
        >
          <span aria-hidden="true">←</span>
        </button>
        <h2 className="mobile-sheet-title">All Controls</h2>
        <span className="mobile-sheet-spacer" aria-hidden="true" />
      </header>

      <div ref={trackRef} className="mobile-sheet-track">
        <section data-section="core" className="mobile-sheet-section">
          <CoreTile />
        </section>
        <section data-section="mod" className="mobile-sheet-section">
          <ModTile />
        </section>
        <section data-section="voice" className="mobile-sheet-section">
          <VoiceTile />
        </section>
        <section data-section="styles" className="mobile-sheet-section">
          <div className="styles-tab">
            <PromptsTile />
            <LibraryTile />
          </div>
        </section>
        <section data-section="saved" className="mobile-sheet-section">
          {savedTab ?? (
            <div className="install-sheet-saved-placeholder">
              Saved sessions are only available in the hosted app.
            </div>
          )}
        </section>
        <section data-section="config" className="mobile-sheet-section">
          <OperatorStrip />
        </section>
      </div>

      <nav className="mobile-sheet-tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            className={`mobile-sheet-tab${tab === t.id ? " mobile-sheet-tab--active" : ""}`}
            onClick={() => gotoTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </div>,
    document.body,
  );
}
