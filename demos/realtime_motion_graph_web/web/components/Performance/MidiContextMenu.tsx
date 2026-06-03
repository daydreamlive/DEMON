"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
  MIDI_MODE_OPTIONS,
  midiTargetLabel,
} from "@/engine/midi/targetRegistry";
import { useMidiStore } from "@/store/useMidiStore";

// Right-click menu for MIDI mapping, opened by the document-level handler
// in useMidi for any mappable control (sliders, knobs, hero faders, stem
// panners, the blend rail, enum dropdowns, and action buttons). It's the
// one place mapping + binding mode live, anchored to the control the user
// just right-clicked — there is no separate config screen to hunt for.
//
// Portaled to document.body (escapes drawer overflow / z-index) and
// position-clamped to the viewport.

/** How a bound CC key reads in the menu. Plain CC numbers show as
 *  "CC 74"; a pitch-bend axis is keyed "pb<channel>" and shows as
 *  "Pitch bend". */
function controllerLabel(key: string): string {
  if (!key.startsWith("pb")) return `CC ${key}`;
  const channel = Number(key.slice(2));
  return Number.isFinite(channel) ? `Pitch bend ch ${channel + 1}` : "Pitch bend";
}

interface Item {
  label: string;
  onClick?: () => void;
  active?: boolean;
  header?: boolean;
  divider?: boolean;
}

function labelFromElement(el: HTMLElement | null, target: string): string {
  if (!el) return midiTargetLabel(target);
  const explicit = el.getAttribute("data-dd-tooltip-title")?.trim();
  if (explicit) return explicit;
  // A <select> (enum target) has no inner label element; its textContent is
  // the whole concatenated option list. Use the curated registry label
  // instead ("Musical key", "Time signature", …) rather than that blob.
  if (el.tagName === "SELECT") return midiTargetLabel(target);
  const sliderLabel = el.querySelector<HTMLElement>(".slider-label")?.textContent?.trim();
  if (sliderLabel) return sliderLabel;
  const stemLabel = el.querySelector<HTMLElement>(".hero-stem-panner-label")?.textContent?.trim();
  if (stemLabel) return `${stemLabel} stem volume`;
  const styleLabel = el.querySelector<HTMLElement>(".hero-style-fader-label")?.textContent?.trim();
  if (styleLabel) return styleLabel;
  const knobLabel = el.querySelector<HTMLElement>(".knob-label")?.textContent?.trim();
  if (knobLabel) return knobLabel;
  const aria = el.getAttribute("aria-label")?.trim();
  if (aria) return aria;
  const childAria = el.querySelector<HTMLElement>("[aria-label]")?.getAttribute("aria-label")?.trim();
  if (childAria) return childAria;
  const text = el.textContent?.replace(/\s+/g, " ").trim();
  return text || midiTargetLabel(target);
}
export function MidiContextMenu() {
  const menu = useMidiStore((s) => s.menu);
  const map = useMidiStore((s) => s.map);
  const closeMenu = useMidiStore((s) => s.closeMenu);
  const startLearn = useMidiStore((s) => s.startLearn);
  const clearBinding = useMidiStore((s) => s.clearBinding);
  const setCcMode = useMidiStore((s) => s.setCcMode);

  const menuRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  useLayoutEffect(() => {
    if (!menu) return;
    const el = menuRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pad = 8;
    let nx = menu.x;
    let ny = menu.y;
    if (nx + rect.width > window.innerWidth - pad) {
      nx = Math.max(pad, window.innerWidth - rect.width - pad);
    }
    if (ny + rect.height > window.innerHeight - pad) {
      ny = Math.max(pad, window.innerHeight - rect.height - pad);
    }
    setPos({ x: nx, y: ny });
  }, [menu]);

  useEffect(() => {
    if (!menu) return;
    const onPointerDown = (e: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        closeMenu();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeMenu();
    };
    const onScroll = () => closeMenu();
    const t = window.setTimeout(() => {
      document.addEventListener("pointerdown", onPointerDown);
      document.addEventListener("keydown", onKey);
      window.addEventListener("scroll", onScroll, true);
    }, 0);
    return () => {
      window.clearTimeout(t);
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [menu, closeMenu]);

  if (!menu) return null;
  const { kind, target, el, extraItems } = menu;
  const targetLabel = labelFromElement(el, target);

  const items: Item[] = [{ header: true, label: targetLabel }];

  if (kind === "cc") {
    const boundCc =
      Object.keys(map.cc).find((k) => map.cc[k] === target) ?? null;
    if (boundCc) {
      const current = map.ccMode?.[boundCc] ?? "absolute";
      items.push({
        header: true,
        label: `Bound to ${controllerLabel(boundCc)} - control type:`,
      });
      for (const m of MIDI_MODE_OPTIONS) {
        items.push({
          label: m.label,
          active: current === m.value,
          onClick: () => setCcMode(boundCc, m.value),
        });
      }
      items.push({ divider: true, label: "divider" });
      items.push({
        label: "Re-learn as same type",
        onClick: () => startLearn("cc", target, el, current),
      });
      items.push({
        label: "Clear mapping",
        onClick: () => clearBinding("cc", target),
      });
    } else {
      items.push({ header: true, label: "MIDI Learn - choose the physical control" });
      for (const m of MIDI_MODE_OPTIONS) {
        items.push({
          label: m.label,
          onClick: () => startLearn("cc", target, el, m.value),
        });
      }
    }
  } else if (kind === "enum") {
    const bound =
      Object.keys(map.ccEnum ?? {}).some((k) => map.ccEnum![k] === target) ||
      Object.keys(map.noteEnum ?? {}).some((k) => map.noteEnum![k] === target);
    items.push({
      label: bound
        ? "Re-learn - twist a knob/fader or press a pad"
        : "MIDI Learn - twist a knob/fader or press a pad",
      onClick: () => startLearn("enum", target, el),
    });
    if (bound) {
      items.push({ divider: true, label: "divider" });
      items.push({
        label: "Clear mapping",
        onClick: () => clearBinding("enum", target),
      });
    }
  } else {
    const bound =
      Object.keys(map.notes).some((k) => map.notes[k] === target) ||
      Object.keys(map.ccActions ?? {}).some((k) => map.ccActions![k] === target);
    items.push({
      label: bound ? "Re-learn - press a pad/button" : "MIDI Learn - press a pad/button",
      onClick: () => startLearn("note", target, el),
    });
    if (bound) {
      items.push({ divider: true, label: "divider" });
      items.push({
        label: "Clear mapping",
        onClick: () => clearBinding("note", target),
      });
    }
  }

  for (const extra of extraItems ?? []) {
    if (extra.dividerBefore) items.push({ divider: true, label: `divider:${extra.label}` });
    items.push({ label: extra.label, onClick: extra.onClick });
  }

  return createPortal(
    <div
      ref={menuRef}
      className="lora-context-menu"
      style={{ left: pos.x, top: pos.y }}
      role="menu"
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((item, i) =>
        item.divider ? (
          <div
            key={`d${i}`}
            className="lora-context-menu-divider"
            role="separator"
          />
        ) : item.header ? (
          <div
            key={`h${i}`}
            className="lora-context-menu-item"
            style={{
              opacity: 0.6,
              fontSize: "0.75em",
              textTransform: "uppercase",
              letterSpacing: "0.04em",
              pointerEvents: "none",
            }}
          >
            {item.label}
          </div>
        ) : (
          <button
            key={item.label}
            type="button"
            className="lora-context-menu-item"
            role="menuitem"
            onClick={() => {
              item.onClick?.();
              closeMenu();
            }}
          >
            {item.active ? "✓ " : " "}
            {item.label}
          </button>
        ),
      )}
    </div>,
    document.body,
  );
}
