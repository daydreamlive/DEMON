"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { loraStrengthDispatcher } from "@/engine/lora/dispatcher";
import { listLoras } from "@/engine/lora/listLoras";
import { getConfig, useConfig } from "@/lib/config";
import { displayLoraName } from "@/lib/loraLabels";
import { LOCAL_MODE } from "@/lib/runtime";
import { isLoraCompatibleWithScale, useLoraStore } from "@/store/useLoraStore";
import { useMidiStore } from "@/store/useMidiStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { LORA_SLIDER_MAX } from "@/types/engine";
import type { LoraCatalogEntry, LoraMetadata } from "@/types/protocol";

// LoRA library tile. Each row: pill-style enable switch, name, full-width
// strength slider with a colored fill. The whole tile also exposes:
//
//  - A search bar at the top filtering on name + description + tags +
//    primary_genre. Empty query shows everything.
//  - A right-click anywhere on the row opens a portaled context menu
//    (LoraContextMenu) with two items:
//       MIDI learn       → arms `useMidiStore.startLearn("cc", "lora_str_<id>", row)`
//       Copy trigger     → clipboard write + transient tooltip flash
//    The row stops the contextmenu's native propagation so the
//    document-level fallback in `useMidi.ts` doesn't *also* arm learn
//    (that fallback's `.lora-row` branch has been removed in tandem;
//    this menu is now the sole entry point for both actions on a row).
//    The copy confirmation rides the same `data-dd-tooltip` chrome as
//    every other tooltip in the drawer: after a successful copy, the
//    row's tooltip text is swapped to "Copied 'word'" and
//    `data-dd-tooltip-show` is forced true for 1.5s.
//  - Hover on a row surfaces the same `data-dd-tooltip-wide` chrome
//    with the description, classification chips, and recommended
//    inference params. Bare LoRAs fall back to id-as-name display.
//
//  Tooltip placement note: the `[data-dd-tooltip]::after` pseudo is
//  `position: absolute`, and `.lora-row` (and every drawer-body
//  ancestor between the row and `.install-sheet`) is unpositioned, so
//  the tooltip walks up to the drawer (which is `position: fixed`).
//  Combined with the default `bottom: calc(100% + 8px); left: 50%`,
//  every tooltip in the drawer — slider labels, OperatorStrip
//  buttons, prompts, library rows — lands in the same screen spot
//  8px above the drawer top edge, viewport-centered. Do NOT add
//  `position: relative` to `.lora-row` or any intermediate ancestor;
//  it kicks the tooltip out of that shared surface.
//  - Toggling a LoRA on, when `engine.auto_prepend_lora_triggers` is
//    true (default), prepends the trigger word to promptA and promptB
//    so the encoder sees what the operator sees. Toggling off removes
//    the trigger wherever it appears as a standalone comma-delimited
//    token — covers the case where multiple LoRAs were stacked and
//    the trigger is no longer at the head of the prompt. Substrings
//    inside larger user-typed phrases are left alone.
//  - LoRAs whose `base_model_scale` doesn't match the active
//    checkpoint scale (2B vs 5B) are hidden by default; an inline
//    footer reports the count and offers a one-click override.
//    `engine.show_incompatible_loras = true` flips the default.

// ── Visible trigger-prepend helpers ─────────────────────────────────────

function _containsTrigger(prompt: string, trigger: string): boolean {
  return prompt.toLowerCase().includes(trigger.toLowerCase());
}

function _prependTrigger(prompt: string, trigger: string): string {
  const trimmed = prompt.trim();
  if (trimmed.length === 0) return trigger;
  return `${trigger}, ${prompt}`;
}

/** Remove every occurrence of ``trigger`` from ``prompt`` where it
 *  appears as a standalone comma-delimited token (head, middle, tail,
 *  or sole token). Substrings of larger phrases are left alone — if
 *  the user typed "punk rock" and the trigger is "rock", we don't
 *  mangle their prompt. Returns the rewritten prompt, or null when
 *  the trigger isn't present as a token (so callers can skip the
 *  write). Case-insensitive on the trigger comparison; preserves the
 *  case of surrounding tokens. */
function _stripTrigger(prompt: string, trigger: string): string | null {
  const needle = trigger.trim().toLowerCase();
  if (!needle) return null;
  const tokens = prompt.split(", ");
  const kept = tokens.filter((tok) => tok.trim().toLowerCase() !== needle);
  if (kept.length === tokens.length) return null;
  return kept.join(", ");
}

// Both prompts get the same treatment when a LoRA is toggled. Iterate
// over [current, setter] pairs so the per-side logic lives once.
function _promptSides(): ReadonlyArray<readonly [string, (v: string) => void]> {
  const perf = usePerformanceStore.getState();
  return [
    [perf.promptA, perf.setPromptA],
    [perf.promptB, perf.setPromptB],
  ] as const;
}

function prependTriggerToPrompts(trigger: string): void {
  for (const [cur, setter] of _promptSides()) {
    if (!_containsTrigger(cur, trigger)) {
      setter(_prependTrigger(cur, trigger));
    }
  }
}

function removeTriggerFromPrompts(trigger: string): void {
  for (const [cur, setter] of _promptSides()) {
    const next = _stripTrigger(cur, trigger);
    if (next !== null) setter(next);
  }
}

// ── Search ──────────────────────────────────────────────────────────────

function matchesQuery(entry: LoraCatalogEntry, q: string): boolean {
  if (!q) return true;
  const needle = q.trim().toLowerCase();
  if (!needle) return true;
  const md = entry.metadata;
  const parts: (string | null | undefined)[] = [
    entry.id,
    entry.name,
    md?.name,
    md?.description,
    md?.primary_trigger_word,
    md?.primary_genre,
    ...(md?.tags ?? []),
    ...(md?.moods ?? []),
    ...(md?.secondary_genres ?? []),
    ...(md?.trigger_words ?? []),
  ];
  return parts
    .filter((s): s is string => typeof s === "string" && s.length > 0)
    .some((s) => s.toLowerCase().includes(needle));
}

// ── Tooltip text ────────────────────────────────────────────────────────

function buildTooltipText(md: LoraMetadata | undefined): string | undefined {
  if (!md || !md.has_metadata) return undefined;
  const parts: string[] = [];
  if (md.description) {
    parts.push(md.description);
  }
  const classBits: string[] = [];
  if (md.primary_genre) classBits.push(md.primary_genre);
  if (md.tags.length > 0) classBits.push(md.tags.join(", "));
  if (classBits.length > 0) parts.push(classBits.join(" • "));
  const recBits: string[] = [];
  if (md.recommended_strength != null) {
    recBits.push(`strength ${md.recommended_strength.toFixed(2)}`);
  }
  if (md.recommended_steps != null) recBits.push(`${md.recommended_steps} steps`);
  if (md.recommended_shift != null) {
    recBits.push(`shift ${md.recommended_shift.toFixed(2)}`);
  }
  if (md.recommended_guidance != null) {
    recBits.push(`guidance ${md.recommended_guidance.toFixed(2)}`);
  }
  if (recBits.length > 0) parts.push(`Recommended: ${recBits.join(", ")}`);
  if (md.primary_trigger_word) {
    parts.push(`Trigger: "${md.primary_trigger_word}" (right-click for menu)`);
  }
  if (parts.length === 0) return undefined;
  return parts.join(" — ");
}

// ── Context menu ────────────────────────────────────────────────────────

interface ContextMenuItem {
  label: string;
  onClick: () => void;
}

interface ContextMenuProps {
  x: number;
  y: number;
  items: ContextMenuItem[];
  onClose: () => void;
}

// Portaled to document.body so it escapes the drawer + library-list
// overflow clipping and z-index. Click position is clamped to the
// viewport before paint (useLayoutEffect) so the menu never opens
// off-screen near a corner.
function LoraContextMenu({ x, y, items, onClose }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ x, y });

  useLayoutEffect(() => {
    const el = menuRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pad = 8;
    let nx = x;
    let ny = y;
    if (nx + rect.width > window.innerWidth - pad) {
      nx = Math.max(pad, window.innerWidth - rect.width - pad);
    }
    if (ny + rect.height > window.innerHeight - pad) {
      ny = Math.max(pad, window.innerHeight - rect.height - pad);
    }
    if (nx !== x || ny !== y) setPos({ x: nx, y: ny });
  }, [x, y]);

  useEffect(() => {
    const onPointerDown = (e: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    const onScroll = () => onClose();
    // Defer attach by one tick so the right-click that opened the menu
    // (still bubbling at the pointer-event level) doesn't immediately
    // dismiss it via the outside-pointerdown handler.
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
  }, [onClose]);

  return createPortal(
    <div
      ref={menuRef}
      className="lora-context-menu"
      style={{ left: pos.x, top: pos.y }}
      role="menu"
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((item) => (
        <button
          key={item.label}
          type="button"
          className="lora-context-menu-item"
          role="menuitem"
          onClick={() => {
            item.onClick();
            onClose();
          }}
        >
          {item.label}
        </button>
      ))}
    </div>,
    document.body,
  );
}

// ── Row ─────────────────────────────────────────────────────────────────

interface RowProps {
  entry: LoraCatalogEntry;
}

function LoraRow({ entry }: RowProps) {
  const { id } = entry;
  const enabled = useLoraStore((s) => s.enabled.has(id));
  const strength = usePerformanceStore(
    (s) => s.sliderTargets[`lora_str_${id}`],
  );
  const fallbackStrength = useLoraStore((s) => s.strengths[id] ?? 0);
  const value = typeof strength === "number" ? strength : fallbackStrength;
  const enable = useLoraStore((s) => s.enable);
  const disable = useLoraStore((s) => s.disable);

  const rowRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  // Transient tooltip override for click confirmations (e.g. "Copied
  // 'roti-1ndstrl'"). When set, the row's data-dd-tooltip text is
  // swapped to this value and data-dd-tooltip-show is forced on for
  // CONFIRM_MS, surfacing the same shared display surface above the
  // drawer that hover uses. No parallel toast chrome needed.
  const [confirmMsg, setConfirmMsg] = useState<string | null>(null);
  const confirmTimerRef = useRef<number | null>(null);
  // Right-click context menu state — set on row contextmenu, cleared
  // by item click / outside pointerdown / Escape / scroll.
  const [menuPos, setMenuPos] = useState<{ x: number; y: number } | null>(null);
  const closeMenu = useCallback(() => setMenuPos(null), []);

  const md = entry.metadata;
  const displayName = displayLoraName(id, entry.name);
  const baseTooltipText = useMemo(() => buildTooltipText(md), [md]);
  const tooltipText = confirmMsg ?? baseTooltipText;
  const trigger = md?.primary_trigger_word ?? null;

  const flashConfirm = useCallback((text: string) => {
    setConfirmMsg(text);
    if (confirmTimerRef.current !== null) {
      window.clearTimeout(confirmTimerRef.current);
    }
    confirmTimerRef.current = window.setTimeout(() => {
      setConfirmMsg(null);
      confirmTimerRef.current = null;
    }, 1500);
  }, []);

  useEffect(
    () => () => {
      if (confirmTimerRef.current !== null) {
        window.clearTimeout(confirmTimerRef.current);
      }
    },
    [],
  );

  function toggle() {
    const remote = useSessionStore.getState().remote;
    const cfg = getConfig();
    const autoPrepend = cfg.engine.auto_prepend_lora_triggers ?? true;
    if (enabled) {
      disable(id);
      remote?.sendDisableLora(id);
      if (autoPrepend && trigger) removeTriggerFromPrompts(trigger);
    } else {
      enable(id);
      const s = useLoraStore.getState().strengths[id] ?? 0;
      remote?.sendEnableLora(id, s);
      if (autoPrepend && trigger) prependTriggerToPrompts(trigger);
    }
  }

  // Right-click anywhere on the row → open the context menu at click
  // coords. stopPropagation keeps the document-level fallback in
  // useMidi.ts (if anything were to read .lora-row again) from also
  // firing on the same gesture.
  function onRowContextMenu(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    setMenuPos({ x: e.clientX, y: e.clientY });
  }

  const menuItems = useMemo<ContextMenuItem[]>(() => {
    const items: ContextMenuItem[] = [
      {
        label: "MIDI learn",
        onClick: () => {
          useMidiStore
            .getState()
            .startLearn("cc", `lora_str_${id}`, rowRef.current);
        },
      },
    ];
    if (trigger) {
      items.push({
        label: `Copy trigger "${trigger}"`,
        onClick: () => {
          void copyTriggerToClipboard(trigger, flashConfirm);
        },
      });
    }
    return items;
  }, [id, trigger, flashConfirm]);

  const setFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const t = (clientX - rect.left) / rect.width;
      const v = Math.max(0, Math.min(1, t)) * LORA_SLIDER_MAX;
      loraStrengthDispatcher.set(id, v);
    },
    [id],
  );

  useEffect(() => {
    const el = trackRef.current;
    if (!el || !enabled) return;
    let dragging = false;
    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return; // right-click → MIDI learn (document handler)
      dragging = true;
      el.setPointerCapture(e.pointerId);
      setFromClientX(e.clientX);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      setFromClientX(e.clientX);
    };
    const onPointerUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      el.releasePointerCapture(e.pointerId);
    };
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
    };
  }, [enabled, setFromClientX]);

  const pct = Math.max(0, Math.min(1, value / LORA_SLIDER_MAX)) * 100;

  return (
    <>
      <div
        ref={rowRef}
        className={`lora-row${enabled ? " enabled" : ""}`}
        data-param={`lora_str_${id}`}
        data-state={enabled ? "enabled" : "disabled"}
        data-dd-tooltip={tooltipText}
        data-dd-tooltip-wide={tooltipText ? "" : undefined}
        data-dd-tooltip-title={displayName}
        data-dd-tooltip-show={confirmMsg !== null ? "true" : undefined}
        onContextMenu={onRowContextMenu}
      >
        <button
          type="button"
          className="lora-switch"
          role="switch"
          aria-checked={enabled}
          onClick={toggle}
          aria-label={enabled ? `Disable ${displayName}` : `Enable ${displayName}`}
        >
          <span className="lora-switch-thumb" aria-hidden="true" />
        </button>
        <span className="lora-row-name" onClick={toggle}>
          {displayName}
        </span>
        <div className="lora-strength">
          <div className="lora-strength-track" ref={trackRef}>
            <div className="lora-strength-fill" style={{ width: `${pct}%` }} />
            <div
              className="lora-strength-thumb"
              style={{ left: `${pct}%` }}
              aria-hidden="true"
            />
          </div>
          <span className="lora-strength-value">{value.toFixed(2)}</span>
        </div>
      </div>
      {menuPos && (
        <LoraContextMenu
          x={menuPos.x}
          y={menuPos.y}
          items={menuItems}
          onClose={closeMenu}
        />
      )}
    </>
  );
}

async function copyTriggerToClipboard(
  trigger: string,
  onFlash: (text: string) => void,
): Promise<void> {
  try {
    if (
      typeof navigator !== "undefined" &&
      navigator.clipboard &&
      window.isSecureContext
    ) {
      await navigator.clipboard.writeText(trigger);
    } else {
      _legacyCopy(trigger);
    }
    onFlash(`Copied "${trigger}"`);
  } catch {
    onFlash("Copy failed");
  }
}

function _legacyCopy(text: string): void {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(ta);
  }
}

// ── Tile ────────────────────────────────────────────────────────────────

export function LibraryTile() {
  const catalog = useLoraStore((s) => s.catalog);
  const setCatalog = useLoraStore((s) => s.setCatalog);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  const sessionScale = useSessionStore((s) => s.checkpointScale);
  const cfg = useConfig();
  const cfgShowAll = cfg.engine.show_incompatible_loras ?? false;

  const [query, setQuery] = useState("");
  // Per-session override that flips the incompatible-LoRA filter when
  // the operator clicks "show all" in the footer. Doesn't touch the
  // persistent config — closing and reopening the tile resets to the
  // configured default.
  const [showAllOverride, setShowAllOverride] = useState(false);
  const showAll = cfgShowAll || showAllOverride;

  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listLoras().then(setCatalog).catch(() => {});
  }, [setCatalog, sessionWsUrl]);

  // Compatibility filter runs first so the "N hidden" count tracks
  // scale incompatibility specifically, not query misses.
  const compatible = useMemo(
    () =>
      showAll
        ? catalog
        : catalog.filter((entry) =>
            isLoraCompatibleWithScale(entry, sessionScale),
          ),
    [catalog, sessionScale, showAll],
  );
  const hiddenCount = catalog.length - compatible.length;

  const filtered = useMemo(
    () => compatible.filter((entry) => matchesQuery(entry, query)),
    [compatible, query],
  );

  if (catalog.length === 0) {
    return (
      <div className="mixer-tile" data-tile="library">
        <div className="mixer-tile-label">LoRA Library</div>
        <div className="lora-empty">no LoRAs found</div>
      </div>
    );
  }

  return (
    <div className="mixer-tile" data-tile="library">
      <div className="mixer-tile-label">LoRA Library</div>
      <div className="lora-search">
        <input
          type="text"
          className="lora-search-input"
          placeholder="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search LoRA library"
        />
      </div>
      <div className="lora-list">
        {filtered.length === 0 ? (
          <div className="lora-empty">no matches</div>
        ) : (
          filtered.map((entry) => <LoraRow key={entry.id} entry={entry} />)
        )}
      </div>
      {hiddenCount > 0 && (
        <div className="lora-hidden-footer">
          <span
            data-dd-tooltip={
              sessionScale
                ? `LoRAs trained for the other scale are hidden because the active checkpoint is ${sessionScale}. Click "show all" to inspect them.`
                : undefined
            }
            data-dd-tooltip-wide={sessionScale ? "" : undefined}
          >
            {hiddenCount} hidden
            {sessionScale ? ` (not ${sessionScale})` : ""}
          </span>
          <button
            type="button"
            className="lora-hidden-toggle"
            onClick={() => setShowAllOverride((v) => !v)}
          >
            {showAllOverride ? "hide" : "show all"}
          </button>
        </div>
      )}
    </div>
  );
}
