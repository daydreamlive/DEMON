"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";

// Custom dropdown used by TimbreRefControl and StructureRefControl. Replaces
// the native <select>, whose closed width was tied to the longest <option>
// and ballooned the MainTile sideways. Here the closed affordance is a
// <button> whose only inline content is the current selection (truncated
// with ellipsis).
//
// The open menu is portaled to document.body and positioned `fixed`. The
// pickers live inside several nested scroll containers (the tabbed drawer's
// `.mixer-rack` overflow-y:auto, the spread-mode `.spread-section` cells, the
// mobile sheet track); any in-flow `position:absolute` menu gets clipped by
// whichever one it overflows. A body-level `fixed` element has no overflow or
// transformed ancestor between it and the viewport, so the list is fully
// visible no matter which direction it opens. Placement is computed once from
// the button's getBoundingClientRect at open time — first paint lands in the
// final spot (no measure-then-reposition flash) — and only re-runs on
// scroll/resize so it tracks the button instead of jumping.

export interface RefSelectOption {
  value: string;
  label: string;
}

export interface RefSelectGroup {
  label: string;
  options: RefSelectOption[];
}

interface Props {
  label: string;
  value: string;
  pinned: RefSelectOption[];
  groups: RefSelectGroup[];
  onSelect: (value: string) => void;
  disabled?: boolean;
  ariaLabel: string;
  /** Optional sibling action button rendered next to the dropdown. Used
   *  for the inline upload affordance so the modal that follows doesn't
   *  occlude the dropdown list the user was just browsing. */
  onUpload?: () => void;
  /** Tooltip / aria-label for the upload button, kind-specific so screen
   *  readers and the one-shot tooltip both get useful copy. */
  uploadLabel?: string;
  /** Long-form description rendered into the panel help bar on hover.
   *  Should explain what this picker controls (audio source, timbre
   *  reference, structure reference, etc.). */
  tooltip?: string;
}

interface MenuPos {
  left: number;
  top: number | "auto";
  bottom: number | "auto";
  minWidth: number;
  maxHeight: number;
  fontSize: string;
}

// Compute the fixed-position box from the trigger button's viewport rect.
// Uses only the rect + viewport (never the menu's own height), so it can run
// synchronously at open time with no second measuring pass. Opens downward by
// default and only flips up when there's too little room below AND more room
// above; max-height clamps the menu to the chosen side with internal scroll.
function computeMenuPos(button: HTMLElement): MenuPos {
  const GAP = 4;
  const MARGIN = 8;
  const rect = button.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  // Mirrors the CSS fallback ceiling (`max-height: min(50vh, 320px)`).
  const cap = Math.min(vh * 0.5, 320);
  const spaceBelow = vh - rect.bottom - GAP - MARGIN;
  const spaceAbove = rect.top - GAP - MARGIN;
  const openUp = spaceBelow < cap && spaceAbove > spaceBelow;
  const side = openUp ? spaceAbove : spaceBelow;
  const maxHeight = Math.max(120, Math.min(cap, side));
  // Keep the menu on screen if the trigger sits near the right edge.
  const maxWidthPx = Math.min(360, vw * 0.9);
  const left = Math.max(MARGIN, Math.min(rect.left, vw - MARGIN - maxWidthPx));
  // The menu used to inherit `.ref-control`'s 0.55em font-size through the
  // DOM; portaling to <body> breaks that cascade, so carry the trigger's
  // resolved size over explicitly to keep the list text matching the
  // closed control (and still scaling with the drawer's em cascade).
  const fontSize = window.getComputedStyle(button).fontSize;
  return {
    left,
    minWidth: rect.width,
    maxHeight,
    fontSize,
    ...(openUp
      ? { top: "auto", bottom: vh - rect.top + GAP }
      : { top: rect.bottom + GAP, bottom: "auto" }),
  };
}

function UploadIcon({ size = 12 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M8 10V2" />
      <path d="M4.5 5.5L8 2l3.5 3.5" />
      <path d="M2.5 10v3a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3" />
    </svg>
  );
}

export function RefSelect({
  label,
  value,
  pinned,
  groups,
  onSelect,
  disabled,
  ariaLabel,
  onUpload,
  uploadLabel,
  tooltip,
}: Props) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<MenuPos | null>(null);
  // Portals need document.body, which is absent during SSR — only render
  // the portal once we've mounted on the client.
  const [mounted, setMounted] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => setMounted(true), []);

  function openMenu() {
    if (buttonRef.current) setPos(computeMenuPos(buttonRef.current));
    setOpen(true);
  }

  useEffect(() => {
    if (!open) return;
    function onPointer(e: PointerEvent) {
      const t = e.target as Node | null;
      if (!t) return;
      if (buttonRef.current?.contains(t)) return;
      if (menuRef.current?.contains(t)) return;
      setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    // The menu is portaled out of the scroll containers, so it no longer
    // moves with them — re-anchor to the button on any scroll (capture, to
    // catch ancestor scrolling) or resize so it stays glued instead of
    // drifting away from its trigger.
    function reanchor() {
      if (buttonRef.current) setPos(computeMenuPos(buttonRef.current));
    }
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", reanchor, true);
    window.addEventListener("resize", reanchor);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", reanchor, true);
      window.removeEventListener("resize", reanchor);
    };
  }, [open]);

  useEffect(() => {
    if (disabled && open) setOpen(false);
  }, [disabled, open]);

  const allOptions = [...pinned, ...groups.flatMap((g) => g.options)];
  const current = allOptions.find((o) => o.value === value);
  const displayed = current?.label ?? value;

  function pick(v: string) {
    onSelect(v);
    setOpen(false);
  }

  const menuStyle: CSSProperties = pos
    ? {
        left: pos.left,
        top: pos.top,
        bottom: pos.bottom,
        minWidth: pos.minWidth,
        maxHeight: pos.maxHeight,
        fontSize: pos.fontSize,
      }
    : { visibility: "hidden" };

  return (
    <div
      className="ref-control"
      data-dd-tooltip={tooltip || undefined}
      data-dd-tooltip-wide={tooltip ? "" : undefined}
      data-dd-tooltip-title={label}
    >
      <span className="ref-control-label">{label}</span>
      <div className="ref-control-anchor">
        <button
          ref={buttonRef}
          type="button"
          className="ref-control-button"
          onClick={() => (open ? setOpen(false) : openMenu())}
          disabled={disabled}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-label={ariaLabel}
          title={displayed}
        >
          <span className="ref-control-button-text">{displayed}</span>
          <span className="ref-control-button-caret" aria-hidden="true" />
        </button>
        {onUpload && (
          <button
            type="button"
            className="ref-control-upload"
            onClick={onUpload}
            disabled={disabled}
            aria-label={uploadLabel ?? "Upload"}
            title={uploadLabel ?? "Upload"}
          >
            <UploadIcon />
          </button>
        )}
        {open &&
          mounted &&
          createPortal(
            <div
              ref={menuRef}
              className="ref-control-menu"
              role="listbox"
              style={menuStyle}
            >
              {pinned.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  role="option"
                  aria-selected={o.value === value}
                  className={`ref-control-option${
                    o.value === value ? " ref-control-option--current" : ""
                  }`}
                  onClick={() => pick(o.value)}
                  title={o.label}
                >
                  {o.label}
                </button>
              ))}
              {groups.map(
                (g) =>
                  g.options.length > 0 && (
                    <div key={g.label} className="ref-control-group">
                      <div className="ref-control-group-label">{g.label}</div>
                      {g.options.map((o) => (
                        <button
                          key={o.value}
                          type="button"
                          role="option"
                          aria-selected={o.value === value}
                          className={`ref-control-option${
                            o.value === value
                              ? " ref-control-option--current"
                              : ""
                          }`}
                          onClick={() => pick(o.value)}
                          title={o.label}
                        >
                          {o.label}
                        </button>
                      ))}
                    </div>
                  ),
              )}
            </div>,
            document.body,
          )}
      </div>
    </div>
  );
}
