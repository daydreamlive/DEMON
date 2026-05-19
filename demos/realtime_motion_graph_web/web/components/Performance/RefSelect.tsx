"use client";

import { useEffect, useRef, useState } from "react";

// Custom dropdown used by TimbreRefControl and StructureRefControl. Replaces
// the native <select>, whose closed width was tied to the longest <option>
// and ballooned the MainTile sideways. Here the closed affordance is a
// <button> whose only inline content is the current selection (truncated
// with ellipsis), and the popup is position: absolute so option content
// cannot reach the tile's intrinsic-size calculation.

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
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

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
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
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
          onClick={() => setOpen((v) => !v)}
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
        {open && (
          <div ref={menuRef} className="ref-control-menu" role="listbox">
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
          </div>
        )}
      </div>
    </div>
  );
}
