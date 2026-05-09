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
}

export function RefSelect({
  label,
  value,
  pinned,
  groups,
  onSelect,
  disabled,
  ariaLabel,
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
    <div className="ref-control">
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
