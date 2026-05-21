"use client";

import { useState, type ReactNode } from "react";

// Collapsible wrapper for a drawer tile. Used in the Styles tab so the
// Tags and LoRA Library panels each sit under their own accordion —
// closed by default, expand to manage what's inside. The header
// replaces the wrapped tile's own `.mixer-tile-label` (hidden via CSS
// while inside `.styles-accordion-body`).

export function CollapsibleTile({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`styles-accordion${open ? " open" : ""}`}>
      <button
        type="button"
        className="styles-accordion-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="styles-accordion-caret" aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
        <span className="styles-accordion-title">{title}</span>
      </button>
      {open && <div className="styles-accordion-body">{children}</div>}
    </div>
  );
}
