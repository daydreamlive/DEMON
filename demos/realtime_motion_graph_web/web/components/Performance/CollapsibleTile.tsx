"use client";

import { useState, type ReactNode } from "react";

// Collapsible wrapper for a drawer tile. Used in the Styles tab so the
// Tags and LoRA Library panels each sit under their own accordion —
// closed by default, expand to manage what's inside. The header
// replaces the wrapped tile's own `.mixer-tile-label` (hidden via CSS
// while inside `.styles-accordion-body`).
//
// Open/closed state persists per `title` for the page's lifetime via a
// module-level map: switching tabs or closing/reopening the Advanced
// drawer unmounts these tiles, and a plain useState would forget the
// user's expansion every time.

const openByTitle = new Map<string, boolean>();

export function CollapsibleTile({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(
    () => openByTitle.get(title) ?? defaultOpen,
  );
  const toggle = () => {
    setOpen((o) => {
      const next = !o;
      openByTitle.set(title, next);
      return next;
    });
  };
  return (
    <div className={`styles-accordion${open ? " open" : ""}`}>
      <button
        type="button"
        className="styles-accordion-head"
        onClick={toggle}
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
