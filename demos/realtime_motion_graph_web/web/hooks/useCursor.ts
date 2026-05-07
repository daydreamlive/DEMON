"use client";

import { useEffect } from "react";

import { initCursor } from "@/engine/cursor";

// Toggle the body class that drives the global `cursor: none` rule
// (app/globals.css). The class is only present while the Performance
// surface is mounted — admin pages, 404, and any future shell screens
// keep the system cursor. Inside the Performance UI, modals (anything
// with role="dialog") get their cursor restored via a separate CSS
// override so users can still see where they're clicking.
export function useCursor() {
  useEffect(() => {
    const handle = initCursor();
    document.body.classList.add("cursor-hidden");
    return () => {
      document.body.classList.remove("cursor-hidden");
      handle.destroy();
    };
  }, []);
}
