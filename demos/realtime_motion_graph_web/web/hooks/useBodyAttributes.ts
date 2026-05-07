"use client";

import { useEffect } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Project performance state onto <body> attributes / classes so DEMON's CSS
// selectors (body[data-mode], body.kiosk, body.cursor-idle, etc.) light up
// without React owning every styling concern.

export function useBodyAttributes() {
  const mode = usePerformanceStore((s) => s.mode);
  const kiosk = usePerformanceStore((s) => s.kiosk);
  const status = useSessionStore((s) => s.status);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.dataset.mode = mode;
    return () => {
      delete document.body.dataset.mode;
    };
  }, [mode]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.classList.toggle("kiosk", kiosk);
    return () => {
      document.body.classList.remove("kiosk");
    };
  }, [kiosk]);

  // session status drives whether the graph + playhead are shown — before
  // the user clicks play, the title screen should not have stray graph
  // lines drawing through the overlay.
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.dataset.session = status;
    return () => {
      delete document.body.dataset.session;
    };
  }, [status]);
}
