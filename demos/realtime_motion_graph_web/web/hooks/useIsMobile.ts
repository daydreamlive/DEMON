"use client";

import { useEffect, useState } from "react";

const QUERY = "(max-width: 768px)";

// Returns true while the viewport matches the mobile breakpoint. SSR-safe:
// initial value is false on the server, then synchronized after mount.
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia(QUERY);
    setIsMobile(mql.matches);
    const onChange = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return isMobile;
}
