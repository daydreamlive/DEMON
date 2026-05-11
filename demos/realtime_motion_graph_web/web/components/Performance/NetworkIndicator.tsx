"use client";

import { useCurveStore } from "@/store/useCurveStore";
import { useNetworkStore } from "@/store/useNetworkStore";

// Subtle bottom-center pill that fades in when the user is likely
// experiencing stutter — slice arrival jitter, pod stress, or a stall.
// Hidden when healthy. Reads only `quality` so changes to telemetry
// fields don't re-render. pointer-events disabled so it never blocks
// the canvas or the AdvancedDrawer handle behind it. aria-live polite
// so screen readers announce the engagement once, not per eval tick.
//
// When the schedule-curves overlay is open, the pill shifts up so it
// clears the curve tab strip at the overlay's bottom — handled via the
// `data-curves-open` attribute and a CSS bottom-offset bump.
export function NetworkIndicator() {
  const quality = useNetworkStore((s) => s.quality);
  const curvesOpen = useCurveStore((s) => s.overlayOpen);
  if (quality === "healthy") return null;

  return (
    <div
      className="network-indicator"
      data-quality={quality}
      data-curves-open={curvesOpen ? "true" : undefined}
      role="status"
      aria-live="polite"
    >
      <svg
        className="network-indicator__bars"
        width="14"
        height="12"
        viewBox="0 0 14 12"
        aria-hidden="true"
      >
        <rect x="0" y="8" width="3" height="4" />
        <rect x="5" y="4" width="3" height="8" />
        <rect x="10" y="0" width="3" height="12" />
      </svg>
      <span className="network-indicator__label">UNSTABLE CONNECTION</span>
    </div>
  );
}
