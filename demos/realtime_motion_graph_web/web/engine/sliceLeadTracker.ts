// Hand-off point between the slice listener (useStartSession) and the
// 125 Hz param sync (useParamSync) for the slice landing-lead report.
//
// The slice listener notes how far ahead of the audible playhead each
// slice landed AT APPLY TIME — which bakes in everything between the
// server's write and the listener's ear: WS transit, decode, and main-
// thread scheduling (a background-throttled tab applies slices in ~1 s
// bursts, so leads measured here go negative even on a perfect network).
// useParamSync ships the WORST lead since its previous tick as
// `slice_lead_s`; the server widens its playback lead until reports stay
// positive. Worst-of-interval (not latest) so a single late slice inside
// an otherwise healthy second still registers.
//
// Module singleton — at most one session is active per page, and both
// hooks already live at app scope.

let worstLeadS: number | null = null;

export function noteSliceLead(leadS: number): void {
  if (!Number.isFinite(leadS)) return;
  worstLeadS = worstLeadS === null ? leadS : Math.min(worstLeadS, leadS);
}

/** Returns the worst lead since the previous take (or null) and resets. */
export function takeSliceLead(): number | null {
  const w = worstLeadS;
  worstLeadS = null;
  return w;
}
