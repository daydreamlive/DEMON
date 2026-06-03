"use client";

import { createContext, useContext } from "react";

// Hosts that vendor this UI (notably daydream-public-demo) need a way
// to intercept high-intent user actions and decide whether to allow
// them — typically to enforce a sign-up wall on anonymous track-change,
// upload, and mic-record, the three clearest "I'm engaged" signals the
// app exposes.
//
// The mechanism is a context-supplied async predicate. Components on
// the click path await it before calling the underlying store action;
// `false` aborts cleanly without side effects (no fan close, no decode,
// no file picker opening, no mic permission prompt). Default gate
// allows everything, so DEMON's own standalone demos (and any other
// consumer that doesn't mount a provider) are unaffected.
//
// Why a context and not a prop: the gated actions sit deep in
// AudioSourceCrate / LiteTrackCarousel / TrackPicker and a host that
// wants to gate them shouldn't have to thread a prop through every
// surface that might add gated affordances later. A single provider
// at the Performance boundary covers them all.

export type ActionGateKind = "track_change" | "upload" | "mic";

export type ActionGate = (kind: ActionGateKind) => boolean | Promise<boolean>;

const ActionGateContext = createContext<ActionGate>(() => true);

export const ActionGateProvider = ActionGateContext.Provider;

export function useActionGate(): ActionGate {
  return useContext(ActionGateContext);
}
