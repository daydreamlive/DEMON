"use client";

import type { StemSourceMode } from "@/engine/audio/loadFixture";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";

// Live inference-source hotswap for the active upload: pick which version
// of the track feeds the model — full mix, instruments-only, or vocals-
// only. Writing setSourceMode() is all this does; useFixtureSwap watches
// the active track's sourceMode and re-runs the swap (server re-encodes
// the chosen stem) without restarting the session. The currently-loaded
// mode is shown active and disabled so a redundant re-swap can't fire.
//
// Disabled wholesale while a swap is in flight (stemStatus === processing)
// so rapid clicks don't queue overlapping re-encodes.

const MODES: ReadonlyArray<{
  mode: StemSourceMode;
  label: string;
  hint: string;
}> = [
  { mode: "full", label: "Full", hint: "Feed the whole mix to inference" },
  {
    mode: "instruments",
    label: "Instr",
    hint: "Feed only the instrumental bed to inference",
  },
  { mode: "vocals", label: "Vocals", hint: "Feed only the vocal stem to inference" },
];

interface Props {
  fixture: string;
  current: StemSourceMode;
  /** Swap in flight — lock the whole switch until it settles. */
  busy: boolean;
}

export function SourceModeSwitch({ fixture, current, busy }: Props) {
  const setSourceMode = useCustomTracksStore((s) => s.setSourceMode);
  return (
    <div
      className="source-mode-switch"
      role="radiogroup"
      aria-label="Inference source"
    >
      {MODES.map(({ mode, label, hint }) => {
        const active = current === mode;
        return (
          <button
            key={mode}
            type="button"
            role="radio"
            aria-checked={active}
            className={`source-mode-option${active ? " is-active" : ""}`}
            disabled={busy || active}
            onClick={() => setSourceMode(fixture, mode)}
            data-dd-tooltip={hint}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
