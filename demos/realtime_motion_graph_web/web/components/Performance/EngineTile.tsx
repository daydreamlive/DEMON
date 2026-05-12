"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { RCFG_MODES, type RcfgMode } from "@/types/engine";

import { SliderGroup } from "./SliderGroup";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// Engine tile. Engine-internal scalars on the top (feedback, shift,
// ode_noise), plus a conditional CFG cluster (guidance_scale,
// cfg_rescale) when RCFG is not off, plus the RCFG mode dropdown in
// the bottom strip mirroring DcwTile's layout.
//
// The CFG sliders only appear when ``rcfgMode != "off"`` — keeping
// them visible when guidance is disabled is just visual noise (the
// server ignores them entirely on the off path). The param-sync tick
// still ships the current values so flipping the dropdown back on
// resumes at whatever the operator last set.

const ALWAYS_SLIDERS = [
  "feedback",
  "shift",
  "ode_noise",
];

export function EngineTile() {
  const rcfgMode = usePerformanceStore((s) => s.rcfgMode);
  const setRcfgMode = usePerformanceStore((s) => s.setRcfgMode);

  return (
    <div className="mixer-tile" data-tile="engine">
      <div className="mixer-tile-label">Engine</div>
      <div className="mixer-channels">
        {ALWAYS_SLIDERS.map((p) => (
          <SliderGroup
            key={p}
            param={p}
            label={defaultLabelFor(p)}
            kbd={kbdHintFor(p)}
          />
        ))}
        {rcfgMode !== "off" && (
          <>
            <SliderGroup
              param="guidance_scale"
              label={defaultLabelFor("guidance_scale")}
              kbd={kbdHintFor("guidance_scale")}
            />
            <SliderGroup
              param="cfg_rescale"
              label={defaultLabelFor("cfg_rescale")}
              kbd={kbdHintFor("cfg_rescale")}
            />
          </>
        )}
      </div>
      <div className="dcw-panel dcw-panel--bottom">
        <label
          className="dcw-row"
          data-dd-tooltip="RCFG mode. 'off' = no guidance (turbo default). Other modes re-introduce classifier-free guidance at near-zero cost over baseline."
        >
          <span className="dcw-row-label">RCFG</span>
          <select
            className="dcw-select"
            value={rcfgMode}
            onChange={(e) => setRcfgMode(e.target.value as RcfgMode)}
          >
            {RCFG_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
