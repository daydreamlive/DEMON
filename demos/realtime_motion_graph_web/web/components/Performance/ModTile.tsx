"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { DCW_MODES, DCW_WAVELETS } from "@/types/engine";

import { Knob } from "./Knob";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// MOD tab — time-variant / model-internal expert knobs. These don't
// map cleanly onto any 40-year analog/digital tradition, so the labels
// stay technical (SHIFT, N.SHARE, JITTER) and live behind their own
// tab instead of CORE.
//
// Drawn as knobs (matches CORE's visual vocabulary — these are still
// continuous "tweak with one hand" params). The DCW on/off + mode +
// wavelet panel rides along as the expert config block.
export function ModTile() {
  const dcwEnabled = usePerformanceStore((s) => s.dcwEnabled);
  const dcwMode = usePerformanceStore((s) => s.dcwMode);
  const dcwWavelet = usePerformanceStore((s) => s.dcwWavelet);
  const toggleDcw = usePerformanceStore((s) => s.toggleDcw);
  const setMode = usePerformanceStore((s) => s.setDcwMode);
  const setWavelet = usePerformanceStore((s) => s.setDcwWavelet);

  return (
    <div className="knob-tile" data-tile="mod">
      <div className="knob-rack">
        <Knob
          param="shift"
          label={defaultLabelFor("shift")}
          kbd={kbdHintFor("shift")}
        />
        <Knob
          param="dcw_scaler"
          label={defaultLabelFor("dcw_scaler")}
          kbd={kbdHintFor("dcw_scaler")}
        />
        <Knob
          param="dcw_high_scaler"
          label={defaultLabelFor("dcw_high_scaler")}
          kbd={kbdHintFor("dcw_high_scaler")}
        />
      </div>
      <div className="dcw-panel knob-dcw">
        <button
          type="button"
          className={`dcw-toggle${dcwEnabled ? " active" : ""}`}
          data-role="dcw-enabled"
          data-dd-tooltip="Toggle DCW (T)"
          onClick={toggleDcw}
        >
          DCW: {dcwEnabled ? "ON" : "OFF"}
        </button>
        <label className="dcw-row">
          <span className="dcw-row-label">DCW mode</span>
          <select
            className="dcw-select"
            title="Cycle DCW mode (Shift + T)"
            value={dcwMode}
            onChange={(e) =>
              setMode(e.target.value as (typeof DCW_MODES)[number])
            }
          >
            {DCW_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label className="dcw-row">
          <span className="dcw-row-label">wavelet</span>
          <select
            className="dcw-select"
            title="Cycle wavelet (Shift + W)"
            value={dcwWavelet}
            onChange={(e) =>
              setWavelet(e.target.value as (typeof DCW_WAVELETS)[number])
            }
          >
            {DCW_WAVELETS.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
