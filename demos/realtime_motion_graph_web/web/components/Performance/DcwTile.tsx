"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { DCW_MODES, DCW_WAVELETS } from "@/types/engine";

import { SliderGroup } from "./SliderGroup";
import { kbdHintFor } from "./SliderTile";

// DCW tile. Two faders (low / high scaler) plus a small panel with the
// ON/OFF toggle, mode select, and wavelet select. The non-numeric state
// rides into the params raw dict (see useParamSync) so the server picks it
// up alongside slider values.

export function DcwTile() {
  const dcwEnabled = usePerformanceStore((s) => s.dcwEnabled);
  const dcwMode = usePerformanceStore((s) => s.dcwMode);
  const dcwWavelet = usePerformanceStore((s) => s.dcwWavelet);
  const toggleDcw = usePerformanceStore((s) => s.toggleDcw);
  const setMode = usePerformanceStore((s) => s.setDcwMode);
  const setWavelet = usePerformanceStore((s) => s.setDcwWavelet);

  return (
    <div className="mixer-tile" data-tile="dcw">
      <div className="mixer-tile-label">DCW</div>
      <div className="mixer-channels">
        <SliderGroup
          param="dcw_scaler"
          label="DCW low"
          kbd={kbdHintFor("dcw_scaler")}
        />
        <SliderGroup
          param="dcw_high_scaler"
          label="DCW high"
          kbd={kbdHintFor("dcw_high_scaler")}
        />
        <div className="dcw-panel">
          <button
            type="button"
            className={`dcw-toggle${dcwEnabled ? " active" : ""}`}
            data-role="dcw-enabled"
            title="Toggle DCW (T)"
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
    </div>
  );
}
