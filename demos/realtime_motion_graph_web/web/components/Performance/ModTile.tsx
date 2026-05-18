"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { DCW_MODES, DCW_WAVELETS } from "@/types/engine";

import { SliderGroup } from "./SliderGroup";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// The MOD tab body — time-variant and model-internal expert knobs.
// These don't map cleanly onto any 40-year analog/digital tradition, so
// the labels stay technical (SHIFT, N.SHARE, JITTER) and live behind
// their own tab instead of the CORE row a casual user touches first.
//
// Absorbs the DCW on/off + mode + wavelet panel from the old DcwTile.
// BASS/TREBLE moved to CORE because they read instantly; the mode +
// wavelet choices are configuration the user almost never reaches
// during a performance.
export function ModTile() {
  const dcwEnabled = usePerformanceStore((s) => s.dcwEnabled);
  const dcwMode = usePerformanceStore((s) => s.dcwMode);
  const dcwWavelet = usePerformanceStore((s) => s.dcwWavelet);
  const toggleDcw = usePerformanceStore((s) => s.toggleDcw);
  const setMode = usePerformanceStore((s) => s.setDcwMode);
  const setWavelet = usePerformanceStore((s) => s.setDcwWavelet);

  return (
    <div className="mixer-tile" data-tile="mod">
      <div className="mixer-tile-label">Mod</div>
      <div className="mixer-channels">
        <SliderGroup
          param="shift"
          label={defaultLabelFor("shift")}
          kbd={kbdHintFor("shift")}
        />
        <SliderGroup
          param="noise_share"
          label={defaultLabelFor("noise_share")}
          kbd={kbdHintFor("noise_share")}
        />
        <SliderGroup
          param="ode_noise"
          label={defaultLabelFor("ode_noise")}
          kbd={kbdHintFor("ode_noise")}
        />
        <div className="dcw-panel">
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
    </div>
  );
}
