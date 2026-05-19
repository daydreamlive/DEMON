"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { DCW_MODES, DCW_WAVELETS } from "@/types/engine";

import { SliderGroup } from "./SliderGroup";
import { kbdHintFor } from "./SliderTile";

// DCW tile. Up to five faders, two selects, one toggle.
//
//  - dcw_scaler        : primary correction strength. Active in every
//                        mode (low: low band; high: high band; pix:
//                        full latent; double: low band).
//  - dcw_high_scaler   : SECOND-band strength. Only meaningful in
//                        ``double`` mode (drives the high band there);
//                        hidden in low/high/pix because the kernel
//                        ignores it.
//  - dcw_mult_blend    : 0 = additive ←→ 1 = multiplicative band update
//  - dcw_mag_phase     : 0 = real-valued correction ←→ 1 = full
//                        magnitude+phase analytic-signal correction
//  - dcw_soft_thresh   : sparsity threshold τ applied to (xb - yb)
//                        before the additive update; τ = 0 is a no-op
//
// The three advanced faders all default to zero. With every fader at
// zero the corrector takes the byte-identical upstream-v0.1.7 fast path.

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
          label={dcwMode === "double" ? "DCW low" : "DCW"}
          kbd={kbdHintFor("dcw_scaler")}
        />
        {dcwMode === "double" && (
          <SliderGroup
            param="dcw_high_scaler"
            label="DCW high"
            kbd={kbdHintFor("dcw_high_scaler")}
          />
        )}
        <SliderGroup
          param="dcw_mult_blend"
          label="mult blend"
          kbd={kbdHintFor("dcw_mult_blend")}
        />
        <SliderGroup
          param="dcw_mag_phase"
          label="mag/phase"
          kbd={kbdHintFor("dcw_mag_phase")}
        />
        <SliderGroup
          param="dcw_soft_thresh"
          label="soft τ"
          kbd={kbdHintFor("dcw_soft_thresh")}
        />
      </div>
      {/* Bottom strip: ON/OFF + mode + wavelet laid out horizontally so
          the tile's vertical footprint matches the height of the faders
          alone instead of being padded to the height of the old
          three-row right-side panel. */}
      <div className="dcw-panel dcw-panel--bottom">
        <button
          type="button"
          className={`dcw-toggle${dcwEnabled ? " active" : ""}`}
          data-role="dcw-enabled"
          data-dd-tooltip="Toggle DCW (T)"
          onClick={toggleDcw}
        >
          DCW: {dcwEnabled ? "ON" : "OFF"}
        </button>
        <label className="dcw-row" data-dd-tooltip="Cycle DCW mode (Shift + T)">
          <span className="dcw-row-label">mode</span>
          <select
            className="dcw-select"
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
        <label className="dcw-row" data-dd-tooltip="Cycle wavelet (Shift + W)">
          <span className="dcw-row-label">wavelet</span>
          <select
            className="dcw-select"
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
