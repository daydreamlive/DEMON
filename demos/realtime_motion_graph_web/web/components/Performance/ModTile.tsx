"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { DCW_MODES, DCW_WAVELETS, RCFG_MODES, type RcfgMode } from "@/types/engine";

import { Knob } from "./Knob";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// MOD tab — time-variant / model-internal expert knobs, grouped into
// three labeled sub-tiles (Engine / DCW / CFG) that visually mirror
// the Styles tab's stacked-card layout. Each subsystem keeps its own
// knobs and dropdowns together inside its own framed panel:
//
//   Engine    shift, feedback, feedback_depth, plus the pipeline
//             depth dropdown to the right of the knob row.
//   DCW       dcw_* knobs (dcw_high_scaler only when mode==="double"),
//             followed by a single-row config strip with the DCW
//             toggle, mode select, and wavelet select.
//   CFG       guidance_scale + cfg_rescale knobs (only when RCFG is
//             engaged), followed by the RCFG mode dropdown.

export function ModTile() {
  const dcwEnabled = usePerformanceStore((s) => s.dcwEnabled);
  const dcwMode = usePerformanceStore((s) => s.dcwMode);
  const dcwWavelet = usePerformanceStore((s) => s.dcwWavelet);
  const toggleDcw = usePerformanceStore((s) => s.toggleDcw);
  const setMode = usePerformanceStore((s) => s.setDcwMode);
  const setWavelet = usePerformanceStore((s) => s.setDcwWavelet);
  const rcfgMode = usePerformanceStore((s) => s.rcfgMode);
  const setRcfgMode = usePerformanceStore((s) => s.setRcfgMode);
  const pipelineDepth = useSessionStore((s) => s.pipelineDepth);
  const maxPipelineDepth = useSessionStore((s) => s.maxPipelineDepth);
  const remote = useSessionStore((s) => s.remote);

  const depthEnabled =
    remote !== null && maxPipelineDepth !== null && maxPipelineDepth >= 1;
  const depthOptions = depthEnabled
    ? Array.from({ length: maxPipelineDepth! }, (_, i) => i + 1)
    : [];
  const depthValue =
    typeof pipelineDepth === "number" ? String(pipelineDepth) : "";

  return (
    <div className="mod-tab" data-tile="mod">
      <div className="mixer-tile" data-tile="mod-engine">
        <div className="mixer-tile-label">Engine</div>
        {/* Knobs on the left, depth select on the right of the same
            row — saves vertical space vs. stacking the select underneath.
            flex-wrap lets the select drop to a new line on narrow
            drawers. */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            flexWrap: "wrap",
          }}
        >
          <div className="knob-rack">
            <Knob
              param="shift"
              label={defaultLabelFor("shift")}
              kbd={kbdHintFor("shift")}
            />
            <Knob
              param="feedback"
              label={defaultLabelFor("feedback")}
              kbd={kbdHintFor("feedback")}
            />
            <Knob
              param="feedback_depth"
              label={defaultLabelFor("feedback_depth")}
              kbd={kbdHintFor("feedback_depth")}
            />
          </div>
          <label
            className="dcw-row"
            title="Pipeline depth — concurrent denoising slots in the StreamDiffusion ring buffer. Low depth = faster param-update latency (best for discrete, snappy changes); high depth = higher throughput / updates per second (best for smooth glides) and better GPU utilization. Capped to the TRT engine's max batch size (or 4 in eager/compile)."
          >
            <span className="dcw-row-label">depth</span>
            <select
              className="dcw-select"
              value={depthValue}
              disabled={!depthEnabled}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10);
                if (!Number.isFinite(v)) return;
                remote?.sendSetDepth(v);
              }}
            >
              {depthOptions.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="mixer-tile" data-tile="mod-dcw">
        <div className="mixer-tile-label">DCW</div>
        <div className="knob-rack">
          <Knob
            param="dcw_scaler"
            label={dcwMode === "double" ? "DCW low" : "DCW"}
            kbd={kbdHintFor("dcw_scaler")}
          />
          {dcwMode === "double" && (
            <Knob
              param="dcw_high_scaler"
              label={defaultLabelFor("dcw_high_scaler")}
              kbd={kbdHintFor("dcw_high_scaler")}
            />
          )}
          <Knob
            param="dcw_mult_blend"
            label="mult blend"
            kbd={kbdHintFor("dcw_mult_blend")}
          />
          <Knob
            param="dcw_mag_phase"
            label="mag/phase"
            kbd={kbdHintFor("dcw_mag_phase")}
          />
          <Knob
            param="dcw_soft_thresh"
            label="soft τ"
            kbd={kbdHintFor("dcw_soft_thresh")}
          />
        </div>
        {/* DCW config row — toggle, mode, wavelet on one horizontal
            line. flex-wrap lets the row break onto two if the drawer
            is narrow. */}
        <div
          className="knob-dcw"
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "12px",
          }}
        >
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

      <div className="mixer-tile" data-tile="mod-cfg">
        <div className="mixer-tile-label">CFG</div>
        {rcfgMode !== "off" && (
          <div className="knob-rack">
            <Knob
              param="guidance_scale"
              label={defaultLabelFor("guidance_scale")}
              kbd={kbdHintFor("guidance_scale")}
            />
            <Knob
              param="cfg_rescale"
              label={defaultLabelFor("cfg_rescale")}
              kbd={kbdHintFor("cfg_rescale")}
            />
          </div>
        )}
        <div className="dcw-panel knob-dcw">
          <label
            className="dcw-row"
            title="RCFG mode. 'off' = no guidance (turbo default). Other modes re-introduce classifier-free guidance at near-zero cost over baseline."
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
    </div>
  );
}
