"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { RCFG_MODES, type RcfgMode } from "@/types/engine";

import { SliderGroup } from "./SliderGroup";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// Engine tile. Engine-internal scalars on the top (feedback +
// feedback_depth, shift), plus a conditional CFG cluster
// (guidance_scale, cfg_rescale) when RCFG is not off, plus the RCFG
// mode dropdown and the live pipeline-depth picker in the bottom
// strip mirroring DcwTile's layout.
//
// The CFG sliders only appear when ``rcfgMode != "off"`` — keeping
// them visible when guidance is disabled is just visual noise (the
// server ignores them entirely on the off path). The param-sync tick
// still ships the current values so flipping the dropdown back on
// resumes at whatever the operator last set.

const ALWAYS_SLIDERS = [
  "feedback",
  "feedback_depth",
  "shift",
];

export function EngineTile() {
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
        <label
          className="dcw-row"
          data-dd-tooltip="Pipeline depth — concurrent denoising slots in the StreamDiffusion ring buffer. Low depth = faster param-update latency (best for discrete, snappy changes); high depth = higher throughput / updates per second (best for smooth glides) and better GPU utilization. Capped to the TRT engine's max batch size (or 4 in eager/compile)."
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
  );
}
