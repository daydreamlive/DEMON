"use client";

import { useMemo } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useKnobManifestStore } from "@/store/useKnobManifestStore";
import { isDcwMode, isDcwWavelet, isRcfgMode } from "@/types/engine";
import type { KnobManifestEntry } from "@demon/client";

import { SliderGroup } from "./SliderGroup";

// Auto-generated control surface. Renders the ENTIRE knob set straight
// from the backend manifest (GET /api/knobs) with zero hand-declared knob
// list — the reference template for a re-skinned or vibecoded interface:
// fetch the manifest, render by `type`, group by `group`, and drive the
// live session through the same stores the shipped tiles use.
//
//   float / int  -> SliderGroup (reuses its drag / wheel / MIDI / tactile
//                   behavior); bounds come from the manifest. Very large
//                   ranges (seed) fall back to a number input.
//   enum         -> <select> over `options`
//   bool         -> checkbox
//
// Numeric knobs dispatch via usePerformanceStore.setSlider, so they ride
// the existing param-sync to the engine. The handful of string/bool knobs
// map to their dedicated store actions; any enum/bool the client doesn't
// know how to bind renders disabled with a marker (future-proofing — there
// are none today).

const GROUP_ORDER = ["core", "groups", "keystones", "guidance", "dcw"];
const GROUP_LABELS: Record<string, string> = {
  core: "Core",
  groups: "Channel Groups",
  keystones: "Keystones",
  guidance: "Guidance",
  dcw: "DCW",
};

// Above this ceiling a fader is useless (seed spans the full uint32) — fall
// back to a number input.
const SLIDER_MAX_RANGE = 1_000_000;

const prettify = (name: string) => name.replace(/_/g, " ");

export function DynamicKnobPanel() {
  const knobs = useKnobManifestStore((s) => s.knobs);
  const loaded = useKnobManifestStore((s) => s.loaded);

  const groups = useMemo(() => {
    const byGroup = new Map<string, Array<[string, KnobManifestEntry]>>();
    for (const [name, spec] of Object.entries(knobs)) {
      const g = spec.group || "core";
      if (!byGroup.has(g)) byGroup.set(g, []);
      byGroup.get(g)!.push([name, spec]);
    }
    const rank = (g: string) => {
      const i = GROUP_ORDER.indexOf(g);
      return i < 0 ? GROUP_ORDER.length : i;
    };
    return [...byGroup.keys()]
      .sort((a, b) => rank(a) - rank(b))
      .map((g) => [g, byGroup.get(g)!] as const);
  }, [knobs]);

  if (!loaded || groups.length === 0) {
    return (
      <div className="knob-tile" data-tile="auto">
        <div className="dyn-knob-empty">
          {loaded ? "No knobs in manifest." : "Loading knob manifest…"}
        </div>
      </div>
    );
  }

  return (
    <div className="knob-tile" data-tile="auto">
      <p className="dyn-knob-note">
        Rendered entirely from <code>/api/knobs</code> — no hand-declared
        knob list. Template for a re-skinned interface.
      </p>
      {groups.map(([group, entries]) => (
        <section key={group} className="dyn-knob-group">
          <h4 className="dyn-knob-group-label">
            {GROUP_LABELS[group] ?? group}
          </h4>
          <div className="knob-rack">
            {entries.map(([name, spec]) => (
              <DynamicKnob key={name} name={name} spec={spec} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function DynamicKnob({ name, spec }: { name: string; spec: KnobManifestEntry }) {
  if (spec.type === "enum") return <EnumKnob name={name} spec={spec} />;
  if (spec.type === "bool") return <BoolKnob name={name} spec={spec} />;
  // float / int — drive through the slider system.
  const max = typeof spec.max === "number" ? spec.max : 1.0;
  if (max > SLIDER_MAX_RANGE) return <NumberKnob name={name} spec={spec} />;
  return (
    <SliderGroup
      param={name}
      label={prettify(name)}
      max={max}
      min={typeof spec.min === "number" ? spec.min : 0}
    />
  );
}

function NumberKnob({ name, spec }: { name: string; spec: KnobManifestEntry }) {
  const fallback = typeof spec.default === "number" ? spec.default : 0;
  const value = usePerformanceStore((s) => s.sliderTargets[name] ?? fallback);
  const setSlider = usePerformanceStore((s) => s.setSlider);
  return (
    <label className="dyn-knob-field" title={spec.description}>
      <span className="dyn-knob-field-label">{prettify(name)}</span>
      <input
        className="dyn-knob-number"
        type="number"
        value={value}
        min={spec.min}
        max={spec.max}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!Number.isNaN(v)) setSlider(name, v);
        }}
      />
    </label>
  );
}

function EnumKnob({ name, spec }: { name: string; spec: KnobManifestEntry }) {
  const options = (spec.options ?? []).map(String);
  // Read every bindable enum field; resolve the active one by knob name.
  const rcfgMode = usePerformanceStore((s) => s.rcfgMode);
  const dcwMode = usePerformanceStore((s) => s.dcwMode);
  const dcwWavelet = usePerformanceStore((s) => s.dcwWavelet);
  const setRcfgMode = usePerformanceStore((s) => s.setRcfgMode);
  const setDcwMode = usePerformanceStore((s) => s.setDcwMode);
  const setDcwWavelet = usePerformanceStore((s) => s.setDcwWavelet);

  // Each binding pairs the store setter with the app vocabulary's type
  // guard: the setter only fires on values the guard admits, and manifest
  // options outside the vocabulary are filtered from the dropdown instead
  // of cast through. The vocabularies can deliberately diverge — the
  // registry declares rcfg "full" but the app hides it (turbo is
  // CFG-distilled; see types/engine.ts) — and this keeps the manifest-
  // driven panel from punching through that decision.
  let value: string | undefined;
  let allows: ((v: string) => boolean) | undefined;
  let onChange: ((v: string) => void) | undefined;
  if (name === "rcfg_mode") {
    value = rcfgMode;
    allows = isRcfgMode;
    onChange = (v) => {
      if (isRcfgMode(v)) setRcfgMode(v);
    };
  } else if (name === "dcw_mode") {
    value = dcwMode;
    allows = isDcwMode;
    onChange = (v) => {
      if (isDcwMode(v)) setDcwMode(v);
    };
  } else if (name === "dcw_wavelet") {
    value = dcwWavelet;
    allows = isDcwWavelet;
    onChange = (v) => {
      if (isDcwWavelet(v)) setDcwWavelet(v);
    };
  }
  const bound = !!onChange;
  const shown = allows ? options.filter(allows) : options;

  return (
    <label className="dyn-knob-field" title={spec.description}>
      <span className="dyn-knob-field-label">{prettify(name)}</span>
      <select
        className="dyn-knob-select"
        value={value ?? shown[0] ?? ""}
        disabled={!bound}
        onChange={(e) => onChange?.(e.target.value)}
      >
        {shown.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      {!bound && <em className="dyn-knob-unbound">no client binding</em>}
    </label>
  );
}

function BoolKnob({ name, spec }: { name: string; spec: KnobManifestEntry }) {
  const dcwEnabled = usePerformanceStore((s) => s.dcwEnabled);
  const setDcwEnabled = usePerformanceStore((s) => s.setDcwEnabled);

  let value: boolean | undefined;
  let onChange: ((v: boolean) => void) | undefined;
  if (name === "dcw_enabled") {
    value = dcwEnabled;
    onChange = setDcwEnabled;
  }
  const bound = !!onChange;

  return (
    <label className="dyn-knob-field dyn-knob-field--bool" title={spec.description}>
      <input
        type="checkbox"
        checked={!!value}
        disabled={!bound}
        onChange={(e) => onChange?.(e.target.checked)}
      />
      <span className="dyn-knob-field-label">{prettify(name)}</span>
      {!bound && <em className="dyn-knob-unbound">no client binding</em>}
    </label>
  );
}
