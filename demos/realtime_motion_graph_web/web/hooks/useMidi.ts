"use client";

import { useEffect } from "react";

import { loraStrengthDispatcher } from "@/engine/lora/dispatcher";
import { readKnob, resetKnobDelta } from "@/engine/midi/absoluteDelta";
import { decodeKnob } from "@/engine/midi/knob";
import { LORA_SLOT_MARKER, type NoteAction } from "@/engine/midi/types";
import { getChannelRange } from "@/lib/config";
import { useCurveStore } from "@/store/useCurveStore";
import { useLoraStore } from "@/store/useLoraStore";
import { useMidiStore } from "@/store/useMidiStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { LORA_SLIDER_MAX, SLIDER_META } from "@/types/engine";

// Web MIDI bootstrap. Asks for navigator.requestMIDIAccess on mount, wires
// onmidimessage to either the learn handler (if learn is active) or the
// CC/note router (otherwise). Mirrors the contextmenu→learn binding from
// app.js.

function noteAction(action: NoteAction): void {
  switch (action) {
    case "seed":
      usePerformanceStore.getState().randomizeSeed();
      return;
    case "send_prompt": {
      const perf = usePerformanceStore.getState();
      const remote = useSessionStore.getState().remote;
      if (remote) {
        remote.sendPrompt(perf.promptA, perf.activeKey, perf.activeTimeSignature);
      }
      return;
    }
    case "mode_toggle":
      usePerformanceStore.getState().toggleMode();
      return;
    case "pause":
      usePerformanceStore.getState().togglePause();
      return;
    case "kiosk_toggle":
      usePerformanceStore.getState().toggleKiosk();
      return;
    case "schedule_curves_toggle":
      useCurveStore.getState().toggleOverlay();
      return;
  }
}

function resolveCcParam(rawName: string): string | null {
  if (rawName === LORA_SLOT_MARKER[0] || rawName === LORA_SLOT_MARKER[1]) {
    const ids = Array.from(useLoraStore.getState().enabled);
    const idx = rawName === LORA_SLOT_MARKER[0] ? 0 : 1;
    return ids[idx] ? `lora_str_${ids[idx]}` : null;
  }
  return rawName;
}

function handleCC(cc: number, value: number): void {
  if (useMidiStore.getState().applyLearn("cc", cc)) return;

  const map = useMidiStore.getState().map;

  // CC-as-action bindings (pads in CC mode mapped to discrete actions
  // like seed-randomize). Fire only on the rising edge — pad presses
  // typically send a non-zero value on press and 0 on release.
  const ccAction = map.ccActions?.[String(cc)];
  if (ccAction && value > 0) {
    noteAction(ccAction);
    return;
  }

  const rawName = map.cc[String(cc)];
  if (!rawName) return;
  const param = resolveCcParam(rawName);
  if (!param) return;

  const meta = SLIDER_META[param];
  // Per-channel range wins over SLIDER_META so MIDI obeys the same caps
  // as the slider widget. Reverse flips the direction of both absolute
  // mapping (knob CW → engine value DOWN) and relative deltas (knob
  // tick UP → engine value DOWN), mirroring SliderGroup's behavior.
  const range = getChannelRange(param);
  const min = range?.min ?? meta?.min ?? 0;
  // LoRA strength sliders (`lora_str_<id>`) aren't in SLIDER_META;
  // their range is fixed by LORA_SLIDER_MAX (matches the LibraryTile
  // widget, edge bars, and useScheduledCurves). Without this branch
  // an absolute MIDI knob's full sweep would map 0..127 → 0..2.0 and
  // the perf-store clamp would silently truncate the top ~10% — the
  // operator-visible slider stops at 1.8 but the MIDI input still
  // crosses it.
  const max = range?.max
    ?? meta?.max
    ?? (param.startsWith("lora_str_") ? LORA_SLIDER_MAX : 2.0);
  const span = Math.max(0, max - min);
  const reverse = range?.reverse ?? false;
  const step = meta?.step ?? 0.05;
  const dirSign = reverse ? -1 : 1;

  const decoded = decodeKnob(cc, value);
  const perf = usePerformanceStore.getState();
  if (decoded.mode === "relative") {
    if (!decoded.delta) return;
    applyMidiBump(param, decoded.delta * step * dirSign);
    return;
  }

  // Absolute knobs: delta-track most of the way (no takeover snap), but
  // hard-snap when the knob is parked at min/max so a fast sweep
  // always reaches the bound.
  const reading = readKnob(cc, value);
  if (reading.absolute !== null) {
    const knobFrac = reading.absolute / 127;
    const fwd = reverse ? 1 - knobFrac : knobFrac;
    applyMidiSet(param, min + fwd * span);
    return;
  }
  if (reading.delta === null) return;
  applyMidiBump(param, (reading.delta / 127) * span * dirSign);
}

/** Setter that propagates to both the perf store (drives engine via
 *  param-sync, also runs the smoothing tween) AND to useLoraStore for
 *  lora_str_<id> params (drives the LoRA UI's strength display, since
 *  LoraRow reads from useLoraStore). Without the LoRA mirror, MIDI
 *  knobs would change the engine's behaviour but the visual slider in
 *  the Library tile would stay frozen.
 *
 *  lora_str_<id> params route through loraStrengthDispatcher so MIDI
 *  knob sweeps debounce into one engine-side refit per gesture,
 *  matching the touch/edge-drag paths. */
function applyMidiSet(param: string, value: number): void {
  if (param.startsWith("lora_str_")) {
    const id = param.slice("lora_str_".length);
    loraStrengthDispatcher.set(id, value);
    return;
  }
  usePerformanceStore.getState().setSlider(param, value);
}

function applyMidiBump(param: string, delta: number): void {
  if (param.startsWith("lora_str_")) {
    const id = param.slice("lora_str_".length);
    // Compute the new absolute target from sliderTargets (kept current
    // by dispatcher.set on every prior bump) and route the result
    // through the dispatcher; clamping happens inside.
    const current = usePerformanceStore.getState().sliderTargets[param] ?? 0;
    loraStrengthDispatcher.set(id, current + delta);
    return;
  }
  usePerformanceStore.getState().bumpSlider(param, delta);
}

function handleNote(note: number): void {
  if (useMidiStore.getState().applyLearn("note", note)) return;
  const action = useMidiStore.getState().map.notes[String(note)];
  if (!action) return;
  noteAction(action);
}

function bindInput(input: MIDIInput): void {
  input.onmidimessage = (e) => {
    const data = e.data;
    if (!data || data.length < 2) return;
    const status = data[0] & 0xf0;
    // Diagnostic for MIDI-learn debugging: when learn is active, dump
    // every raw message so we can see whether the pad fires a status
    // byte the dispatcher doesn't route (e.g. 0xa0 aftertouch, 0xc0
    // program change, or note-on with velocity 0 used as the "press"
    // signal). Drop this log once the pad-bind issue is fully diagnosed.
    if (useMidiStore.getState().learn) {
      const statusHex = (data[0] | 0).toString(16).padStart(2, "0");
      console.log(
        `[midi-learn] raw: status=0x${statusHex} data1=${data[1]} data2=${data[2] ?? "n/a"} len=${data.length}`,
      );
    }
    if (status === 0xb0) {
      // Control change.
      handleCC(data[1], data[2]);
    } else if (status === 0x90 && data[2] > 0) {
      // Note on (velocity > 0).
      handleNote(data[1]);
    } else if (status === 0x80 || (status === 0x90 && data[2] === 0)) {
      // Note off (or note-on vel=0, which some controllers send instead
      // of a proper 0x80 note-off). When LEARN is active, treat this as
      // a binding hint too — some "press" pads only emit on release, so
      // refusing to bind on note-off leaves those pads unbindable. The
      // normal dispatch (non-learn) still ignores note-off, matching the
      // long-standing fire-on-rising-edge behavior for action buttons.
      if (useMidiStore.getState().learn) {
        handleNote(data[1]);
      }
    }
  };
}

export function useMidi() {
  useEffect(() => {
    if (typeof navigator === "undefined" || !navigator.requestMIDIAccess) {
      useMidiStore.getState().setStatus("MIDI N/A", "off");
      return;
    }

    let access: MIDIAccess | null = null;
    let cancelled = false;

    navigator
      .requestMIDIAccess({ sysex: false })
      .then((a) => {
        if (cancelled) return;
        access = a;
        useMidiStore.getState().setAvailable(true);
        useMidiStore
          .getState()
          .setStatus(`MIDI ${a.inputs.size} dev`, "ok");
        a.inputs.forEach(bindInput);
        a.onstatechange = () => {
          a.inputs.forEach(bindInput);
          // Plug/unplug invalidates the per-CC `lastValue` cache used
          // by readKnob() — the next message from a reconnected
          // controller would otherwise compute its delta against a
          // pre-disconnect value, yanking every bound slider on the
          // first wiggle. Clearing here means the first message after
          // any state change is treated as a fresh baseline (no
          // motion), matching first-time-mapping behaviour.
          resetKnobDelta();
          useMidiStore
            .getState()
            .setStatus(`MIDI ${a.inputs.size} dev`, "ok");
        };
      })
      .catch(() => {
        useMidiStore.getState().setStatus("MIDI denied", "warn");
      });

    // Right-click → MIDI learn. Targets:
    //   .slider-group[data-param=...]      → CC (vertical faders)
    //   .knob-group[data-param=...]        → CC (rotary knobs — Knob.tsx)
    //   .hero-style-fader[data-param=...]  → CC (bottom-bay LoRA strength
    //                                       faders in HeroMacros — set to
    //                                       `lora_slot_<0|1>` so the
    //                                       binding survives LoRA swaps)
    //   #blend-control[data-param=...]     → CC (Tags A↔B blend slider,
    //                                       intentionally NOT a
    //                                       slider-group — keeps the
    //                                       horizontal rail styling)
    //   [data-midi-learn=...]              → note (transport buttons, send-prompt, etc.)
    //
    // LoRA rows used to have a `.lora-row[data-param=...]` branch here.
    // It's gone — LibraryTile now owns its own right-click via a
    // proper context menu (MIDI learn + Copy trigger), which calls
    // `useMidiStore.startLearn` directly and stops the contextmenu's
    // propagation so this document-level handler never sees it.
    const onContextMenu = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const slider = target.closest<HTMLElement>(".slider-group");
      if (slider?.dataset.param) {
        e.preventDefault();
        useMidiStore.getState().startLearn("cc", slider.dataset.param, slider);
        return;
      }
      const knob = target.closest<HTMLElement>(".knob-group");
      if (knob?.dataset.param) {
        e.preventDefault();
        useMidiStore.getState().startLearn("cc", knob.dataset.param, knob);
        return;
      }
      const heroFader = target.closest<HTMLElement>(".hero-style-fader");
      if (heroFader?.dataset.param) {
        e.preventDefault();
        useMidiStore
          .getState()
          .startLearn("cc", heroFader.dataset.param, heroFader);
        return;
      }
      const blendEl = target.closest<HTMLElement>("#blend-control");
      if (blendEl?.dataset.param) {
        e.preventDefault();
        useMidiStore.getState().startLearn("cc", blendEl.dataset.param, blendEl);
        return;
      }
      const learnEl = target.closest<HTMLElement>("[data-midi-learn]");
      if (learnEl?.dataset.midiLearn) {
        e.preventDefault();
        useMidiStore
          .getState()
          .startLearn("note", learnEl.dataset.midiLearn, learnEl);
        return;
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && useMidiStore.getState().learn) {
        useMidiStore.getState().cancelLearn();
        useMidiStore.getState().setStatus("Learn cancelled", "info");
      }
    };
    document.addEventListener("contextmenu", onContextMenu);
    document.addEventListener("keydown", onKeyDown);

    return () => {
      cancelled = true;
      document.removeEventListener("contextmenu", onContextMenu);
      document.removeEventListener("keydown", onKeyDown);
      if (access) {
        access.inputs.forEach((i) => {
          i.onmidimessage = null;
        });
        access.onstatechange = null;
      }
    };
  }, []);
}
