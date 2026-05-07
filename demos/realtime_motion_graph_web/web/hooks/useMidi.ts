"use client";

import { useEffect } from "react";

import { readKnob } from "@/engine/midi/absoluteDelta";
import { decodeKnob } from "@/engine/midi/knob";
import { LORA_SLOT_MARKER, type NoteAction } from "@/engine/midi/types";
import { useCurveStore } from "@/store/useCurveStore";
import { useLoraStore } from "@/store/useLoraStore";
import { useMidiStore } from "@/store/useMidiStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { SLIDER_META } from "@/types/engine";

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
      if (remote) remote.sendPrompt(perf.promptA, perf.activeKey);
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
  const max = meta?.max ?? 2.0;
  const step = meta?.step ?? 0.05;

  const decoded = decodeKnob(cc, value);
  const perf = usePerformanceStore.getState();
  if (decoded.mode === "relative") {
    if (!decoded.delta) return;
    applyMidiBump(param, decoded.delta * step);
    return;
  }

  // Absolute knobs: delta-track most of the way (no takeover snap), but
  // hard-snap when the knob is parked at min/max so a fast sweep
  // always reaches the bound.
  const reading = readKnob(cc, value);
  if (reading.absolute !== null) {
    applyMidiSet(param, (reading.absolute / 127) * max);
    return;
  }
  if (reading.delta === null) return;
  applyMidiBump(param, (reading.delta / 127) * max);
}

/** Setter that propagates to both the perf store (drives engine via
 *  param-sync, also runs the smoothing tween) AND to useLoraStore for
 *  lora_str_<id> params (drives the LoRA UI's strength display, since
 *  LoraRow reads from useLoraStore). Without the LoRA mirror, MIDI
 *  knobs would change the engine's behaviour but the visual slider in
 *  the Library tile would stay frozen. */
function applyMidiSet(param: string, value: number): void {
  usePerformanceStore.getState().setSlider(param, value);
  if (param.startsWith("lora_str_")) {
    const id = param.slice("lora_str_".length);
    useLoraStore.getState().setStrength(id, value);
  }
}

function applyMidiBump(param: string, delta: number): void {
  usePerformanceStore.getState().bumpSlider(param, delta);
  if (param.startsWith("lora_str_")) {
    const id = param.slice("lora_str_".length);
    // Mirror the bumped value (the perf store already clamped) so the
    // LoRA UI shows the same number.
    const v = usePerformanceStore.getState().sliderTargets[param] ?? 0;
    useLoraStore.getState().setStrength(id, v);
  }
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
    if (status === 0xb0) {
      // Control change.
      handleCC(data[1], data[2]);
    } else if (status === 0x90 && data[2] > 0) {
      // Note on (velocity > 0).
      handleNote(data[1]);
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
          useMidiStore
            .getState()
            .setStatus(`MIDI ${a.inputs.size} dev`, "ok");
        };
      })
      .catch(() => {
        useMidiStore.getState().setStatus("MIDI denied", "warn");
      });

    // Right-click → MIDI learn. Targets:
    //   .slider-group[data-param=...] → CC
    //   .lora-row[data-param=...]     → CC (Phase 11 will populate)
    //   [data-midi-learn=...]         → note (transport buttons, send-prompt, etc.)
    const onContextMenu = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const slider = target.closest<HTMLElement>(".slider-group");
      if (slider?.dataset.param) {
        e.preventDefault();
        useMidiStore.getState().startLearn("cc", slider.dataset.param, slider);
        return;
      }
      const loraRow = target.closest<HTMLElement>(".lora-row");
      if (loraRow?.dataset.param) {
        e.preventDefault();
        useMidiStore.getState().startLearn("cc", loraRow.dataset.param, loraRow);
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
