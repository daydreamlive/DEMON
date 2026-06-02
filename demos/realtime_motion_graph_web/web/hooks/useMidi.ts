"use client";

import { useEffect } from "react";

import type { StemOverlayKind } from "@/engine/audio/loadFixture";
import { loraStrengthDispatcher } from "@/engine/lora/dispatcher";
import { decodeRelativeDelta, MIDI_TICK_T } from "@/engine/midi/absoluteDelta";
import { applyEnumCC, cycleEnum } from "@/engine/midi/enumTargets";
import { LORA_SLOT_MARKER, type CcMode, type NoteAction } from "@/engine/midi/types";
import { getChannelRange } from "@/lib/config";
import { tToValue, valueToT } from "@/lib/sliderMapping";
import { useCurveStore } from "@/store/useCurveStore";
import { useLoraStore } from "@/store/useLoraStore";
import { useMidiStore } from "@/store/useMidiStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import {
  STEM_OVERLAY_MAX,
  useStemOverlayStore,
} from "@/store/useStemOverlayStore";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
  SLIDER_META,
} from "@/types/engine";

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** The control's default value — the anchor the spring modes return to.
 *  Read live (applyConfig overwrites sliderDefaults after boot). LoRA /
 *  stem params aren't in sliderDefaults, so fall back to their natural
 *  rest. */
function liveDefault(param: string): number {
  const d = usePerformanceStore.getState().sliderDefaults[param];
  if (typeof d === "number") return d;
  if (param.startsWith("lora_str_")) {
    return LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX;
  }
  return 0; // stem overlays + anything unknown rest at 0
}

/** The unity the slider WIDGET uses, so absolute/relative MIDI rides the
 *  exact same rail curve as the fader and the scroll wheel. Only the
 *  Voice-channel faders (params starting with "ch") render unity-anchored
 *  at 1.0 (see VoiceTile); everything else is linear. */
function uiUnity(param: string): number | undefined {
  return param.startsWith("ch") ? 1.0 : undefined;
}

/** Current value of any mappable param, for relative/delta math. */
function readCurrent(param: string): number {
  const stem = stemKindFromParam(param);
  if (stem) return useStemOverlayStore.getState().volumes[stem] ?? 0;
  return usePerformanceStore.getState().sliderTargets[param] ?? liveDefault(param);
}

// Stem overlay params (`stem_vocals` / `stem_instruments`) aren't perf-
// store sliders — their level lives in useStemOverlayStore, range 0..MAX.
function stemKindFromParam(param: string): StemOverlayKind | null {
  if (param === "stem_vocals") return "vocals";
  if (param === "stem_instruments") return "instruments";
  return null;
}

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
    case "dcw_enabled":
      usePerformanceStore.getState().toggleDcw();
      return;
    case "smooth":
      usePerformanceStore.getState().toggleSmooth();
      return;
    case "lufs":
      usePerformanceStore.getState().toggleLufs();
      return;
    case "loop":
      usePerformanceStore.getState().toggleLoop();
      return;
    case "seek_start":
      useSessionStore.getState().player?.seek(0);
      return;
    case "drawer_toggle":
      document.dispatchEvent(new CustomEvent("dd:toggle-drawer"));
      return;
    case "drawer_expand_toggle":
      document.dispatchEvent(new CustomEvent("dd:expand-toggle-drawer"));
      return;
    case "stem_vocals_toggle":
      useStemOverlayStore.getState().toggle("vocals");
      return;
    case "stem_instruments_toggle":
      useStemOverlayStore.getState().toggle("instruments");
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

/** `centered` joystick spring: the controller rests at CC 64, which maps
 *  to the control's default. Pushing toward 127 ramps to one rail, toward
 *  0 ramps to the other; each side is an independent linear ramp from the
 *  default, so a default that sits on a bound just leaves that side flat
 *  (no broken fallback). Releasing the stick sends 64, so the parameter
 *  springs straight back to its default — no release detection needed. */
function springCentered(
  value: number,
  min: number,
  max: number,
  reverse: boolean,
  def: number,
): number {
  const hi = reverse ? min : max; // pushing toward 127 lands here
  const lo = reverse ? max : min; // pushing toward 0 lands here
  const out =
    value >= 64
      ? def + ((value - 64) / 63) * (hi - def)
      : def + ((64 - value) / 64) * (lo - def);
  return Math.max(min, Math.min(max, out));
}

/** `unipolar` mod / pitch-wheel spring: the controller rests at CC 0,
 *  which maps to the control's default, and sweeps to the farther rail end
 *  at CC 127 (the end with the most travel from the default, for maximum
 *  throw). Releasing the wheel sends 0, so the parameter springs back to
 *  its default. */
function springUnipolar(
  value: number,
  min: number,
  max: number,
  reverse: boolean,
  def: number,
): number {
  let away = def - min <= max - def ? max : min;
  if (reverse) away = away === max ? min : max;
  const out = def + (value / 127) * (away - def);
  return Math.max(min, Math.min(max, out));
}

/** Drive a continuous param from a 0..127 controller value. Shared by the
 *  CC path (`handleCC`) and the pitch-bend path (`handlePitchBend`); `key`
 *  is the map bucket key (a CC number as a string, or "pb<channel>" for a
 *  pitch-bend axis). `fallbackMode` is used only when the binding has no
 *  explicit type yet (absolute for CC, centered for pitch bend, which is
 *  always a centre-resting controller). */
function applyContinuous(key: string, value: number, fallbackMode: CcMode): void {
  const map = useMidiStore.getState().map;
  const rawName = map.cc[key];
  if (!rawName) return;
  const param = resolveCcParam(rawName);
  if (!param) return;

  const meta = SLIDER_META[param];
  // Per-channel range wins over SLIDER_META so MIDI obeys the same caps
  // as the slider widget.
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
    ?? (param.startsWith("lora_str_")
      ? LORA_SLIDER_MAX
      : stemKindFromParam(param)
        ? STEM_OVERLAY_MAX
        : 2.0);
  const reverse = range?.reverse ?? false;
  // Per-binding controller type, chosen at learn time or via the right-click
  // menu. Old maps have no stored type; existing CC maps behave as absolute,
  // and pitch-bend maps behave as centered because pitch bend rests at center.
  const mode = map.ccMode?.[key] ?? fallbackMode;

  // Spring modes anchor on the control's live default and return there on
  // release (the hardware sends its physical rest value when let go).
  if (mode === "centered") {
    applyMidiSet(param, springCentered(value, min, max, reverse, liveDefault(param)));
    return;
  }
  if (mode === "unipolar") {
    applyMidiSet(param, springUnipolar(value, min, max, reverse, liveDefault(param)));
    return;
  }

  // Knobs / faders / encoders ride the same rail curve the on-screen
  // control uses (unity only for the Voice-channel faders), so reverse is
  // applied by the mapping — exactly like the slider widget and scroll wheel.
  const m = { min, max, reverse, unity: uiUnity(param) };

  // Relative: endless encoder. One 2's-complement tick == one scroll-wheel
  // notch (MIDI_TICK_T in T-space), stepping from the param's current value.
  if (mode === "relative") {
    const ticks = decodeRelativeDelta(value);
    if (!ticks) return;
    const newT = clamp01(valueToT(readCurrent(param), m) + ticks * MIDI_TICK_T);
    applyMidiSet(param, tToValue(newT, m));
    return;
  }

  // Absolute (default): the knob/fader's physical position IS the value.
  // CC 0 → bottom of rail, 127 → top, straight through the control's mapping.
  applyMidiSet(param, tToValue(value / 127, m));
}

function handleCC(cc: number, value: number): void {
  const key = String(cc);
  if (useMidiStore.getState().applyLearn("cc", key)) return;

  const map = useMidiStore.getState().map;

  // CC-as-action bindings (pads in CC mode mapped to discrete actions
  // like seed-randomize). Fire only on the rising edge — pad presses
  // typically send a non-zero value on press and 0 on release.
  const ccAction = map.ccActions?.[key];
  if (ccAction && value > 0) {
    noteAction(ccAction);
    return;
  }

  // Enum knob bindings (DCW mode, RCFG mode, …) quantize the sweep across
  // the target's options. Checked before the continuous-param path so an
  // enum CC never falls through to setSlider.
  const enumId = map.ccEnum?.[key];
  if (enumId) {
    applyEnumCC(enumId, value);
    return;
  }

  applyContinuous(key, value, "absolute");
}

/** Pitch-bend (status 0xE0) — what joystick X-axes and pitch wheels send.
 *  14-bit, centre-resting at 8192. We fold it to a 0..127 value with the
 *  rest centre landing exactly on 64 so it rides the same `centered` spring
 *  math as a CC joystick. Bound under "pb<channel>" in the same `cc` bucket
 *  as everything else, defaulting to the `centered` type. Without this, the
 *  most common joystick/pitch-wheel axis on real hardware is invisible. */
function handlePitchBend(channel: number, lsb: number, msb: number): void {
  const key = `pb${channel}`;
  const combined = (lsb & 0x7f) | ((msb & 0x7f) << 7); // 0..16383, centre 8192
  const value =
    combined <= 8192
      ? (combined / 8192) * 64
      : 64 + ((combined - 8192) / (16383 - 8192)) * 63;
  if (useMidiStore.getState().applyLearn("cc", key)) return;
  applyContinuous(key, value, "centered");
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
  const stemKind = stemKindFromParam(param);
  if (stemKind) {
    // setVolume clamps to [0, STEM_OVERLAY_MAX]; mirror the drag path's
    // enabled = volume > 0 so sliding to zero mutes the overlay.
    const store = useStemOverlayStore.getState();
    store.setVolume(stemKind, value);
    store.setEnabled(stemKind, value > 0);
    return;
  }
  usePerformanceStore.getState().setSlider(param, value);
}

function handleNote(note: number): void {
  if (useMidiStore.getState().applyLearn("note", String(note))) return;
  const map = useMidiStore.getState().map;
  const key = String(note);
  // Enum-cycle pads advance the bound dropdown one step (wraps).
  const enumId = map.noteEnum?.[key];
  if (enumId) {
    cycleEnum(enumId);
    return;
  }
  const action = map.notes[key];
  if (!action) return;
  noteAction(action);
}

function bindInput(input: MIDIInput): void {
  input.onmidimessage = (e) => {
    const data = e.data;
    if (!data || data.length < 2) return;
    const status = data[0] & 0xf0;
    // Diagnostic for MIDI-learn debugging: when learn is active, dump
    // every raw message so we can see exactly what a control sends —
    // CC (0xb0), note (0x90/0x80), pitch bend (0xe0, joystick X / pitch
    // wheel), aftertouch (0xa0/0xd0), etc. Useful for confirming a
    // controller's actual messages on real hardware.
    if (useMidiStore.getState().learn) {
      const statusHex = (data[0] | 0).toString(16).padStart(2, "0");
      console.log(
        `[midi-learn] raw: status=0x${statusHex} data1=${data[1]} data2=${data[2] ?? "n/a"} len=${data.length}`,
      );
    }
    if (status === 0xb0) {
      // Control change.
      handleCC(data[1], data[2]);
    } else if (status === 0xe0) {
      // Pitch bend (joystick X-axis, pitch wheel). 14-bit: data1 = LSB,
      // data2 = MSB; low nibble of the status byte is the channel.
      handlePitchBend(data[0] & 0x0f, data[1], data[2] ?? 0);
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

function bindMidiLearnMenu(): () => void {
  const openFor = (
    e: MouseEvent,
    kind: "cc" | "note" | "enum",
    el: HTMLElement,
    paramTarget: string,
  ) => {
    e.preventDefault();
    useMidiStore
      .getState()
      .openMenu({ x: e.clientX, y: e.clientY, kind, target: paramTarget, el });
  };

  const onContextMenu = (e: MouseEvent) => {
    const target = e.target as HTMLElement | null;
    if (!target) return;
    const midiTarget = target.closest<HTMLElement>("[data-midi-target]");
    const midiKind = midiTarget?.dataset.midiKind as
      | "cc"
      | "note"
      | "enum"
      | undefined;
    if (midiTarget?.dataset.midiTarget && midiKind) {
      return openFor(e, midiKind, midiTarget, midiTarget.dataset.midiTarget);
    }
    const slider = target.closest<HTMLElement>(".slider-group");
    if (slider?.dataset.param) return openFor(e, "cc", slider, slider.dataset.param);
    const knob = target.closest<HTMLElement>(".knob-group");
    if (knob?.dataset.param) return openFor(e, "cc", knob, knob.dataset.param);
    const heroFader = target.closest<HTMLElement>(".hero-style-fader");
    if (heroFader?.dataset.param)
      return openFor(e, "cc", heroFader, heroFader.dataset.param);
    const learnEl = target.closest<HTMLElement>("[data-midi-learn]");
    if (learnEl?.dataset.midiLearn)
      return openFor(e, "note", learnEl, learnEl.dataset.midiLearn);
    const stemPanner = target.closest<HTMLElement>(".hero-stem-panner");
    if (stemPanner?.dataset.param)
      return openFor(e, "cc", stemPanner, stemPanner.dataset.param);
    const blendEl = target.closest<HTMLElement>("#blend-control");
    if (blendEl?.dataset.param)
      return openFor(e, "cc", blendEl, blendEl.dataset.param);
    const enumEl = target.closest<HTMLElement>("[data-midi-enum]");
    if (enumEl?.dataset.midiEnum)
      return openFor(e, "enum", enumEl, enumEl.dataset.midiEnum);
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
    document.removeEventListener("contextmenu", onContextMenu);
    document.removeEventListener("keydown", onKeyDown);
  };
}

export function useMidi() {
  // Subscribe to the user's toggle. Web MIDI's permission prompt
  // fires the moment ``requestMIDIAccess`` is called — running it on
  // mount used to interrupt every desktop visit with an OS-modal
  // "Allow MIDI?" dialog, and on mobile the prompt is pure noise
  // (no controllers, the user is on a phone). Now we wait for an
  // explicit toggle via MidiInToggle, persisted to localStorage.
  const enabled = useMidiStore((s) => s.enabled);

  useEffect(() => bindMidiLearnMenu(), []);

  useEffect(() => {
    // ``available: false`` for the lifetime of the no-MIDI state —
    // every consumer (MidiBadge etc.) reads this and renders an "off"
    // affordance accordingly.
    if (!enabled) {
      useMidiStore.getState().setAvailable(false);
      useMidiStore.getState().setStatus("MIDI off", "off");
      return;
    }
    if (typeof navigator === "undefined" || !navigator.requestMIDIAccess) {
      // The browser has no Web MIDI at all (Safari < 18, mobile
      // Safari, hostile environments). Keep the toggle visible but
      // tell the user the platform can't deliver.
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

    return () => {
      cancelled = true;
      if (access) {
        access.inputs.forEach((i) => {
          i.onmidimessage = null;
        });
        access.onstatechange = null;
      }
    };
    // Re-run on enable flip so toggling MidiInToggle off releases the
    // device bindings and toggling on re-prompts (or, if previously
    // approved, silently reconnects).
  }, [enabled]);
}
