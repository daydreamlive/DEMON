"use client";

import { useEffect } from "react";

import { togglePauseAndAudio } from "@/engine/audio/togglePauseAndAudio";
import { getChannelRange } from "@/lib/config";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { DCW_MODES, DCW_WAVELETS, SLIDER_META } from "@/types/engine";

// Keyboard layout. Chord = letter held + ▲▼; single-tap keys do
// toggles/cycles/sends. Display-side source of truth lives in
// engine/keyboard/bindings.ts (rendered by ConfigModal).
//
//   A + ▲▼      remix (denoise)
//   G + ▲▼      structure (hint_strength)
//   C + ▲▼      timbre (timbre_strength)
//   B + ▲▼      prompt blend
//   E/D/H + ▲▼  feedback / feedback depth / shift (engine)
//   W/Y + ▲▼    DCW low / DCW high
//   T            toggle DCW on/off
//   Shift+T      cycle DCW mode
//   Shift+W      cycle DCW wavelet
//   0..7 + ▲▼  channel-gain ch_g0..ch_g7 (digit code, layout-independent)
//   Shift + 1..6 + ▲▼  channel sliders ch13/14/19/23/29/56
//   Enter        send prompt (when no editable focused)
//   ⌘/Ctrl+Enter inside #prompt-a/#prompt-b → send prompt
//   F            randomize seed
//   Space        pause / resume
//   K            toggle kiosk mode
//   Esc / O      toggle Advanced Controls drawer
//   R            record / stop audio (no modifiers — ⌘R still reloads)

const HELD_KEYS = new Set<string>();
const HELD_DIGITS = new Set<string>(); // KeyboardEvent.code, e.g. "Digit3"

// param mapped per chord modifier
const ENGINE_DCW_CHORDS: Record<string, string> = {
  e: "feedback",
  d: "feedback_depth",
  h: "shift",
  w: "dcw_scaler",
  y: "dcw_high_scaler",
};

const CH_GAIN_PARAMS = [
  "ch_g0",
  "ch_g1",
  "ch_g2",
  "ch_g3",
  "ch_g4",
  "ch_g5",
  "ch_g6",
  "ch_g7",
] as const;

// Channels in the order the screen presents them (top-down in the tile).
const CH_PARAMS = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"] as const;

export function useKeyboardShortcuts() {
  useEffect(() => {
    function inEditable(target: EventTarget | null): boolean {
      const el = target as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName;
      return (
        tag === "TEXTAREA" ||
        tag === "INPUT" ||
        tag === "SELECT" ||
        el.isContentEditable
      );
    }

    function bumpParam(param: string, direction: 1 | -1): void {
      const meta = SLIDER_META[param];
      const step = meta?.step ?? 0.05;
      // Reverse channels flip the arrow-key direction so ArrowUp still
      // means "drive the slider thumb UP" — which on a reverse channel
      // corresponds to a DECREASE in the engine value. Mirrors the slider
      // drag and MIDI knob behavior.
      const dirSign = getChannelRange(param)?.reverse ? -1 : 1;
      usePerformanceStore.getState().bumpSlider(param, direction * step * dirSign);
    }

    function bumpBlend(direction: 1 | -1): void {
      // prompt_blend lives in the slider system — bumpSlider gives us
      // smoothing (when enabled), clamping via SLIDER_META, and graph
      // sampling for free.
      usePerformanceStore.getState().bumpSlider("prompt_blend", direction * 0.05);
    }

    function sendPrompt(): void {
      const { promptA, promptB, activeKey, activeTimeSignature } =
        usePerformanceStore.getState();
      const remote = useSessionStore.getState().remote;
      if (remote) {
        remote.sendPrompt(promptA, activeKey, activeTimeSignature, promptB);
      }
    }

    function cycleDcwMode(): void {
      const s = usePerformanceStore.getState();
      const i = DCW_MODES.indexOf(s.dcwMode);
      const next = DCW_MODES[(i + 1) % DCW_MODES.length];
      s.setDcwMode(next);
    }
    function cycleWavelet(): void {
      const s = usePerformanceStore.getState();
      const i = DCW_WAVELETS.indexOf(s.dcwWavelet);
      const next = DCW_WAVELETS[(i + 1) % DCW_WAVELETS.length];
      s.setDcwWavelet(next);
    }

    function onKeyDown(e: KeyboardEvent) {
      // ⌘/Ctrl+Enter while in a prompt textarea sends — runs BEFORE the
      // inEditable early-return so users can fire while typing.
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        const el = e.target as HTMLElement | null;
        if (
          el?.tagName === "TEXTAREA" &&
          (el.id === "prompt-a" || el.id === "prompt-b")
        ) {
          e.preventDefault();
          sendPrompt();
          return;
        }
      }

      if (inEditable(e.target)) return;

      const k = e.key.toLowerCase();
      HELD_KEYS.add(k);
      if (e.code.startsWith("Digit")) HELD_DIGITS.add(e.code);

      if (k === "arrowup" || k === "arrowdown") {
        const dir: 1 | -1 = k === "arrowup" ? 1 : -1;

        if (HELD_KEYS.has("a")) {
          e.preventDefault();
          bumpParam("denoise", dir);
          return;
        }
        if (HELD_KEYS.has("g")) {
          e.preventDefault();
          bumpParam("hint_strength", dir);
          return;
        }
        if (HELD_KEYS.has("c")) {
          e.preventDefault();
          bumpParam("timbre_strength", dir);
          return;
        }
        if (HELD_KEYS.has("b")) {
          e.preventDefault();
          bumpBlend(dir);
          return;
        }

        for (const [letter, param] of Object.entries(ENGINE_DCW_CHORDS)) {
          if (HELD_KEYS.has(letter)) {
            e.preventDefault();
            bumpParam(param, dir);
            return;
          }
        }

        // Channels (Shift + 1..6). Checked before plain digits so Shift+1
        // routes to ch13 instead of ch_g1.
        if (e.shiftKey) {
          for (let i = 1; i <= 6; i++) {
            if (HELD_DIGITS.has(`Digit${i}`)) {
              e.preventDefault();
              bumpParam(CH_PARAMS[i - 1], dir);
              return;
            }
          }
        } else {
          for (let i = 0; i < 8; i++) {
            if (HELD_DIGITS.has(`Digit${i}`)) {
              e.preventDefault();
              bumpParam(CH_GAIN_PARAMS[i], dir);
              return;
            }
          }
        }
      }

      if (k === "enter") {
        // Enter on a focused button/link must keep its native click
        // behavior; otherwise (focus on body) treat it as send-prompt.
        const el = e.target as HTMLElement | null;
        const tag = el?.tagName;
        if (tag === "BUTTON" || tag === "A") return;
        sendPrompt();
        return;
      }
      if (k === "f") {
        usePerformanceStore.getState().randomizeSeed();
        return;
      }
      if (k === " ") {
        e.preventDefault();
        togglePauseAndAudio();
        return;
      }
      if (k === "k") {
        usePerformanceStore.getState().toggleKiosk();
        return;
      }
      // Shift cases must precede the plain T/W toggles below.
      if (k === "t" && e.shiftKey) {
        cycleDcwMode();
        return;
      }
      if (k === "w" && e.shiftKey) {
        cycleWavelet();
        return;
      }
      if (k === "t") {
        usePerformanceStore.getState().toggleDcw();
        return;
      }
      if (k === "o" || k === "escape") {
        document.dispatchEvent(new CustomEvent("dd:toggle-drawer"));
        return;
      }
      // Plain `r` toggles recording. Skip when any modifier is held so
      // ⌘R / ⌘⇧R / Ctrl+R / Alt+R remain pure browser-reload shortcuts.
      if (k === "r" && !e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        document.dispatchEvent(new CustomEvent("dd:toggle-record"));
        return;
      }
    }

    function onKeyUp(e: KeyboardEvent) {
      HELD_KEYS.delete(e.key.toLowerCase());
      if (e.code.startsWith("Digit")) HELD_DIGITS.delete(e.code);
    }
    function onBlur() {
      HELD_KEYS.clear();
      HELD_DIGITS.clear();
    }

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      HELD_KEYS.clear();
      HELD_DIGITS.clear();
    };
  }, []);
}
