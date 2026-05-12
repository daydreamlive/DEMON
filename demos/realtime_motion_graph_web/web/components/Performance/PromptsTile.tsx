"use client";

import { useEffect, useRef } from "react";

import { computePromptTags, usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

export function PromptsTile() {
  const promptA = usePerformanceStore((s) => s.promptA);
  const promptB = usePerformanceStore((s) => s.promptB);
  const blend = usePerformanceStore((s) => s.blend);
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const activeTimeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const setPromptA = usePerformanceStore((s) => s.setPromptA);
  const setPromptB = usePerformanceStore((s) => s.setPromptB);
  const setBlend = usePerformanceStore((s) => s.setBlend);

  function sendPrompt() {
    const remote = useSessionStore.getState().remote;
    // Server expects `tags` as the prompt string. Use computePromptTags so
    // we pick the right side as the operator drags the A↔B blend slider.
    if (remote) {
      remote.sendPrompt(
        computePromptTags({ promptA, promptB, blend }),
        activeKey,
        activeTimeSignature,
      );
    }
  }

  // Auto-submit on blend change. Debounced so dragging the slider doesn't
  // spam the server (one prompt message per ~180ms of stillness is plenty
  // for the operator's gesture to land). Initial mount skips so we don't
  // re-emit the server's own initial prompt back at it.
  const firstBlendEffect = useRef(true);
  useEffect(() => {
    if (firstBlendEffect.current) {
      firstBlendEffect.current = false;
      return;
    }
    const handle = window.setTimeout(() => {
      const remote = useSessionStore.getState().remote;
      if (!remote) return;
      remote.sendPrompt(
        computePromptTags({ promptA, promptB, blend }),
        activeKey,
        activeTimeSignature,
      );
    }, 180);
    return () => window.clearTimeout(handle);
  }, [blend, promptA, promptB, activeKey, activeTimeSignature]);

  return (
    <div className="mixer-tile mixer-tile-prompts" data-tile="prompts">
      <div className="mixer-tile-label">Tags</div>
      <div id="prompt-section">
        <div className="prompt-slot">
          <label
            className="prompt-label"
            htmlFor="prompt-a"
            data-dd-tooltip="Primary tags — text the model conditions on. With the blend at 0, these are the only tags driving the output."
            data-dd-tooltip-wide=""
          >
            Tags A
          </label>
          <textarea
            id="prompt-a"
            className="prompt-input"
            rows={2}
            value={promptA}
            onChange={(e) => setPromptA(e.target.value)}
          />
        </div>
        {/* data-param wrapper makes the right-click → MIDI learn
            handler in useMidi.ts pick this up (kind="cc",
            target="prompt_blend") without adopting slider-group
            styling. useMidi has a #blend-control branch in the
            contextmenu handler and special-cases this param to route
            writes through setBlend instead of setSlider. */}
        <div
          id="blend-control"
          data-param="prompt_blend"
          data-dd-tooltip="Crossfade between Tags A and Tags B. 0 = pure A, 1 = pure B. Hold B and use ▲▼ on desktop to nudge. Right-click to MIDI-learn."
          data-dd-tooltip-wide=""
        >
          <span className="blend-label">A</span>
          <input
            type="range"
            id="prompt-blend"
            min="0"
            max="1"
            step="0.01"
            value={blend}
            onChange={(e) => setBlend(parseFloat(e.target.value))}
          />
          <span className="blend-value" id="blend-value">
            {blend.toFixed(2)}
          </span>
          <span className="blend-label">B</span>
          <kbd className="desktop-only blend-kbd">B + ▲▼</kbd>
        </div>
        <div className="prompt-slot">
          <label
            className="prompt-label"
            htmlFor="prompt-b"
            data-dd-tooltip="Secondary tags — interpolates with A based on the blend slider. With the blend at 1, only B drives the output."
            data-dd-tooltip-wide=""
          >
            Tags B
          </label>
          <textarea
            id="prompt-b"
            className="prompt-input"
            rows={2}
            value={promptB}
            onChange={(e) => setPromptB(e.target.value)}
          />
        </div>
        <button
          id="send-prompt"
          className="send-prompt-btn"
          data-midi-learn="send_prompt"
          data-dd-tooltip="Send tags — Enter (out of textarea) or ⌘/Ctrl + Enter (in textarea); right-click to MIDI-learn"
          type="button"
          onClick={sendPrompt}
        >
          Send Tags
          <kbd className="desktop-only send-kbd">⏎</kbd>
        </button>
      </div>
    </div>
  );
}
