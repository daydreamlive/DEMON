"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

export function PromptsTile() {
  const promptA = usePerformanceStore((s) => s.promptA);
  const promptB = usePerformanceStore((s) => s.promptB);
  // Read sliderTargets (instant) so the thumb tracks the cursor without
  // the smoothing lag — the engine sees the smoothed sliderValues via
  // usePromptBlendSync.
  const blend = usePerformanceStore(
    (s) => s.sliderTargets.prompt_blend ?? 0,
  );
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const activeTimeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const setPromptA = usePerformanceStore((s) => s.setPromptA);
  const setPromptB = usePerformanceStore((s) => s.setPromptB);
  const setSlider = usePerformanceStore((s) => s.setSlider);

  // Send Tags is the only path that pays the server-side text encoder:
  // it ships both A and B so the backend caches a cond pair for each,
  // and the blend slider then lerps between them per tick via the cheap
  // set_prompt_blend channel (usePromptBlendSync). Editing the
  // textareas does NOT auto-submit — the operator decides when to
  // commit new tags.
  function sendPrompt() {
    const remote = useSessionStore.getState().remote;
    if (remote) {
      remote.sendPrompt(promptA, activeKey, activeTimeSignature, promptB);
    }
  }

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
            styling. MIDI writes flow through the generic setSlider
            path now that prompt_blend lives in sliderValues. */}
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
            onChange={(e) => setSlider("prompt_blend", parseFloat(e.target.value))}
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
