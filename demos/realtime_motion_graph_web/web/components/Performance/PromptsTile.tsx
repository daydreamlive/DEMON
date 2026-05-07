"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

export function PromptsTile() {
  const promptA = usePerformanceStore((s) => s.promptA);
  const promptB = usePerformanceStore((s) => s.promptB);
  const blend = usePerformanceStore((s) => s.blend);
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const setPromptA = usePerformanceStore((s) => s.setPromptA);
  const setPromptB = usePerformanceStore((s) => s.setPromptB);
  const setBlend = usePerformanceStore((s) => s.setBlend);

  function sendPrompt() {
    const remote = useSessionStore.getState().remote;
    // Server expects `tags` as the prompt string. Blend is handled via
    // params; for the prompt-text wire we just pick A (matches app.js).
    if (remote) remote.sendPrompt(promptA, activeKey);
  }

  return (
    <div className="mixer-tile mixer-tile-prompts" data-tile="prompts">
      <div className="mixer-tile-label">Prompts</div>
      <div id="prompt-section">
        <div className="prompt-slot">
          <label className="prompt-label" htmlFor="prompt-a">
            Prompt A
          </label>
          <textarea
            id="prompt-a"
            className="prompt-input"
            rows={2}
            value={promptA}
            onChange={(e) => setPromptA(e.target.value)}
          />
        </div>
        <div id="blend-control">
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
          <label className="prompt-label" htmlFor="prompt-b">
            Prompt B
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
          title="Send prompt — Enter (out of textarea) or ⌘/Ctrl + Enter (in textarea); right-click to MIDI-learn"
          type="button"
          onClick={sendPrompt}
        >
          Send Prompt
          <kbd className="desktop-only send-kbd">⏎</kbd>
        </button>
      </div>
    </div>
  );
}
