"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import { RecordToggle } from "./RecordToggle";
import { SliderGroup } from "./SliderGroup";

interface Props {
  onOpenAllControls: () => void;
}

// Mobile-first "Lite" mixer. Remix is also exposed here (in addition to
// the always-visible left-edge rail) so users who already have the drawer
// open can adjust it without reaching back to the rail. Both surfaces
// bind to the same zustand value, so they stay in sync automatically.
export function LiteControls({ onOpenAllControls }: Props) {
  const seed = usePerformanceStore((s) => s.seed);
  const promptA = usePerformanceStore((s) => s.promptA);
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const randomize = usePerformanceStore((s) => s.randomizeSeed);
  const setPromptA = usePerformanceStore((s) => s.setPromptA);

  function sendPrompt() {
    const remote = useSessionStore.getState().remote;
    if (remote) remote.sendPrompt(promptA, activeKey);
  }

  return (
    <div className="lite-controls">
      <div className="lite-row lite-row--main">
        <SliderGroup param="denoise" label="remix" />
        <SliderGroup param="hint_strength" label="structure" />
        <button
          type="button"
          className="lite-seed-btn"
          data-midi-learn="seed"
          onClick={randomize}
          aria-label="Randomize seed"
        >
          <span className="lite-seed-icon" aria-hidden="true">
            <svg
              className="seed-dice"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="3" />
              <circle cx="8.5" cy="8.5" r="1.1" fill="currentColor" stroke="none" />
              <circle cx="15.5" cy="15.5" r="1.1" fill="currentColor" stroke="none" />
            </svg>
          </span>
          <span className="lite-seed-value">{seed.toFixed(2)}</span>
        </button>
      </div>

      <div className="lite-row lite-row--prompt">
        <input
          type="text"
          className="lite-prompt-input"
          value={promptA}
          onChange={(e) => setPromptA(e.target.value)}
          placeholder="Describe a sound..."
          aria-label="Prompt"
        />
        <button
          type="button"
          className="lite-send-btn"
          data-midi-learn="send_prompt"
          onClick={sendPrompt}
        >
          Send
        </button>
        <RecordToggle />
      </div>

      <button
        type="button"
        className="lite-all-controls"
        onClick={onOpenAllControls}
      >
        All controls
        <span className="lite-all-controls-arrow" aria-hidden="true">→</span>
      </button>
    </div>
  );
}
