"use client";

import { swapCheckpointWhileLive } from "@/hooks/useStartSession";
import { showCheckpointPicker } from "@/lib/servedCheckpoints";
import { useSessionStore } from "@/store/useSessionStore";

// The Model select: which of the pod's served checkpoints the session
// runs. One component, two placements (start overlay before the first
// Start, Config tab after), both bound to useSessionStore's
// selectedCheckpoint. Renders nothing unless the pod serves two or more,
// so single-model pods look exactly as before. Picking the default entry
// stores null, which omits `checkpoint` from the wire. While idle a pick
// is only recorded (the next Start reads it); while live it restarts the
// session on that model.
export function ModelSelect({ id }: { id: string }) {
  const served = useSessionStore((s) => s.servedCheckpoints);
  const selected = useSessionStore((s) => s.selectedCheckpoint);
  const status = useSessionStore((s) => s.status);
  if (!showCheckpointPicker(served)) return null;

  const def = served.find((c) => c.default) ?? served[0];
  const value = selected ?? def.name;
  const busy =
    status === "loading-fixture" ||
    status === "connecting" ||
    status === "reconnecting";

  return (
    <select
      id={id}
      className="fixture-select"
      aria-label="Model"
      value={value}
      disabled={busy}
      onChange={(e) => {
        const name = e.target.value;
        void swapCheckpointWhileLive(name === def.name ? null : name);
      }}
    >
      {served.map((c) => (
        <option key={c.name} value={c.name}>
          {c.default ? `${c.name} (default)` : c.name}
        </option>
      ))}
    </select>
  );
}

// Config-tab placement: section label, the select, one line of help.
export function ModelPicker() {
  const served = useSessionStore((s) => s.servedCheckpoints);
  if (!showCheckpointPicker(served)) return null;
  return (
    <section className="operator-section">
      <h3 className="operator-section-label">Model</h3>
      <div className="operator-row">
        <ModelSelect id="checkpoint-select" />
      </div>
      <p className="operator-help">
        Changing it restarts the session on that model; track, prompt, knobs and LoRAs carry over.
      </p>
    </section>
  );
}

// Start-overlay placement: label and the select, below the Start mark.
export function StartModelPicker() {
  const served = useSessionStore((s) => s.servedCheckpoints);
  if (!showCheckpointPicker(served)) return null;
  return (
    <div className="start-model">
      <label className="operator-section-label" htmlFor="start-checkpoint-select">
        Model
      </label>
      <ModelSelect id="start-checkpoint-select" />
    </div>
  );
}
