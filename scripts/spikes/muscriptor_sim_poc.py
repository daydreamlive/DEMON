#!/usr/bin/env python3
"""SPIKE PoC (no audio, no DEMON model): live-MIDI integration on a real canvas.

Exercises the REAL integration path — ``_CanvasMirror`` +
``LiveMidiTranscriber`` subscribing to a real ``EventBus`` — without
loading the DEMON generation stack. A feeder thread publishes
``AudioReady`` slices that sweep a write frontier across a real 48 kHz
music canvas at real-time pace, exactly as the runner does (0.36 s
windows every ~0.04 s). The only GPU resident is MuScriptor-small
(~0.2 GB). No sounddevice, no speakers.

What it answers:
  * per-5s-window transcribe wall time through the streaming path
  * whether the self-pacing worker keeps up with a real-time frontier
  * MIDI TRAIL: how far behind the audio position each note event lands
    (t_emit - onset_position), the honest "does it feel real-time" number
  * late-% for the best-case DEMON playback lead (1.35 s)

Usage:
    uv run python scripts/spikes/muscriptor_sim_poc.py --run-for 40
    uv run python scripts/spikes/muscriptor_sim_poc.py --model small --stride 1.5
"""

import argparse
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

SR = 48000
WIN = 0.36          # runner slice width (s)
SLICE_DT = 0.04     # runner inter-slice interval (s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="small", choices=("small", "medium", "large"))
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--guard", type=float, default=0.3)
    p.add_argument("--run-for", type=float, default=40.0)
    p.add_argument("--fixture", default="inside_confusion_loop_60s_gsm.wav")
    args = p.parse_args()

    from acestep.fixtures import audio_fixture
    from acestep.streaming.events import AudioReady, EventBus
    from acestep.analysis.midi_live import LiveMidiTranscriber

    import soundfile as sf
    path = str(audio_fixture(args.fixture))
    canvas, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        import librosa
        canvas = librosa.resample(canvas.T, orig_sr=sr, target_sr=SR).T
    N = canvas.shape[0]
    dur = N / SR
    print(f"[poc] canvas: {path.split(os.sep)[-1]}  {dur:.1f}s  {canvas.shape}")

    # ---- real bus + shim session (only the fields the transcriber uses) ----
    bus = EventBus()
    session = SimpleNamespace(
        initial_buffer=canvas.copy(),
        bus=bus,
        audio_eng=None,   # unused: we inject a virtual playhead
    )

    # Collected note observations: (t_emit_rel, onset_canvas_s, pitch, instr)
    notes: list = []
    t0_holder = {"t0": None}

    def on_note(ev, onset_canvas_s):
        t0 = t0_holder["t0"]
        if t0 is None:
            return
        notes.append((time.monotonic() - t0, onset_canvas_s, ev.pitch, ev.instrument))

    trans = LiveMidiTranscriber(
        session,
        model_size=args.model,
        stride_s=args.stride,
        guard_s=args.guard,
        midi_port=None,          # no MIDI port -> no scheduling, on_note only
        on_note=on_note,
    )

    # ---- feeder: sweep the frontier at real time ----
    stop = threading.Event()

    def feeder():
        t0 = t0_holder["t0"]
        w = int(WIN * SR)
        while not stop.is_set():
            elapsed = time.monotonic() - t0
            frontier = int(elapsed * SR)      # unwrapped, real-time
            start = frontier % N
            seg = np.empty((w, canvas.shape[1]), np.float32)
            end = start + w
            if end <= N:
                seg[:] = canvas[start:end]
            else:
                k = N - start
                seg[:k] = canvas[start:]
                seg[k:] = canvas[: end - N]
            bus.publish(AudioReady(
                audio=seg, start_sample=start, num_samples=w,
                channels=canvas.shape[1], tick_ms=0.0, dec_ms=0.0,
                num_gens=1, params={}, published_wall_s=time.monotonic(),
            ))
            time.sleep(SLICE_DT)

    t0 = time.monotonic()
    t0_holder["t0"] = t0
    trans.attach()
    ft = threading.Thread(target=feeder, name="feeder", daemon=True)
    ft.start()
    print(f"[poc] running {args.run_for:.0f}s, model={args.model} "
          f"stride={args.stride}s guard={args.guard}s (no audio)", flush=True)

    try:
        while time.monotonic() - t0 < args.run_for:
            time.sleep(5.0)
            print(f"[poc] {trans.stats.summary()}  emitted_notes={len(notes)}",
                  flush=True)
    finally:
        stop.set()
        trans.detach()

    # ---- report ----
    print("\n" + "=" * 68)
    print(f"POC RESULT  model={args.model}  stride={args.stride}s")
    print("=" * 68)
    w = np.array(trans.stats.window_wall_s[1:] or [0.0])   # drop warmup window
    fe = np.array(trans.stats.first_event_s or [0.0])
    print(f"windows transcribed : {trans.stats.windows}  "
          f"(over {args.run_for:.0f}s wall)")
    print(f"per-window wall (s) : mean={w.mean():.3f}  p50={np.median(w):.3f}  "
          f"p95={np.percentile(w,95):.3f}  max={w.max():.3f}")
    print(f"effective RTF       : {w.mean()/5.0:.3f}  "
          f"(5s of audio per window)")
    print(f"first-event lat (s) : mean={fe.mean():.3f}  "
          f"(window start -> first note out)")
    print(f"notes emitted       : {len(notes)}")

    if notes:
        arr = np.array([(te, oc) for te, oc, _, _ in notes])
        # first pass only (run-for < canvas dur): onset canvas s == frontier
        # wall time, so trail = t_emit - onset position.
        trail = arr[:, 0] - arr[:, 1]
        trail = trail[(trail > -1.0) & (trail < 30.0)]   # guard against wrap
        print(f"\nMIDI TRAIL behind audio position (s):")
        print(f"  mean={trail.mean():.2f}  p50={np.median(trail):.2f}  "
              f"p95={np.percentile(trail,95):.2f}  max={trail.max():.2f}")
        for lead in (0.5, 1.0, 1.35):
            late = float((trail > lead).mean()) * 100.0
            print(f"  late-% @ playback-lead {lead:>4.2f}s : {late:5.1f}%  "
                  f"(note event arrives after the ear at that lead)")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
