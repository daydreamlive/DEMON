#!/usr/bin/env python3
"""SPIKE driver: DEMON streaming session + live MuScriptor MIDI.

Creates a local-playback streaming session on a fixture (or supplied
audio), attaches :class:`acestep.analysis.midi_live.LiveMidiTranscriber`
to the session bus, and runs until Ctrl+C. Prints per-window transcribe
wall time, late-note counts, and tick_ms so the contention question is
answered by the same run that demos the feel.

MIDI out (optional): create a loopMIDI port and pass --midi-port; point
an Ableton track at it. Without a port the spike logs notes only.

Setup (spike-only deps, not in pyproject):
    uv pip install muscriptor python-rtmidi
    # weights are gated: accept the license at
    # https://huggingface.co/MuScriptor/muscriptor-small first

Usage:
    uv run python scripts/spikes/muscriptor_live_midi.py
    uv run python scripts/spikes/muscriptor_live_midi.py \
        --model medium --stride 1.0 --midi-port "demon-midi 1"
    uv run python scripts/spikes/muscriptor_live_midi.py --list-ports
"""

import argparse
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="small",
                   choices=("small", "medium", "large"))
    p.add_argument("--stride", type=float, default=1.0,
                   help="seconds of frontier advance per transcribe window")
    p.add_argument("--guard", type=float, default=0.3,
                   help="right-context guard before emitting an onset")
    p.add_argument("--midi-port", default=None,
                   help="mido output port name (see --list-ports)")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--fixture", default="inside_confusion_loop_60s_gsm.wav")
    p.add_argument("--checkpoint", default=None,
                   help="default: server default checkpoint resolution")
    p.add_argument("--decoder-accel", default="tensorrt")
    p.add_argument("--vae-accel", default="tensorrt")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--prompt", default="instrumental music")
    p.add_argument("--baseline-s", type=float, default=20.0,
                   help="run this long WITHOUT the transcriber first, for a "
                        "tick_ms baseline in the same process")
    args = p.parse_args()

    if args.list_ports:
        import mido
        print("\n".join(mido.get_output_names()) or "(no MIDI output ports)")
        return 0

    import numpy as np
    import torch  # noqa: F401  (CUDA init before session)

    from acestep.fixtures import audio_fixture
    from acestep.nodes import Audio
    from acestep.streaming import registry
    from acestep.streaming.config import SessionConfig
    from acestep.streaming.session import StreamingSession
    from acestep.streaming.source import SAMPLE_RATE, _load_known_fixture_waveform

    from acestep.analysis.midi_live import LiveMidiTranscriber

    # ---- session -----------------------------------------------------
    waveform = _load_known_fixture_waveform(args.fixture)
    cfg = SessionConfig.from_dict({
        "prompt": args.prompt, "steps": args.steps, "depth": args.depth,
    })
    checkpoint = args.checkpoint or os.environ.get(
        "ACESTEP_CHECKPOINT", "ACE-Step-v1-3.5B"
    )
    print(f"[spike] creating session checkpoint={checkpoint} "
          f"decoder={args.decoder_accel} vae={args.vae_accel}", flush=True)
    streaming = StreamingSession.create(
        audio=Audio(waveform=waveform, sample_rate=SAMPLE_RATE),
        config=cfg,
        checkpoint=checkpoint,
        decoder_backend=args.decoder_accel,
        vae_backend=args.vae_accel,
        session_id=registry.new_session_id(),
    )

    # Local playback so the feel test is audible without a browser.
    try:
        streaming.audio_eng.start()
    except Exception as exc:
        print(f"[spike] local playback unavailable ({exc}); continuing "
              f"headless (late-note metric needs a moving playhead!)")

    # ---- tick loop on a thread, spike logic on main ------------------
    import threading
    run_t = threading.Thread(target=streaming.run, name="runner", daemon=True)
    run_t.start()

    # Baseline tick_ms without the transcriber attached (same process,
    # same session — the honest control for the contention number).
    from acestep.streaming.events import AudioReady
    baseline: list = []
    sub = streaming.bus.subscribe(
        lambda ev: baseline.append(ev.tick_ms)
        if isinstance(ev, AudioReady) else None,
        name="tick_baseline",
    )
    print(f"[spike] {args.baseline_s:.0f}s tick baseline...", flush=True)
    time.sleep(args.baseline_s)
    streaming.bus.unsubscribe(sub)
    if baseline:
        b = np.array(baseline)
        print(f"[spike] baseline tick_ms: mean={b.mean():.1f} "
              f"p95={np.percentile(b, 95):.1f} n={len(b)}", flush=True)

    # ---- transcriber -------------------------------------------------
    trans = LiveMidiTranscriber(
        streaming,
        model_size=args.model,
        stride_s=args.stride,
        guard_s=args.guard,
        midi_port=args.midi_port,
    )
    trans.attach()
    print("[spike] transcriber attached — Ctrl+C to stop", flush=True)

    try:
        while True:
            time.sleep(10.0)
            print(f"[spike] {trans.stats.summary()}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        trans.detach()
        print(f"\n[spike] FINAL {trans.stats.summary()}")
        if baseline:
            print(f"[spike] baseline tick_ms mean={np.array(baseline).mean():.1f} "
                  f"(compare against tick_ms above for contention)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
