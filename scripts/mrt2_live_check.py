#!/usr/bin/env python
"""Live smoke test for the MRT2 backend against a running sidecar.

Drives MRT2Backend exactly the way PipelineRunner does (sync_source →
read_knobs → produce → render_window), with a simulated real-time
playhead, for ~12 seconds of audio: prompt A for the first stretch, a
prompt change mid-stream, and a temperature move near the end. Writes
the assembled rolling-window audio to ``out/mrt2_live_check.wav`` and
prints frontier/latency stats.

Run on the Windows side (repo venv) with the sidecar already up in the
MRT2 venv:

    (WSL)  source ~/.venvs/mrt2/bin/activate && \
           python /mnt/c/_dev/projects/DEMON/scripts/mrt2_sidecar.py
    (Win)  .venv/Scripts/python.exe scripts/mrt2_live_check.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acestep.streaming.generator_backend import TickContext  # noqa: E402
from acestep.streaming.knobs import KnobState  # noqa: E402
from acestep.streaming.mrt2 import protocol as mp  # noqa: E402
from acestep.streaming.mrt2.backend import (  # noqa: E402
    WINDOW_S,
    WINDOW_SAMPLES,
    MRT2Backend,
    mrt2_knob_specs,
)


class _State:
    prompt_text = "warm analog synthwave, steady beat"
    prompt_text_b = ""

    def __init__(self):
        self.params = {}


def main() -> int:
    run_s = 12.0
    knobs = KnobState(mrt2_knob_specs())
    backend = MRT2Backend(config=None, state=_State(), midi_knobs=knobs)

    buf = np.zeros((WINDOW_SAMPLES, mp.CHANNELS), dtype=np.float32)
    t_start = time.monotonic()
    first_audio_s = None
    writes = 0
    emitted = 0
    prompt_changed = temp_changed = False

    while True:
        wall = time.monotonic() - t_start
        if wall >= run_s:
            break
        playhead_s = wall % WINDOW_S  # simulated real-time playback
        ctx = TickContext(playhead_s=playhead_s, buffer_duration_s=WINDOW_S)

        if wall > 6.0 and not prompt_changed:
            prompt_changed = True
            print(f"[{wall:5.2f}s] prompt -> 'aggressive drum and bass'")
            backend.handle_set_prompt("aggressive drum and bass, fast breaks")
        if wall > 9.5 and not temp_changed:
            temp_changed = True
            print(f"[{wall:5.2f}s] mrt2_temperature -> 2.2")
            knobs.update({"mrt2_temperature": 2.2})

        backend.sync_source(ctx)
        raw = backend.read_knobs()
        fresh = backend.produce(raw, ctx, "generate")
        if not fresh:
            continue
        chunk = backend.render_window(playhead_s)
        while chunk is not None:
            s, e = chunk.start_sample, chunk.start_sample + chunk.pcm.shape[0]
            buf[s:e] = chunk.pcm
            writes += 1
            emitted = max(emitted, backend._abs_written)
            if first_audio_s is None and np.abs(chunk.pcm).max() > 1e-4:
                first_audio_s = wall
                print(f"[{wall:5.2f}s] first non-silent audio "
                      f"(frontier {emitted / mp.SAMPLE_RATE:.2f}s)")
            chunk = backend.render_window(playhead_s)
        backend.on_fresh_generation(raw)

    frontier_s = backend._abs_written / mp.SAMPLE_RATE
    lead_s = frontier_s - (time.monotonic() - t_start)
    print(f"done: wall={run_s:.1f}s frontier={frontier_s:.2f}s "
          f"lead={lead_s:+.2f}s writes={writes} "
          f"sidecar_lost={backend.client.lost}")
    print(f"params echo: {backend.state.params}")

    out = Path(__file__).resolve().parent.parent / "out"
    out.mkdir(exist_ok=True)
    path = out / "mrt2_live_check.wav"
    try:
        import soundfile as sf

        sf.write(str(path), buf[: int(frontier_s * mp.SAMPLE_RATE)], mp.SAMPLE_RATE)
        print(f"wrote {path}")
    except ImportError:
        from scipy.io import wavfile

        wavfile.write(
            str(path), mp.SAMPLE_RATE,
            (buf[: int(frontier_s * mp.SAMPLE_RATE)] * 32767).astype(np.int16),
        )
        print(f"wrote {path}")

    # Evaluate BEFORE close() — close marks the client lost by design.
    ok = (
        backend.client.lost is False
        and first_audio_s is not None
        and abs(lead_s) < 2.0
    )
    backend.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
