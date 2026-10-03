"""M2: drive a live ``--checkpoint yue2-3b`` server over the wire.

Black-box, like the golden client it reuses: connects as the demo page
does (config + PCM stub), plays a simulated real-time playhead that
wraps at the song end (the loop), and walks the knobs through phases:

  anchor    defaults (seed = song seed, d = 1.0): ring re-renders the anchor
  seed      seed 1: new acoustic noise, same composition
  denoise   seed 1 + yue2_denoise 0.5: re-noise the anchor, last 16 steps
  x0        seed 1 + yue2_denoise 1.0 + x0_target 0.5: pull toward the anchor
  style     seed 0 + style prompt change (fast restyle, frozen semantics)

Each phase plays ``--loops`` full passes of the song so every window is
re-decoded under the phase's knobs, then the mirrored buffer is saved as
a wav. Reports slice tick/decode p50/p95, slices per phase, RMS of each
wav and of its difference to the anchor, nvidia-smi peak memory, and the
ready time. Writes one JSON next to the wavs.

    python scripts/spikes/yue2_ring_drive.py --url ws://localhost:1318 \
        --duration 30 --out notes/family_merge/yue2_out/ring/m2_30s
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tests.golden.client import SAMPLE_RATE, GoldenClient

PROMPT = "city pop, female vocal, bright synths, groovy bass"
PROMPT_STYLE = "dark industrial techno, distorted kick, male vocal"
PARAMS_TICK_S = 0.1


class GpuSampler(threading.Thread):
    """nvidia-smi memory.used (MiB) once a second; keeps the samples."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples: list[int] = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                self.samples.append(int(out.strip().splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def save_wav(path: Path, audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


class Playhead:
    """Real-time playback position that wraps at the buffer end."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.t0 = time.monotonic()

    def position(self) -> float:
        return (time.monotonic() - self.t0) % self.seconds

    def wraps(self) -> int:
        return int((time.monotonic() - self.t0) // self.seconds)


def play(client: GoldenClient, playhead: Playhead, values: dict, seconds: float) -> list:
    """Stream params for ``seconds`` while pumping; returns the slices
    received in that span."""
    first = len(client.slices)
    end = time.monotonic() + seconds
    next_params = 0.0
    while time.monotonic() < end:
        now = time.monotonic()
        if now >= next_params:
            client.send_params(values, playhead.position())
            next_params = now + PARAMS_TICK_S
        client.pump(timeout=0.02)
    return client.slices[first:]


def phase_report(slices, audio, anchor):
    # Settled slices re-decode the cached latent and carry a ~0 ms tick;
    # only ticks that ran the ring count.
    ticks = [s.tick_ms for s in slices if s.tick_ms > 1.0]
    decs = [s.dec_ms for s in slices if s.dec_ms > 0]
    diff = audio - anchor
    return {
        "slices": len(slices), "ring_ticks": len(ticks),
        "tick_ms_p50": percentile(ticks, 50), "tick_ms_p95": percentile(ticks, 95),
        "dec_ms_p50": percentile(decs, 50), "dec_ms_p95": percentile(decs, 95),
        "rms": rms(audio), "rms_diff_vs_anchor": rms(diff),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="ws://localhost:1318")
    parser.add_argument("--duration", type=float, default=30.0, help="yue2_duration_s")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--loops", type=float, default=1.15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gpu = GpuSampler()
    gpu.start()
    t_connect = time.monotonic()
    client = GoldenClient(args.url)
    client.send_config({
        "telemetry_version": 1, "backend": "yue2", "prompt": PROMPT, "prompt_b": PROMPT,
        "yue2_duration_s": args.duration, "depth": args.depth,
    })
    client.send_pcm(np.zeros((9600, 2), np.float32), 2)
    ready = client.wait_ready(timeout=600)
    report = {"ready_s": time.monotonic() - t_connect, "duration_s": ready["duration"],
              "pipeline_depth": ready["pipeline_depth"],
              "knobs": sorted(ready["knob_manifest"]["knobs"]), "phases": {}}
    anchor = client.buffer.copy()
    save_wav(args.out / "anchor_initial.wav", anchor)
    report["anchor_rms"] = rms(anchor)
    print(f"ready in {report['ready_s']:.1f}s, song {ready['duration']:.2f}s, anchor rms {report['anchor_rms']:.4f}",
          flush=True)

    defaults = {k: v.get("default") for k, v in ready["knob_manifest"]["knobs"].items()
                if v.get("type") in ("float", "int")}
    playhead = Playhead(ready["duration"])
    span = ready["duration"] * args.loops
    phases = [
        ("anchor", dict(defaults), None),
        ("seed", {**defaults, "seed": 1}, None),
        ("denoise", {**defaults, "seed": 1, "yue2_denoise": 0.5}, None),
        ("x0", {**defaults, "seed": 1, "x0_target": 0.5}, None),
        ("style", dict(defaults), PROMPT_STYLE),
    ]
    for name, values, prompt in phases:
        if prompt is not None:
            client.send_prompt(prompt, prompt)
        slices = play(client, playhead, values, span)
        audio = client.buffer.copy()
        save_wav(args.out / f"{name}.wav", audio)
        rep = phase_report(slices, audio, anchor)
        rep["values"] = values
        rep["wraps_so_far"] = playhead.wraps()
        report["phases"][name] = rep
        print(name, json.dumps({k: v for k, v in rep.items() if k != "values"}), flush=True)

    errors = [d for _, d in client.events if d.get("type") == "error"]
    report["errors"] = errors
    client.close()
    gpu.stop()
    report["nvidia_smi_mib_max"] = max(gpu.samples) if gpu.samples else None
    report["nvidia_smi_mib_min"] = min(gpu.samples) if gpu.samples else None
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "phases"}, indent=2))


if __name__ == "__main__":
    main()
