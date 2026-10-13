"""GPU validation of the streaming MuScriptor decoder (run on Ryan's say-so).

Loads a MuScriptor model, streams a real wav through
:class:`~acestep.analysis.notes_stream.StreamingNotesDecoder` in
``--frame-ms`` frames as a live capture would arrive, runs one hop every
``--hop-ms`` over the newest 5 s behind the frontier, and prints per hop:
wall, prompt tokens, new tokens, notes committed, and per note the
onset-to-emission latency (frontier at commit minus onset, plus the hop's
wall: what a listener would see if the hop ran in real time). Ends with
p50/p95 of hop wall and latency and, with ``--compare-batch``, the note
count of the batch ``transcribe`` on the same file.

``--dry-run`` reads the wav and prints the hop plan without importing
torch or loading anything (the CPU check).

Usage:
  python scripts/analysis/notes_stream_probe.py --wav path.wav [--size small]
      [--frame-ms 100] [--hop-ms 200] [--max-s 30] [--instruments a,b]
      [--pad-mode mask|short] [--guard-s 0.15] [--cap 256] [--budget-s 0.5]
      [--compare-batch] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acestep.analysis import notes_stream as ns  # noqa: E402  (torch-free)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--wav", required=True, type=Path, help="audio file to stream")
    p.add_argument("--size", default="small", choices=("small", "medium", "large"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--frame-ms", type=float, default=100.0, help="capture frame size")
    p.add_argument("--hop-ms", type=float, default=200.0, help="decoder hop")
    p.add_argument("--max-s", type=float, default=30.0, help="stream at most this much audio")
    p.add_argument("--instruments", default="", help="comma-separated MuScriptor group names")
    p.add_argument("--pad-mode", default="mask", choices=("mask", "short"))
    p.add_argument("--guard-s", type=float, default=ns.GUARD_S)
    p.add_argument("--cap", type=int, default=ns.NEW_TOKEN_CAP, help="new tokens per hop")
    p.add_argument("--budget-s", type=float, default=ns.HOP_WALL_BUDGET_S, help="wall per hop")
    p.add_argument("--compare-batch", action="store_true", help="also run batch transcribe")
    p.add_argument("--dry-run", action="store_true", help="hop plan only, no torch, no model")
    a = p.parse_args(argv)
    if a.frame_ms <= 0 or a.hop_ms <= 0 or a.max_s <= 0:
        p.error("--frame-ms, --hop-ms and --max-s must be positive")
    if a.hop_ms < a.frame_ms:
        p.error("--hop-ms must be at least --frame-ms")
    if not a.wav.is_file():
        p.error(f"no such file: {a.wav}")
    a.instruments = [s.strip() for s in a.instruments.split(",") if s.strip()]
    return a


def load_mono(path: Path, max_s: float) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    return audio[: int(max_s * sr)], int(sr)


def hop_plan(n_samples: int, sr: int, frame_ms: float, hop_ms: float):
    """``(window_start_s, frontier_s)`` per hop: the frontier grows by
    whole frames, a hop runs once ``hop_ms`` of new audio has arrived."""
    frame = int(round(sr * frame_ms / 1000.0))
    per_hop = max(1, int(round(hop_ms / frame_ms)))
    plan = []
    k = 0
    while True:
        k += per_hop
        end = min(k * frame, n_samples)
        f = end / sr
        plan.append((max(0.0, f - ns.WINDOW_S), f))
        if end >= n_samples:
            return plan


def pct(xs, q):
    return float(np.percentile(np.asarray(xs), q)) if xs else float("nan")


def main(argv=None) -> int:
    a = parse_args(argv)
    audio, sr = load_mono(a.wav, a.max_s)
    plan = hop_plan(len(audio), sr, a.frame_ms, a.hop_ms)
    print(f"wav={a.wav} sr={sr} dur_s={len(audio) / sr:.2f} hops={len(plan)} "
          f"frame_ms={a.frame_ms} hop_ms={a.hop_ms} pad_mode={a.pad_mode}")
    if a.dry_run:
        for ws, f in plan[:5]:
            print(f"  window [{ws:.2f}, {f:.2f})")
        print("dry run: no model loaded")
        return 0

    from muscriptor.transcription_model import TranscriptionModel

    t0 = time.perf_counter()
    tm = TranscriptionModel.load_model(a.size, device=a.device, dtype="float16")
    print(f"loaded {a.size} in {time.perf_counter() - t0:.1f}s")
    backend = ns.MuScriptorStreamBackend(tm, pad_mode=a.pad_mode)
    dec = ns.StreamingNotesDecoder(backend, new_token_cap=a.cap, guard_s=a.guard_s,
                                   hop_wall_budget_s=a.budget_s)
    instruments = a.instruments or None
    walls, latencies, total, truncated = [], [], 0, 0
    print("hop  frontier  wall_ms  prompt  new  notes  open  trunc")
    for i, (ws, f) in enumerate(plan):
        s0, s1 = int(round(ws * sr)), int(round(f * sr))
        r = dec.step(audio[s0:s1], sr, ws, f, instruments)
        walls.append(r.wall_s)
        truncated += int(r.truncated)
        total += len(r.new_notes)
        for n in r.new_notes:
            latencies.append(f - n.start_s + r.wall_s)
        print(f"{i:4d}  {f:8.2f}  {r.wall_s * 1000:7.1f}  {r.prompt_tokens:6d}  "
              f"{r.new_tokens:3d}  {len(r.new_notes):5d}  {len(r.open_notes):4d}  "
              f"{r.reason or ''}")
        for n in r.new_notes:
            print(f"        note pitch={n.pitch:3d} {n.instrument:24s} onset={n.start_s:7.2f} "
                  f"latency_ms={(f - n.start_s + r.wall_s) * 1000:6.0f}")
    print(f"notes={total} truncated_hops={truncated}/{len(plan)}")
    print(f"hop wall ms p50={pct(walls, 50) * 1000:.1f} p95={pct(walls, 95) * 1000:.1f} "
          f"max={max(walls) * 1000:.1f}")
    print(f"onset->emission ms p50={pct(latencies, 50) * 1000:.0f} "
          f"p95={pct(latencies, 95) * 1000:.0f}")
    if a.compare_batch:
        import torch

        t0 = time.perf_counter()
        wav = torch.from_numpy(audio)[None, :]
        n_batch = sum(1 for ev in tm.transcribe((wav, sr), instruments=instruments)
                      if type(ev).__name__ == "NoteStartEvent")
        print(f"batch transcribe notes={n_batch} wall_s={time.perf_counter() - t0:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
