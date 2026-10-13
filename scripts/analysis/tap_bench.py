"""No-model simulator for the live analysis tap (CPU only, no GPU).

Feeds a real 48 kHz fixture through a real :class:`AudioEngine` and
:class:`EventBus`, simulates the generator's writes (ring refinement ahead
of the playhead, or an append-only frontier) and a client playhead, and
drives a real :class:`AnalysisTap` against a fake host whose engine takes
``--latency`` seconds per cell and returns one note per cell. Reports
per-cell wall time, how far ahead of (positive) or behind (negative) the
playhead each update landed, re-analysis counts and dropped counts.

A ring run has two phases: churn (each write carries different music, as
under a moving denoise or prompt) for the first ``--steady-after``
seconds, then steady (each write carries the audio already there, as when
the generator re-renders the same music lap after lap). The observer's
content check keeps steady cells clean, so settled updates appear there.

Usage::

    python scripts/analysis/tap_bench.py --seconds 20 --latency 0.15
    python scripts/analysis/tap_bench.py --seconds 10 --steady-after 4
    python scripts/analysis/tap_bench.py --append-only --latency 0.3
    python scripts/analysis/tap_bench.py --audio path/to/48k.wav
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acestep.analysis.engine import AnalysisResult  # noqa: E402
from acestep.analysis.registry import AnalyzerOption, AnalyzerSpec  # noqa: E402
from acestep.analysis.tap import AnalysisTap  # noqa: E402
from acestep.streaming.audio_engine import AudioEngine  # noqa: E402
from acestep.streaming.events import (  # noqa: E402
    AnalysisState,
    AnalysisUpdate,
    AudioReady,
    EventBus,
)

SR = 48000
DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"

BENCH_SPEC = AnalyzerSpec(
    name="bench_notes",
    display_name="Bench notes",
    item_kind="note",
    make_engine=lambda size, device: None,
    probe=lambda: None,
    sizes=("small",),
    default_size="small",
    window_s=4.0,
    context_s=0.5,
    placement="cpu",
    options=(AnalyzerOption("horizon_s", "float", default=8.0, bounds=(4.0, 30.0)),),
)


class FakeHost:
    """Single worker, bounded queue (drop-oldest), fixed latency."""

    def __init__(self, latency_s: float, capacity: int = 8):
        self.latency_s = latency_s
        self.capacity = capacity
        self._q: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._stop = False
        self.walls: list = []
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, analyzer, size, request, on_result, on_error, cancel_token):
        with self._cv:
            if len(self._q) >= self.capacity:
                _r, _ok, err, _t = self._q.popleft()
                from acestep.analysis.engine import EngineBusy

                err(EngineBusy("dropped"))
            self._q.append((request, on_result, on_error, cancel_token))
            self._cv.notify()

    def cancel(self, cancel_token):
        with self._cv:
            before = len(self._q)
            self._q = collections.deque(x for x in self._q if x[3] is not cancel_token)
            return before - len(self._q)

    def engine_info(self, analyzer, size):
        return {"name": "fake", "size": size or "small", "loaded": True, "licence": ""}

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                request, on_result, _err, _tok = self._q.popleft()
            t0 = time.monotonic()
            time.sleep(self.latency_s)
            wall = time.monotonic() - t0
            self.walls.append(wall)
            on_result(AnalysisResult(
                kind="note", range_start_s=request.t0_s, span_s=request.span_s,
                items=[{"start_s": request.t0_s + 0.25, "end_s": request.t0_s + 0.5,
                        "pitch": 60, "instrument": "piano"}],
                source_epoch=request.source_epoch, revision=request.revision,
                wall_s=wall,
            ))


def _load(path: str | None) -> np.ndarray:
    import soundfile as sf

    if path is None:
        from acestep.fixtures import audio_fixture

        path = str(audio_fixture(DEFAULT_FIXTURE))
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path}: need 48 kHz audio, got {sr}")
    return audio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audio", help="48 kHz audio file (default: a 60 s fixture)")
    ap.add_argument("--seconds", type=float, default=20.0, help="simulated wall time")
    ap.add_argument("--latency", type=float, default=0.15, help="fake engine seconds per cell")
    ap.add_argument("--horizon", type=float, default=8.0)
    ap.add_argument("--append-only", action="store_true", help="simulate an append-only canvas")
    ap.add_argument("--write-every", type=float, default=0.25, help="seconds between generator writes")
    ap.add_argument("--write-span", type=float, default=2.0, help="seconds per generator write")
    ap.add_argument("--steady-after", type=float, default=None,
                    help="ring: seconds of churn before writes repeat the canvas (default: half the run)")
    args = ap.parse_args()
    steady_after = args.seconds / 2.0 if args.steady_after is None else args.steady_after
    rng = np.random.default_rng(0)

    src = _load(args.audio)
    canvas = src.copy() if not args.append_only else np.zeros_like(src)
    eng = AudioEngine(canvas, SR)
    bus = EventBus()
    host = FakeHost(args.latency)
    updates: list = []
    states: list = []

    def on_event(ev):
        if isinstance(ev, AnalysisUpdate) and not ev.clear:
            updates.append((ev, time.monotonic() - t_start))
        elif isinstance(ev, AnalysisState):
            states.append(ev)

    bus.subscribe(on_event, name="bench")
    seq = iter(range(1, 10 ** 9))
    tap = AnalysisTap(
        spec=BENCH_SPEC, options=BENCH_SPEC.validate_options({"horizon_s": args.horizon}),
        submitter=host, audio_eng=eng, bus=bus, refines_audio=not args.append_only,
        source_epoch=0, next_seq=lambda: next(seq),
    )
    duration = len(src) / SR
    t_start = time.monotonic()
    steady_at = t_start + steady_after
    tap.start()
    frontier = 0.0
    lead_s = 1.5  # ring writes land this far ahead of the playhead
    next_write = t_start
    while time.monotonic() - t_start < args.seconds:
        now = time.monotonic()
        playhead = (now - t_start) % duration
        eng.position = int(playhead * SR)
        eng.position_report_wall_s = now
        if now >= next_write:
            next_write += args.write_every
            if args.append_only:
                end = min(duration, (now - t_start) + 3.0)
                s0, s1 = int(frontier * SR), int(end * SR)
                if s1 > s0:
                    eng.patch_window(src[s0:s1], s0)
                    frontier = end
            else:
                start = (playhead + lead_s) % duration
                s0 = int(start * SR)
                s1 = min(len(src), s0 + int(args.write_span * SR))
                if now < steady_at:
                    # Churn: different music from elsewhere in the source.
                    offset = int(rng.integers(SR, len(src) - SR))
                    block = np.take(src, np.arange(s0, s1) + offset, axis=0, mode="wrap")
                else:
                    block = src[s0:s1]
                eng.patch_window(block, s0)
            bus.publish(AudioReady(
                audio=np.zeros((1, 2), np.float32), start_sample=0, num_samples=1,
                channels=2, tick_ms=80.0 + 5.0 * np.random.rand(), dec_ms=5.0,
                num_gens=0, params={},
            ))
        time.sleep(0.02)
    tap.close()
    host.close()
    bus.close()

    per_cell = collections.Counter(int(u.range_start_s // BENCH_SPEC.window_s) for u, _ in updates
                                   if u.basis != "settled" or args.append_only)
    leads = [((u.range_start_s - u.playhead_s + duration / 2) % duration) - duration / 2
             for u, _ in updates if u.playhead_s is not None]
    print(f"canvas={'append-only' if args.append_only else 'ring'} seconds={args.seconds} "
          f"latency={args.latency} horizon={args.horizon}")
    if host.walls:
        print(f"wall per cell: median {statistics.median(host.walls):.3f}s max {max(host.walls):.3f}s "
              f"n={len(host.walls)}")
    print(f"updates={len(updates)} settled={sum(1 for u, _ in updates if u.basis == 'settled')} "
          f"late={sum(1 for u, _ in updates if u.late)}")
    if not args.append_only:
        for label, lo, hi in (("churn", 0.0, steady_after), ("steady", steady_after, args.seconds)):
            phase = [u for u, t in updates if lo <= t < hi]
            settled_cells = {int(u.range_start_s // BENCH_SPEC.window_s) for u in phase if u.basis == "settled"}
            print(f"  {label} [{lo:.1f}s, {hi:.1f}s): updates={len(phase)} "
                  f"settled={sum(1 for u in phase if u.basis == 'settled')} settled_cells={len(settled_cells)}")
    if leads:
        print(f"range start minus playhead: median {statistics.median(leads):+.2f}s "
              f"min {min(leads):+.2f}s max {max(leads):+.2f}s")
    if per_cell:
        print(f"analyses per cell: max {max(per_cell.values())} "
              f"mean {statistics.mean(per_cell.values()):.2f} cells={len(per_cell)}")
    print(f"dropped={tap.dropped} modes={[s.mode for s in states]}")


if __name__ == "__main__":
    main()
