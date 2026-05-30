"""PyMusicLooper-backed loop-point refinement for live demo buffers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from acestep.engine.obs import logger


@dataclass(frozen=True)
class SmartLoopResult:
    start_sec: float
    end_sec: float
    score: float


def refine_loop_points(
    audio: np.ndarray,
    sample_rate: int,
    *,
    approx_start_sec: float,
    approx_end_sec: float,
    anchor_duration_sec: float | None = None,
    duration_flex_pct: float = 0.03,
    max_edge_shift_sec: float = 0.5,
    min_loop_duration_sec: float = 1.0,
    disable_pruning: bool = False,
) -> SmartLoopResult:
    """Run PyMusicLooper near the supplied loop band and return the best hit.

    PyMusicLooper works from an audio file, so the live buffer snapshot is
    written to a temporary WAV. ``approx_*`` limits the search to the user's
    intended region, while min/max duration constraints keep the result close
    to the initially supplied loop size.
    """
    from pymusiclooper.analysis import LoopNotFoundError
    from pymusiclooper.core import MusicLooper

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    duration_sec = len(audio) / sample_rate
    if duration_sec <= 0:
        raise LoopNotFoundError("empty audio buffer")

    approx_start_sec = max(0.0, min(duration_sec, float(approx_start_sec)))
    approx_end_sec = max(0.0, min(duration_sec, float(approx_end_sec)))
    if approx_end_sec <= approx_start_sec:
        raise LoopNotFoundError("degenerate loop band")

    loop_len = (
        float(anchor_duration_sec)
        if anchor_duration_sec and anchor_duration_sec > 0
        else approx_end_sec - approx_start_sec
    )
    if loop_len < min_loop_duration_sec:
        raise LoopNotFoundError("loop band is shorter than smart-loop minimum")
    flex = max(0.0, min(1.0, float(duration_flex_pct)))
    min_loop = max(min_loop_duration_sec, min(duration_sec, loop_len * (1.0 - flex)))
    max_loop = max(min_loop, min(duration_sec, loop_len * (1.0 + flex)))
    max_edge_shift_sec = max(0.0, float(max_edge_shift_sec))

    with tempfile.TemporaryDirectory(prefix="demon-smart-loop-") as td:
        path = Path(td) / "live-buffer.wav"
        sf.write(path, audio, sample_rate, subtype="PCM_16")
        looper = MusicLooper(str(path))
        pairs = looper.find_loop_pairs(
            min_duration_multiplier=0.0,
            min_loop_duration=min_loop,
            max_loop_duration=max_loop,
            approx_loop_start=approx_start_sec,
            approx_loop_end=approx_end_sec,
            brute_force=False,
            disable_pruning=disable_pruning,
        )

        valid: list[tuple[object, float, float]] = []
        for pair in pairs:
            start = float(looper.samples_to_seconds(pair.loop_start))
            end = float(looper.samples_to_seconds(pair.loop_end))
            candidate_len = end - start
            if candidate_len < min_loop_duration_sec:
                continue
            if candidate_len < min_loop or candidate_len > max_loop:
                continue
            if abs(start - approx_start_sec) > max_edge_shift_sec:
                continue
            if abs(end - approx_end_sec) > max_edge_shift_sec:
                continue
            valid.append((pair, start, end))

        if not valid:
            raise LoopNotFoundError(
                "no PyMusicLooper pairs satisfied duration/edge guards"
            )
        best, start, end = valid[0]
        logger.debug(
            "smart_loop_refined approx=({:.3f},{:.3f}) result=({:.3f},{:.3f}) score={:.6f}",
            approx_start_sec,
            approx_end_sec,
            start,
            end,
            float(best.score),
        )
        return SmartLoopResult(
            start_sec=float(start),
            end_sec=float(end),
            score=float(best.score),
        )
