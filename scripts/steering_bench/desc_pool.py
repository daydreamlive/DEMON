"""All five descriptors for many clips, in parallel (about 1.5 CPU-s per
10 s clip serially, HPSS dominating)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concepts import DESCRIPTORS  # noqa: E402


def _one(job):
    import proxies

    y, sr = job
    y = np.asarray(y).reshape(-1).astype(np.float32) / 32768.0
    return [proxies.measure(k, y, sr) for k in DESCRIPTORS]


def measure_clips(int16_clips, sr: int, workers: int | None = None) -> np.ndarray:
    """``[n, 5]`` (order DESCRIPTORS) for int16 clips."""
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = [(c, int(sr)) for c in int16_clips]
    if workers == 1 or len(jobs) < 4:
        return np.array([_one(j) for j in jobs]).reshape(len(jobs), len(DESCRIPTORS))
    with ProcessPoolExecutor(workers) as ex:
        return np.array(list(ex.map(_one, jobs, chunksize=4))).reshape(len(jobs), len(DESCRIPTORS))
