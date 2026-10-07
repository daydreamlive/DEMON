# Copied verbatim from DEMON-steer (ryanontheinside/feat/steering-generic, HEAD 83f5b77b)
# scripts/steering/proxies.py, so the bench descriptors are the production proxies.
"""Cheap audio proxies for steering discovery and sanity checks.

Each proxy maps mono float audio to one scalar whose sign of change says
which way a concept moved. Log-mel / STFT statistics only (no learned
model), chosen per concept:

* ``centroid``   spectral centroid in Hz (brightness)
* ``lowhigh_db`` energy below 250 Hz over energy above 2.5 kHz, in dB
                 (warmth / spectral tilt)
* ``flatness``   mean spectral flatness (roughness / noisiness)
* ``onset_rate`` detected onsets per second (density / busyness)
* ``perc_ratio`` percussive share of HPSS energy (drums vs pads)
"""

from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP = 512


def _power(y: np.ndarray) -> np.ndarray:
    import librosa

    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2


def _active(p: np.ndarray) -> np.ndarray:
    e = p.sum(axis=0)
    keep = e > (1e-4 * e.max() if e.max() > 0 else 0)
    return p[:, keep] if keep.any() else p


def centroid(y: np.ndarray, sr: int) -> float:
    p = _active(_power(y))
    f = np.linspace(0, sr / 2, p.shape[0])
    w = p.sum(axis=0)
    c = (f[:, None] * p).sum(axis=0) / np.maximum(w, 1e-12)
    return float((c * w).sum() / max(w.sum(), 1e-12))


def lowhigh_db(y: np.ndarray, sr: int) -> float:
    p = _power(y)
    f = np.linspace(0, sr / 2, p.shape[0])
    lo = p[f < 250.0].sum()
    hi = p[f > 2500.0].sum()
    return float(10.0 * np.log10(max(lo, 1e-12) / max(hi, 1e-12)))


def flatness(y: np.ndarray, sr: int) -> float:
    p = _active(_power(y)) + 1e-12
    g = np.exp(np.log(p).mean(axis=0))
    a = p.mean(axis=0)
    return float((g / a).mean())


def onset_rate(y: np.ndarray, sr: int) -> float:
    import librosa

    on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="frames")
    return float(len(on) / max(len(y) / sr, 1e-6))


def perc_ratio(y: np.ndarray, sr: int) -> float:
    import librosa

    s = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    h, pc = librosa.decompose.hpss(s)
    eh = float((np.abs(h) ** 2).sum())
    ep = float((np.abs(pc) ** 2).sum())
    return ep / max(eh + ep, 1e-12)


PROXIES = {
    "centroid": centroid,
    "lowhigh_db": lowhigh_db,
    "flatness": flatness,
    "onset_rate": onset_rate,
    "perc_ratio": perc_ratio,
}


def measure(name: str, y: np.ndarray, sr: int) -> float:
    return PROXIES[name](np.asarray(y, dtype=np.float32), int(sr))
