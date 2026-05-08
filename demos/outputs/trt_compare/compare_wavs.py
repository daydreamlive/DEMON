"""Compare end-to-end stream-cover WAVs between TRT 10.13 and 10.16.

10.16 full-TRT crashes on this system, so we compare the 10.13 baseline
against a 10.16 run with TRT decoder + eager VAE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

DIR = Path(__file__).resolve().parent

WAV_A = DIR / "stream_cover_trt10_13.wav"
WAV_B = DIR / "stream_cover_trt10_16_decoder_only.wav"


def load(path):
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    return data, sr


a, sra = load(WAV_A)
b, srb = load(WAV_B)

print(f"A: {WAV_A.name}  shape={a.shape}  sr={sra}  rms={float(np.sqrt(np.mean(a**2))):.4f}")
print(f"B: {WAV_B.name}  shape={b.shape}  sr={srb}  rms={float(np.sqrt(np.mean(b**2))):.4f}")

if a.shape != b.shape:
    n = min(a.shape[0], b.shape[0])
    print(f"truncate to common length: {n} samples")
    a = a[:n]
    b = b[:n]

diff = a - b
mse = float(np.mean(diff ** 2))
mad = float(np.mean(np.abs(diff)))
mxd = float(np.max(np.abs(diff)))
rms_diff = float(np.sqrt(mse))
rms_a = float(np.sqrt(np.mean(a ** 2)))
print()
print(f"mean_abs_diff = {mad:.6e}")
print(f"max_abs_diff  = {mxd:.6e}")
print(f"rms_diff      = {rms_diff:.6e}")
print(f"rms_a         = {rms_a:.6e}")
print(f"snr (rms_a / rms_diff) = {rms_a / max(rms_diff, 1e-12):.2f}")
