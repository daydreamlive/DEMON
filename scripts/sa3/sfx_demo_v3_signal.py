"""Score the /sfx v3 wire-smoke snapshots (scripts/sa3/sfx_demo_wire_smoke.mjs).

For every ``drift_<layer>_tNNN.wav`` (a full-canvas mirror snapshot taken
every 5 s) this reports, per snapshot:

* ``flatness``: mean spectral flatness over STFT frames (1 = white noise);
* ``hf``: share of energy above 6 kHz;
* ``rms_db``: canvas RMS in dBFS;
* for input layers, ``corr``: waveform Pearson correlation with the fitted
  input (``input_ambience_fitted.wav``), ``spec_corr``: correlation of the
  dB magnitude spectrograms, and ``hum``: share of energy within 6 Hz of
  the input's 110/220/330 Hz hum.

It also scores ``oneshot_input_mirror.wav``: canvas length and tail level.
Writes ``signal.json`` next to the WAVs and prints a compact table.

    python scripts/sa3/sfx_demo_v3_signal.py E:/Projects/sa3-variants/sfx/demo_v3
"""

from __future__ import annotations

import json
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 48000
NFFT = 2048
HOP = 1024


def read_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32767
    return data.reshape(-1, ch).mean(axis=1)


def stft_power(x: np.ndarray) -> np.ndarray:
    win = np.hanning(NFFT)
    n = 1 + max(0, (len(x) - NFFT) // HOP)
    frames = np.stack([x[i * HOP: i * HOP + NFFT] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2  # [frames, bins]


def score(x: np.ndarray, ref: np.ndarray | None) -> dict:
    p = stft_power(x) + 1e-12
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    flat = np.exp(np.log(p).mean(axis=1)) / p.mean(axis=1)
    loud = p.sum(axis=1) > 1e-6 * p.sum(axis=1).max()
    total = p.sum()
    out = {
        "flatness": round(float(flat[loud].mean()), 4),
        "hf": round(float(p[:, freqs > 6000].sum() / total), 4),
        "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 1),
    }
    hum = np.zeros_like(freqs, dtype=bool)
    for f0 in (110, 220, 330):
        hum |= np.abs(freqs - f0) < 6
    out["hum"] = round(float(p[:, hum].sum() / total), 4)
    if ref is not None:
        n = min(len(x), len(ref))
        out["corr"] = round(float(np.corrcoef(x[:n], ref[:n])[0, 1]), 3)
        a = 10 * np.log10(stft_power(x[:n]) + 1e-10).ravel()
        b = 10 * np.log10(stft_power(ref[:n]) + 1e-10).ravel()
        out["spec_corr"] = round(float(np.corrcoef(a, b)[0, 1]), 3)
    return out


def main(out_dir: str) -> None:
    root = Path(out_dir)
    ref_path = root / "input_ambience_fitted.wav"
    ref = read_mono(ref_path) if ref_path.exists() else None
    series: dict[str, list] = defaultdict(list)
    for f in sorted(root.glob("drift_*_t*.wav")):
        m = re.match(r"drift_(\w+?)_t(\d+)\.wav", f.name)
        key, t = m.group(1), int(m.group(2))
        series[key].append({"t": t, **score(read_mono(f), ref if key.startswith("in") else None)})
    result: dict = {"drift": series}
    if ref is not None:
        result["input_reference"] = score(ref, ref)
    shot = root / "oneshot_input_mirror.wav"
    if shot.exists():
        x = read_mono(shot)
        tail = x[-int(0.25 * SR):]
        peak = np.max(np.abs(x)) + 1e-12
        result["oneshot_input"] = {
            "canvas_s": round(len(x) / SR, 3),
            "peak_db": round(float(20 * np.log10(peak)), 1),
            "tail250ms_rms_db": round(float(20 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-12)), 1),
        }
    (root / "signal.json").write_text(json.dumps(result, indent=2))
    for key, rows in series.items():
        cols = [c for c in ("flatness", "hf", "rms_db", "hum", "corr", "spec_corr") if c in rows[0]]
        print(key)
        print("  t     " + "  ".join(f"{c:>9}" for c in cols))
        for r in rows:
            print(f"  {r['t']:>3}   " + "  ".join(f"{r[c]:>9}" for c in cols))
    for k in ("input_reference", "oneshot_input"):
        if k in result:
            print(k, result[k])


if __name__ == "__main__":
    main(sys.argv[1])
