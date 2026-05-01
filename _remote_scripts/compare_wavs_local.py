"""Compare PT vs TRT wavs programmatically."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import sys
DIR = Path(sys.argv[1])
PREFIX = sys.argv[2]  # e.g., t2m_xl_turbo_h100  or  t2m_xl_turbo
SEEDS = [1528, 42, 9999]

print("seed       samples    corr     pt_rms      trt_rms     max_diff   pt_peak   trt_peak  has_nan")
print("-" * 100)
all_pass = True
for seed in SEEDS:
    pt_path = DIR / f"{PREFIX}_pytorch_seed_{seed}.wav"
    trt_path = DIR / f"{PREFIX}_tensorrt_seed_{seed}.wav"
    pw, sr_p = sf.read(str(pt_path))
    tw, sr_t = sf.read(str(trt_path))
    assert sr_p == sr_t, f"sample rate mismatch {sr_p} vs {sr_t}"
    n = min(pw.shape[0], tw.shape[0])
    pw = pw[:n].astype(np.float64).flatten()
    tw = tw[:n].astype(np.float64).flatten()

    if pw.std() > 0 and tw.std() > 0:
        corr = float(np.corrcoef(pw, tw)[0, 1])
    else:
        corr = float("nan")
    pt_rms = float(np.sqrt((pw ** 2).mean()))
    trt_rms = float(np.sqrt((tw ** 2).mean()))
    max_diff = float(np.abs(pw - tw).max())
    pt_peak = float(np.abs(pw).max())
    trt_peak = float(np.abs(tw).max())
    has_nan = bool(np.isnan(tw).any())

    passed = (not has_nan) and corr > 0.95 and trt_rms > 1e-4
    all_pass &= passed
    flag = "PASS" if passed else "FAIL"
    print(f"{seed:>5}  {n:>10d}  {corr:7.4f}  {pt_rms:9.4f}  {trt_rms:9.4f}  {max_diff:9.4f}  {pt_peak:8.4f}  {trt_peak:8.4f}   {has_nan}   {flag}")

print()
print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
