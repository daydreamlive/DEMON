"""Cross-compare local Windows and H100 wavs across seeds and backends."""
from pathlib import Path
import numpy as np
import soundfile as sf

LOCAL = Path(r"C:\_dev\projects\ACE-Step-1.5_alt\test_output\text_to_music_xl_turbo_compare")
H100 = Path(r"C:\_dev\projects\ACE-Step-1.5_alt\test_output\text_to_music_xl_turbo_compare_h100")
SEEDS = [1528, 42, 9999]

def load(path):
    w, sr = sf.read(str(path))
    return w.astype(np.float64).flatten(), sr

def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() > 0 and b.std() > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return float("nan")

print(f"{'comparison':<60} {'corr':>8}")
print("-" * 70)
for seed in SEEDS:
    files = {
        "local_pt":  LOCAL / f"t2m_xl_turbo_pytorch_seed_{seed}.wav",
        "local_trt": LOCAL / f"t2m_xl_turbo_tensorrt_seed_{seed}.wav",
        "h100_pt":   H100 / f"t2m_xl_turbo_h100_pytorch_seed_{seed}.wav",
        "h100_trt":  H100 / f"t2m_xl_turbo_h100_tensorrt_seed_{seed}.wav",
    }
    wavs = {name: load(p)[0] for name, p in files.items() if p.exists()}
    pairs = [
        ("local_pt", "local_trt"),
        ("h100_pt",  "h100_trt"),
        ("local_pt", "h100_pt"),
        ("local_trt", "h100_trt"),
        ("local_pt", "h100_trt"),
    ]
    for a, b in pairs:
        if a in wavs and b in wavs:
            print(f"  seed {seed:>5}  {a:>10} vs {b:<10}  corr = {corr(wavs[a], wavs[b]):>8.4f}")
    print()
