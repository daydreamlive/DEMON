"""Few-step acoustic solving with the released YuE2 weights (no retraining).

For each composition (three styles, 60 s, forced length), the 32-step
uniform midpoint solve is the reference; every other configuration
starts from the SAME noise (same seed) and is compared against it:

* uniform midpoint grids at 24, 16, 12, 8, 6, 4 steps (2 NFE per step);
* plain Euler at the same NFE (2N Euler steps for N midpoint steps);
* non-uniform midpoint grids at 12 and 8 steps (noise-dense also at 6
  and 4): the uniform grid warped
  by the time shift ``f(s) = a*s / (1 + (a-1)*s)``; ``a = 3`` puts the
  steps near the noise end (s = 1), ``a = 1/3`` near the data end.

Metrics per configuration vs the reference: latent relative L2, log-mel
spectral distance (mean abs dB over the mel spectrogram, 10 ms hop),
integrated loudness (BS.1770, torchaudio) and its delta, long-term mel
spectrum (LTAS) drift in dB. Scale references per composition: a seed
change at 32 steps (a different performance of the same song) and the
eager 32-step solve (the TRT-vs-eager numerical floor). One decoded wav
per configuration.

    python scripts/spikes/yue2_fewstep.py --out DIR [--seconds 60]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from acestep.engine.yue2_velocity import midpoint_velocity, raw_time, truncated_grid, upstream_noise

SR = 48000

COMPOSITIONS = [
    ("citypop", "city pop, female vocal, bright synths, groovy bass",
     "[Verse]\nNeon lights on the harbor road\nWe drive until the morning glows\n"
     "[Chorus]\nStay with me tonight\nCity of a thousand lights\n"),
    ("metal", "heavy metal, distorted guitars, aggressive drums, fast", ""),
    ("folk", "acoustic folk, fingerpicked guitar, warm, gentle", ""),
]


def uniform(n: int) -> list:
    return truncated_grid(n, 1.0).tolist()


def shifted(n: int, a: float) -> list:
    return [a * s / (1.0 + (a - 1.0) * s) for s in uniform(n)]


def configs() -> list:
    rows = [(f"mid{n}", "mid", uniform(n)) for n in (32, 24, 16, 12, 8, 6, 4)]
    rows += [(f"euler{2 * n}", "euler", uniform(2 * n)) for n in (16, 12, 8, 6, 4)]
    for n in (12, 8):
        rows.append((f"mid{n}_noisedense", "mid", shifted(n, 3.0)))
        rows.append((f"mid{n}_datadense", "mid", shifted(n, 1.0 / 3.0)))
    for n in (6, 4):
        rows.append((f"mid{n}_noisedense", "mid", shifted(n, 3.0)))
    return rows


@torch.inference_mode()
def solve(velocity, bundle, x, grid, method):
    """Midpoint (upstream's step, any grid) or Euler from ``x`` (s=1)."""
    for s, s_next in zip(grid[:-1], grid[1:]):
        h = s - s_next
        if method == "mid":
            v = midpoint_velocity(velocity, bundle, x, [s], h)
        else:
            v = velocity(bundle, x, [raw_time(s)]).clone()
        x = x - v * h
    return x


def save_wav(path: Path, audio_2n: torch.Tensor):
    pcm = (audio_2n.clamp(-1, 1).float().cpu().numpy().T * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def rel_l2(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def features(audio_2n: torch.Tensor) -> dict:
    import librosa
    import torchaudio.functional as AF

    audio = audio_2n.clamp(-1, 1).float()
    mono = audio.mean(0).cpu().numpy()
    mel = librosa.feature.melspectrogram(y=mono, sr=SR, n_fft=2048, hop_length=480, n_mels=128)
    return {
        "mel_db": librosa.power_to_db(mel, ref=1.0, amin=1e-8, top_db=None),
        "ltas_db": librosa.power_to_db(mel.mean(axis=1), ref=1.0, amin=1e-10, top_db=None),
        "lufs": float(AF.loudness(audio.cpu(), SR)),
    }


def compare(f, ref) -> dict:
    floor = ref["mel_db"].max() - 80.0  # ignore differences below -80 dB re the peak
    a = np.maximum(f["mel_db"], floor)
    b = np.maximum(ref["mel_db"], floor)
    return {
        "logmel_db": float(np.abs(a - b).mean()),
        "ltas_db": float(np.abs(f["ltas_db"] - ref["ltas_db"]).mean()),
        "lufs": f["lufs"],
        "lufs_delta": f["lufs"] - ref["lufs"],
    }


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--only", nargs="*", help="composition names to run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from acestep.engine.yue2_context import YuE2Context
    from acestep.engine.yue2_runtime import trt_dir, weights_root
    from acestep.engine.yue2_trt import TRTVelocity

    ctx = YuE2Context(weights_root(), trt_dir=trt_dir())
    frames = int(round(args.seconds * 25))
    report = {"frames": frames, "configs": [c[0] for c in configs()], "compositions": {}}
    for name, style, lyrics in COMPOSITIONS:
        if args.only and name not in args.only:
            continue
        comp = ctx.compose(style=style, lyrics=lyrics, seed=0, max_frames=frames, exact_frames=frames)
        bundle = ctx.bundle(comp)
        trt = isinstance(ctx.velocity, TRTVelocity) and ctx.velocity.covers(bundle)
        noise = upstream_noise(bundle.seed, bundle.frames).to(ctx.device, torch.bfloat16)[None]
        rows = {}
        ref_latent, ref_ms = timed(lambda: solve(ctx.velocity, bundle, noise, uniform(32), "mid"))
        # Sanity: the generic solver at 32 uniform steps IS the production solve.
        prod = ctx.solve(bundle)
        ref_audio = ctx.decode_full(ref_latent)
        ref = features(ref_audio)
        for cname, method, grid in configs():
            if cname == "mid32":
                latent, ms = ref_latent, ref_ms
            else:
                latent, ms = timed(lambda: solve(ctx.velocity, bundle, noise, grid, method))
            nfe = (len(grid) - 1) * (2 if method == "mid" else 1)
            rows[cname] = {"method": method, "steps": len(grid) - 1, "nfe": nfe, "solve_ms": ms}
            if not torch.isfinite(latent).all():
                rows[cname]["diverged"] = True  # the grid breaks the solve
                print(name, cname, "DIVERGED (non-finite latent)", flush=True)
                continue
            audio = ctx.decode_full(latent)
            save_wav(args.out / f"{name}_{cname}.wav", audio)
            rows[cname].update(latent_rel_l2=rel_l2(latent, ref_latent),
                               **compare(features(audio), ref))
            print(name, cname, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                           for k, v in rows[cname].items()}), flush=True)
        # Scale references.
        seed1 = solve(ctx.velocity, bundle,
                      upstream_noise(1, bundle.frames).to(ctx.device, torch.bfloat16)[None],
                      uniform(32), "mid")
        audio = ctx.decode_full(seed1)
        save_wav(args.out / f"{name}_ref_seed1_mid32.wav", audio)
        rows["ref_seed1_mid32"] = {"latent_rel_l2": rel_l2(seed1, ref_latent), **compare(features(audio), ref)}
        if trt:
            eager = solve(ctx.eager_velocity, bundle, noise, uniform(32), "mid")
            rows["ref_eager_mid32"] = {"latent_rel_l2": rel_l2(eager, ref_latent),
                                       **compare(features(ctx.decode_full(eager)), ref)}
        report["compositions"][name] = {
            "style": style, "lyrics": bool(lyrics), "frames": bundle.frames,
            "cond_tokens": bundle.cond_tokens, "nar": "trt" if trt else "eager",
            "generic_solver_equals_production": bool(torch.equal(prod, ref_latent)),
            "rows": rows,
        }
        print(name, "refs", json.dumps(rows["ref_seed1_mid32"]), json.dumps(rows.get("ref_eager_mid32")),
              "equal_prod", report["compositions"][name]["generic_solver_equals_production"], flush=True)
        ctx.release_bundle(bundle)
        (args.out / "fewstep.json").write_text(json.dumps(report, indent=2))
    ctx.close()


if __name__ == "__main__":
    main()
