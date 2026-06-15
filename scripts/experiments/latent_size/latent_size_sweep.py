"""Minimum acceptable latent size for the loop-focused workflow.

Science task: "We currently loop over one section of 60 second latent.
Does it need to be 60 seconds?"

Why it matters: the diffusion window length T (frames = seconds × 25) is
the size of the tensor the DiT denoises every pass. A shorter window
means proportionally less DiT compute per (re)diffusion — i.e. lower
param-update latency (knob-to-ear) — and a smaller activation footprint
(VRAM). The cost is quality: a too-short section may lose musical
structure or loop too tightly. This sweeps the window and measures both
sides so we can pick the smallest *acceptable* size.

Design: pure text-to-music (the loop-focused generation case), fixed
prompts/seeds/knobs, vary only ``duration``. Eager backend so the
latency curve is clean compute scaling, free of torch.compile recompiles
or TRT profile quantization. For each render we record generate +
decode wall time and the same loop/quality proxies as the loop-prompting
study, and save a WAV for human listening.

Latency caveat: these are single-shot eager timings, NOT the live
knob-to-ear number (that path is windowed + TRT). The transferable
result is the *scaling with T*; realizing a sub-60 s window live also
needs a matching TRT engine profile (today: 60/120/240 s only).

Run (Linux):
    .venv/bin/python scripts/experiments/latent_size/latent_size_sweep.py
Run (Windows):
    .venv/Scripts/python.exe scripts\\experiments\\latent_size\\latent_size_sweep.py

Outputs WAVs + metrics.json under test_output/experiments/latent_size/
(both gitignored) and prints a markdown summary for the results doc.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(
    p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists()
)
# Force OUR repo to the front (a sibling ACE-Step checkout shadows `acestep`).
while str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS  # noqa: E402
from acestep.engine.session import Session  # noqa: E402
from acestep.nodes.types import Curve  # noqa: E402


# One rhythmic (seams audible, structure matters) + one melodic prompt.
PROMPTS = [
    {"slug": "techno", "tags": "driving techno, punchy kick, dark hypnotic groove, modular bleeps",
     "bpm": 130, "key": "F minor"},
    {"slug": "lofi", "tags": "lo-fi hip hop, mellow piano, vinyl crackle, boom bap drums",
     "bpm": 80, "key": "C minor"},
]

DURATIONS = [60.0, 45.0, 30.0, 20.0, 15.0, 10.0, 8.0, 6.0]
SEEDS = [1528, 42, 9999]
STEPS = 8
SHIFT = 3.0
CFG = 7.5

SR = 48000
WIN_S = 0.5  # head/tail window for the seam proxies


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return {"mean": round(sum(xs) / n, 4), "min": round(xs[0], 4),
            "p50": round(xs[n // 2], 4), "max": round(xs[-1], 4), "n": n}


def _mono(wav: torch.Tensor) -> torch.Tensor:
    w = wav.detach().float().cpu()
    while w.dim() > 2:
        w = w.squeeze(0)
    if w.dim() == 2:
        w = w.mean(0)
    return w


def _log_spec_profile(seg: torch.Tensor) -> torch.Tensor:
    n_fft = 2048
    if seg.numel() < n_fft:
        seg = torch.nn.functional.pad(seg, (0, n_fft - seg.numel()))
    spec = torch.stft(seg, n_fft=n_fft, hop_length=512,
                      window=torch.hann_window(n_fft), return_complex=True)
    return torch.log1p(spec.abs().mean(dim=1))


def clip_metrics(wav: torch.Tensor) -> dict:
    """Loop/quality proxies for one decoded clip (see loop_prompting study)."""
    m = _mono(wav)
    n = m.numel()
    w = min(int(WIN_S * SR), n // 2)
    head, tail = m[:w], m[-w:]

    hp, tp = _log_spec_profile(head), _log_spec_profile(tail)
    seam_spec_dist = float(torch.linalg.vector_norm(hp - tp) / math.sqrt(hp.numel()))
    rms = lambda x: float(torch.sqrt(torch.clamp((x ** 2).mean(), min=1e-12)))
    to_db = lambda r: 20.0 * math.log10(max(r, 1e-9))

    fl = int(0.05 * SR)
    frames = m[: (n // fl) * fl].reshape(-1, fl)
    frame_rms_db = torch.tensor([to_db(rms(f)) for f in frames])
    silence_frac = float((frame_rms_db < -45.0).float().mean())

    prof = _log_spec_profile(m)
    freqs = torch.linspace(0, SR / 2, prof.numel())
    centroid_hz = float((freqs * prof).sum() / torch.clamp(prof.sum(), min=1e-9))

    return {
        "seam_spec_dist": round(seam_spec_dist, 5),
        "silence_frac": round(silence_frac, 4),
        "rms_db": round(to_db(rms(m)), 2),
        "centroid_hz": round(centroid_hz, 1),
    }


def save_wav(wav: torch.Tensor, path: Path) -> None:
    import soundfile as sf
    w = wav.detach().float().cpu()
    while w.dim() > 2:
        w = w.squeeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), w.T.numpy(), SR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--durations", type=float, nargs="+", default=DURATIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--shift", type=float, default=SHIFT)
    ap.add_argument("--cfg", type=float, default=CFG)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "test_output" / "experiments" / "latent_size")
    args = ap.parse_args()

    t0 = time.time()
    print("[1] loading session (eager)...", flush=True)
    session = Session()
    print(f"    ready in {time.time() - t0:.1f}s", flush=True)

    rows = []
    for p in PROMPTS:
        for dur in args.durations:
            cond = session.encode_text(
                tags=p["tags"], lyrics="[instrumental]",
                instruction=TASK_INSTRUCTIONS["text2music"],
                bpm=p["bpm"], duration=dur, key=p["key"],
            )
            neg = session.null_conditioning(cond)
            T = int(round(dur * 25))
            guidance = Curve(tensor=torch.full((T,), args.cfg, dtype=torch.bfloat16))
            for seed in args.seeds:
                torch.cuda.synchronize()
                tg = time.time()
                lat = session.generate(
                    conditioning=cond, negative=neg, guidance_curve=guidance,
                    seed=seed, duration=dur, steps=args.steps, shift=args.shift,
                )
                torch.cuda.synchronize()
                gen_s = time.time() - tg

                td = time.time()
                audio = session.decode(lat)
                torch.cuda.synchronize()
                dec_s = time.time() - td

                met = clip_metrics(audio.waveform)
                save_wav(audio.waveform, args.out / p["slug"] / f"dur{int(dur)}s__seed{seed}.wav")
                row = {"prompt": p["slug"], "duration_s": dur, "frames": T, "seed": seed,
                       "gen_s": round(gen_s, 4), "dec_s": round(dec_s, 4), **met}
                rows.append(row)
                print(f"  {p['slug']:7s} dur={dur:5.1f}s (T={T:4d}) seed={seed:<5d} "
                      f"gen={gen_s:.3f}s dec={dec_s:.3f}s "
                      f"seam={met['seam_spec_dist']:.4f} sil={met['silence_frac']:.3f}",
                      flush=True)

    # Aggregate per duration across prompts+seeds.
    agg = {}
    base_gen = None
    for dur in args.durations:
        sub = [r for r in rows if r["duration_s"] == dur]
        g = _stats([r["gen_s"] for r in sub])
        if dur == 60.0:
            base_gen = g["mean"]
        agg[str(dur)] = {
            "gen_s": g,
            "dec_s": _stats([r["dec_s"] for r in sub]),
            "seam_spec_dist": _stats([r["seam_spec_dist"] for r in sub]),
            "silence_frac": _stats([r["silence_frac"] for r in sub]),
        }

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"durations": args.durations, "seeds": args.seeds,
                   "steps": args.steps, "shift": args.shift, "cfg": args.cfg,
                   "prompts": [p["slug"] for p in PROMPTS]},
        "rows": rows, "aggregate": agg,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=2))

    print("\n### Latency + quality vs window size (mean across prompts × seeds)\n")
    print("| duration | frames | gen_s | gen speedup vs 60s | dec_s | seam_spec_dist | silence_frac |")
    print("|---|---|---|---|---|---|---|")
    for dur in args.durations:
        a = agg[str(dur)]
        g = a["gen_s"]["mean"]
        sp = f"{base_gen / g:.2f}x" if base_gen and g else "—"
        print(f"| {dur:.0f}s | {int(dur*25)} | {g:.3f} | {sp} | "
              f"{a['dec_s']['mean']:.3f} | {a['seam_spec_dist']['mean']:.4f} | "
              f"{a['silence_frac']['mean']:.4f} |")
    print(f"\nWAVs + metrics.json under {args.out}")


if __name__ == "__main__":
    main()
