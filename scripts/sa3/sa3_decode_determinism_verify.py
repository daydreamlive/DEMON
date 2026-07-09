"""SA3 deterministic-decode verification: production codec, repeat renders.

Post-fix companion to ``sa3_decode_determinism_gate.py``. Where the gate
proved the raw decode stage is stochastic, this drives the PRODUCTION
surface (``SA3Context`` + ``make_codec`` → ``SA3SAMECodec.decode_full``)
and shows:

  [fix-off]   decode_seed=None (legacy behavior): decoding the same latent
              twice still differs — the pre-fix contrast.
  [fix-on]    decode_seed=<seed>: two decodes are bit-identical.
  [seed-diff] a different decode_seed gives a different (but equally
              plausible) render — seeds are distinct noise realizations.
  [A/B]       seeded-noise vs noise-disabled decode of the same latent,
              as listening artifacts for the sound-character call.

Writes WAV pairs for listening and prints an objective-diff table.

Run:
    .venv/Scripts/python.exe scripts/sa3/sa3_decode_determinism_verify.py \
        --model=small-music --prompt "warm analog house groove, 124 bpm" \
        --duration 10 --steps 8 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
for _p in (str(_REPO_ROOT),):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from acestep import paths  # noqa: E402
from acestep.engine.sa3_context import SA3Context  # noqa: E402
from acestep.engine.sa3_stream_helpers import sa3_decode_noise_mode  # noqa: E402


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    sig_rms = a.pow(2).mean().sqrt().item()
    diff_rms = (a - b).pow(2).mean().sqrt().item()
    return {
        "bit_identical": bool(torch.equal(a, b)),
        "max_abs_diff": d.max().item(),
        "rms_diff": diff_rms,
        "diff_db_rel_signal": (
            20.0 * torch.log10(torch.tensor(diff_rms / sig_rms)).item()
            if diff_rms > 0 and sig_rms > 0 else float("-inf")
        ),
    }


def fmt(name: str, s: dict) -> str:
    return (
        f"{name:<34} identical={str(s['bit_identical']):<5} "
        f"max_abs={s['max_abs_diff']:.3e} rms={s['rms_diff']:.3e} "
        f"rel_dB={s['diff_db_rel_signal']:+.1f}"
    )


def save_wav(path: Path, audio_cn: torch.Tensor, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio_cn.float().cpu().numpy().T, samplerate=sr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="small-music")
    ap.add_argument("--prompt", default="warm analog house groove, 124 bpm, deep bassline")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        paths.models_dir() / "sa3" / "spike_out" / "decode_determinism"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    context = SA3Context(
        model_id=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"[load] SA3Context({args.model!r}) in {time.time() - t0:.1f}s")
    codec = context.make_codec()
    print(f"[codec] {type(codec).__name__}")
    sr = context.sample_rate

    latent = context.sam.generate(
        prompt=args.prompt, duration=args.duration, steps=args.steps,
        seed=args.seed, cfg_scale=1.0, return_latents=True,
    )
    print(f"[gen] latent shape={tuple(latent.shape)} (seed={args.seed})")

    results: dict[str, dict] = {}

    # fix-off: the legacy unseeded path (decode_seed=None)
    off1 = codec.decode_full(latent)
    off2 = codec.decode_full(latent)
    results["fix-off (unseeded, 2 decodes)"] = stats(off1, off2)

    # fix-on: production decode with the generation seed pinned
    on1 = codec.decode_full(latent, decode_seed=args.seed)
    on2 = codec.decode_full(latent, decode_seed=args.seed)
    results["fix-on (seeded, 2 decodes)"] = stats(on1, on2)

    # a different decode seed is a different (valid) realization
    other = codec.decode_full(latent, decode_seed=args.seed + 1)
    results["fix-on seedA vs seedB"] = stats(on1, other)

    # A/B: seeded noise vs noise disabled outright
    with sa3_decode_noise_mode(context.sam, enabled=False):
        noiseless = codec.decode_full(latent)
    results["seeded vs noise-off (A/B)"] = stats(on1, noiseless)

    # Windowed codecs (medium): the production per-tick render path.
    if hasattr(codec, "decode_window"):
        n = min(latent.shape[-1] * context.downsampling_ratio, 4 * sr)
        w1 = codec.decode_window(latent, 0, n)
        w2 = codec.decode_window(latent, 0, n)
        results["hot path decode_window (x2)"] = stats(w1, w2)

    tag = f"{args.model}_seed{args.seed}_{int(args.duration)}s"
    wavs = {
        f"verify_{tag}_fixoff_run1.wav": off1,
        f"verify_{tag}_fixoff_run2.wav": off2,
        f"verify_{tag}_fixon_run1.wav": on1,
        f"verify_{tag}_fixon_run2.wav": on2,
        f"verify_{tag}_fixon_otherseed.wav": other,
        f"verify_{tag}_ab_noise_off.wav": noiseless,
    }
    for name, audio in wavs.items():
        save_wav(out_dir / name, audio, sr)
        print(f"[save] {out_dir / name}")

    print("\n=== verify results ===")
    for name, s in results.items():
        print(fmt(name, s))

    ok = (
        results["fix-on (seeded, 2 decodes)"]["bit_identical"]
        and not results["fix-off (unseeded, 2 decodes)"]["bit_identical"]
        and not results["fix-on seedA vs seedB"]["bit_identical"]
    )
    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'} "
          "(seeded repeat identical, unseeded repeat differs, seeds distinct)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
