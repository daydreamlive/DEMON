"""A/B listening test: generate same prompts+seeds on bf16 and W8A8 FP8.

Writes paired WAVs to ``benchmarks-pr17/listen/{bf16,fp8}/NN_<tag>.wav``
so the user can A/B them. Each pair shares prompt, seed, BPM, and key,
so any audible difference is attributable to the decoder backend.

Run::

    uv run python benchmarks-pr17/fp8_listen_test.py
    uv run python benchmarks-pr17/fp8_listen_test.py --duration 30
    uv run python benchmarks-pr17/fp8_listen_test.py --prompts 0 3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import soundfile as sf
import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.paths import trt_engines_dir


# Prompt mix: diverse content so quantization artifacts surface across
# different timbral regimes — transients, sustained tones, vocal-like
# textures, and dense polyphony.
PROMPTS = [
    ("dance", "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads", 128, "F minor"),
    ("jazz",  "jazz piano trio, brushed drums, walking bass, late-night club", 140, "Bb major"),
    ("ambient", "ambient electronic, slow evolving pads, glassy textures, no drums", 80, "C minor"),
    ("metal", "metal, aggressive guitar riffs, fast double kick, growling vocals", 180, "E minor"),
    ("orch", "classical orchestral, sweeping strings, brass, timpani", 90, "D major"),
    ("folk", "acoustic folk, fingerpicked guitar, soft harmonica, brushes", 100, "G major"),
]


ENGINE_VARIANTS = {
    "bf16": "decoder_xl-turbo_mixed_refit_b4_60s",
    "fp8":  "decoder_xl-turbo_fp8_refit_b4_60s",
}

# Shared VAE engines — same for both runs (decoder is the only thing
# we're testing).
SHARED_ENGINES = {
    "vae_encode": "vae_encode_fp16_60s",
    "vae_decode": "vae_decode_fp16_60s",
}


def _resolve_engines(decoder_dir_name: str) -> dict[str, str]:
    root = trt_engines_dir()
    out: dict[str, str] = {}
    out["decoder"] = str(root / decoder_dir_name / f"{decoder_dir_name}.engine")
    for key, name in SHARED_ENGINES.items():
        out[key] = str(root / name / f"{name}.engine")
    for k, v in out.items():
        if not Path(v).exists():
            raise FileNotFoundError(f"Missing {k} engine: {v}")
    return out


def _generate_one(
    session: Session,
    *,
    tag: str,
    prompt: str,
    bpm: int,
    key: str,
    seed: int,
    duration: float,
    steps: int,
    shift: float,
    out_path: Path,
) -> dict:
    """Generate one track end-to-end. Returns timing info."""
    t0 = time.perf_counter()
    cond = session.encode_text(
        tags=prompt,
        instruction=TASK_INSTRUCTIONS["text2music"],
        bpm=bpm, duration=duration, key=key,
    )
    t_text = time.perf_counter() - t0

    t1 = time.perf_counter()
    latent = session.generate(
        conditioning=cond,
        seed=seed,
        denoise=1.0,
        steps=steps,
        shift=shift,
        method="ode",
    )
    t_diffuse = time.perf_counter() - t1

    t2 = time.perf_counter()
    audio = session.decode(latent)
    t_decode = time.perf_counter() - t2

    # audio.waveform is [B, C, T]; soundfile.write needs [T, C] (or 1D).
    wav = audio.waveform.detach().cpu().float().numpy()
    if wav.ndim == 3:
        wav = wav[0]
    wav = wav.T  # [T, C]
    sf.write(str(out_path), wav, audio.sample_rate, format="WAV")
    total = time.perf_counter() - t0
    print(
        f"  {tag:>8s}  seed={seed}  text={t_text:5.2f}s  "
        f"diffuse={t_diffuse:5.2f}s  decode={t_decode:5.2f}s  "
        f"total={total:5.2f}s -> {out_path.name}"
    )
    return {
        "tag": tag,
        "seed": seed,
        "text_s": t_text,
        "diffuse_s": t_diffuse,
        "decode_s": t_decode,
        "total_s": total,
        "wav_path": str(out_path),
    }


def _run_variant(
    variant: str,
    *,
    prompts: list[tuple[str, str, int, str]],
    seed: int,
    duration: float,
    steps: int,
    shift: float,
    out_dir: Path,
) -> list[dict]:
    """Open a Session for one variant, generate all prompts, close.

    Each Session opens its own TRT engines; closing it frees the
    decoder weights before we load the other variant. We deliberately
    do not run them concurrently — 8 GB + 4 GB of decoder weights both
    resident would risk OOM on a 32 GB card alongside intermediates.
    """
    engines = _resolve_engines(ENGINE_VARIANTS[variant])
    print(f"[{variant}] engines:")
    for k, v in engines.items():
        size_mb = Path(v).stat().st_size / 1e6
        print(f"  {k}: {v}  ({size_mb:.0f} MB)")

    session = Session(
        config_path="acestep-v15-xl-turbo",
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=engines,
    )
    print(f"[{variant}] session ready")

    records = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (tag, prompt, bpm, key) in enumerate(prompts):
        out_path = out_dir / f"{i+1:02d}_{tag}.wav"
        rec = _generate_one(
            session,
            tag=tag, prompt=prompt, bpm=bpm, key=key,
            seed=seed,
            duration=duration, steps=steps, shift=shift,
            out_path=out_path,
        )
        rec["prompt"] = prompt
        rec["bpm"] = bpm
        rec["key"] = key
        records.append(rec)

    try:
        session.close()
    except Exception as e:
        print(f"[{variant}] session.close raised: {e}")
    del session
    gc.collect()
    torch.cuda.empty_cache()
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--duration", type=float, default=30.0,
        help="Audio duration per prompt in seconds (default: 30).",
    )
    ap.add_argument(
        "--steps", type=int, default=8,
        help="Diffusion steps (default: 8).",
    )
    ap.add_argument(
        "--shift", type=float, default=3.0,
        help="ODE shift (default: 3.0).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Seed used for every prompt; both variants share it so the "
        "noise trajectory is identical and any difference is the decoder.",
    )
    ap.add_argument(
        "--prompts", nargs="*", type=int, default=None,
        help="Indexes into PROMPTS to render (default: all). "
        "e.g. --prompts 0 2 4",
    )
    ap.add_argument(
        "--out-dir", type=str,
        default=str(Path(__file__).parent / "listen"),
    )
    ap.add_argument(
        "--only", choices=("both", "bf16", "fp8"), default="both",
        help="Skip one variant for debugging (default: both).",
    )
    ap.add_argument(
        "--fp8-tag", type=str, default="fp8",
        help="Subfolder name for the fp8 variant under --out-dir "
        "(default: 'fp8'). Use to preserve multiple FP8 formulations "
        "side by side: 'fp8_sq_a05', 'fp8_absmax', etc.",
    )
    args = ap.parse_args()

    prompts = PROMPTS if args.prompts is None else [PROMPTS[i] for i in args.prompts]
    out_root = Path(args.out_dir).resolve()
    print(f"[setup] output root: {out_root}")
    print(f"[setup] duration={args.duration}s steps={args.steps} shift={args.shift} seed={args.seed}")
    print(f"[setup] prompts: {[p[0] for p in prompts]}")

    all_records: dict[str, list[dict]] = {}
    for variant in ("bf16", "fp8"):
        if args.only != "both" and args.only != variant:
            continue
        # Allow the fp8 variant to write to a custom subfolder so we
        # can preserve multiple formulations side by side without the
        # rename dance. bf16 always lands in 'bf16/' since there's only
        # one reference.
        subdir = variant if variant == "bf16" else args.fp8_tag
        print()
        print("=" * 60)
        print(f"VARIANT: {variant}  (out subdir: {subdir})")
        print("=" * 60)
        records = _run_variant(
            variant,
            prompts=prompts,
            seed=args.seed,
            duration=args.duration,
            steps=args.steps,
            shift=args.shift,
            out_dir=out_root / subdir,
        )
        all_records[variant] = records

    # Summary
    print()
    print("=" * 60)
    print("LISTEN TEST SUMMARY")
    print("=" * 60)
    if "bf16" in all_records and "fp8" in all_records:
        for bf, fp in zip(all_records["bf16"], all_records["fp8"]):
            assert bf["seed"] == fp["seed"]
            print(
                f"  {bf['tag']:>8s}  bf16 diffuse={bf['diffuse_s']:5.2f}s  "
                f"fp8 diffuse={fp['diffuse_s']:5.2f}s  "
                f"speedup={bf['diffuse_s']/max(fp['diffuse_s'], 1e-6):.2f}x"
            )

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "duration": args.duration,
                "steps": args.steps,
                "shift": args.shift,
                "engines": {v: ENGINE_VARIANTS[v] for v in all_records},
                "records": all_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[save] manifest: {manifest_path}")
    print(f"[save] wavs under: {out_root}/<bf16,fp8>/")


if __name__ == "__main__":
    main()
