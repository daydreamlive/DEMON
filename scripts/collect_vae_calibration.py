#!/usr/bin/env python3
"""Dump VAE calibration latents for INT8 PTQ.

Generates ``--num-latents`` latents through ``Session.generate()`` over a
varied prompt mix, slices each to ``[1, 64, --frames]`` (matching the
:class:`VAEDecodeInt8Config` opt shape), and saves as ``latent_NNNN.pt``
under ``<models>/calibration/vae_latents/``.

The TRT INT8 entropy calibrator in :mod:`acestep.engine.trt.vae_decode_int8`
reads exactly that directory pattern.

Usage:
    uv run python scripts/collect_vae_calibration.py
    uv run python scripts/collect_vae_calibration.py --num-latents 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Curve
from acestep.paths import default_trt_engines, models_dir, trt_engine_path


PROMPTS = [
    # Original mix
    ("dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads", 128, "F minor"),
    ("jazz piano trio, brushed drums, walking bass", 140, "Bb major"),
    ("ambient electronic, slow pads, evolving textures", 80, "C minor"),
    ("metal, aggressive guitar riffs, fast double kick, growling vocals", 180, "E minor"),
    ("hip hop beat, 808 bass, trap hi-hats, dark synths", 140, "F# minor"),
    ("classical orchestral, sweeping strings, brass, timpani", 90, "D major"),
    ("acoustic folk, fingerpicked guitar, soft harmonica, brushes", 100, "G major"),
    ("synthwave, retro drum machine, analog synth, neon", 120, "A minor"),
    # Transient-heavy: where int8 saturation hurts most
    ("techno, hard punchy kick drums, fast hi-hats, snare hits, 140 bpm, no melody", 140, "C minor"),
    ("rock drum kit solo, kick snare hi-hat, fast fills, no other instruments", 160, "A minor"),
    ("trap beat, 808 sub bass, snappy snares, claps, hi-hat rolls, sparse", 140, "G minor"),
    ("fingerstyle acoustic guitar, sharp transient plucks, percussive slaps, no vocals", 110, "D major"),
    ("marimba and xylophone duet, percussive mallets, quick attacks", 120, "C major"),
    ("hard rock, distorted power chords, heavy double kick drums, snare cracks", 150, "E minor"),
    # Low-signal: where int8 noise floor hurts most
    ("ambient drone, very quiet, slow evolving texture, sub-bass rumble", 60, "A minor"),
    ("vinyl crackle and room tone, distant piano hits, sparse, dynamic range", 80, "F major"),
]


def _load_session(use_trt: bool) -> Session:
    """Prefer TRT decoder if engines exist (much faster generation).

    Falls back to eager when any required engine is missing.
    """
    if use_trt:
        try:
            engines = default_trt_engines(
                decoder="decoder_mixed_refit_b8_60s",
                vae_encode="vae_encode_fp16_60s",
                vae_decode="vae_decode_fp16_60s",
            )
            for k, p in engines.items():
                if not Path(p).exists():
                    print(f"  [trt] missing {k}: {p}; falling back to eager")
                    use_trt = False
                    break
            if use_trt:
                print("  [trt] using existing 60s engines for decoder + vae")
                return Session(
                    decoder_backend="tensorrt",
                    vae_backend="tensorrt",
                    use_flash_attention=True,
                    trt_engines=engines,
                )
        except Exception as e:
            print(f"  [trt] init failed: {e}; falling back to eager")

    return Session(decoder_backend="eager", vae_backend="eager", use_flash_attention=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-latents", type=int, default=32,
                    help="Total latents to collect (default: 32)")
    ap.add_argument("--frames", type=int, default=250,
                    help="Latent frames per sample. Must equal "
                         "VAEDecodeInt8Config.opt_frames (default: 250 = 10s)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--no-trt", action="store_true",
                    help="Skip TRT decoder fast-path; use eager Session")
    ap.add_argument("--output-dir", default=None,
                    help="Default: <models>/calibration/vae_latents")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else models_dir() / "calibration" / "vae_latents"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Need at least frames worth, but the 60s TRT decoder profile maxes at
    # T=1500 (60s). Cap so we don't trip its profile.
    DECODER_MAX_S = 60.0
    desired = args.frames / 25.0
    if desired >= DECODER_MAX_S:
        duration = DECODER_MAX_S  # exact: T == frames
    else:
        duration = min(desired + 2.0, DECODER_MAX_S)

    print(f"Calibration output: {out_dir}")
    print(f"Per-latent shape:   [1, 64, {args.frames}]  ({args.frames / 25.0:.1f}s)")
    print(f"Generate duration:  {duration:.1f}s, {args.steps} steps, cfg={args.cfg}")
    print(f"Target count:       {args.num_latents}")
    print()

    print("Loading session...")
    t0 = time.time()
    session = _load_session(use_trt=not args.no_trt)
    print(f"  Session ready in {time.time() - t0:.1f}s")

    # Pre-encode each prompt's conditioning once and reuse across seeds.
    print("\nPre-encoding prompts...")
    prompt_packs = []
    for tags, bpm, key in PROMPTS:
        cond = session.encode_text(
            tags=tags, lyrics="[instrumental]",
            instruction=TASK_INSTRUCTIONS["text2music"],
            bpm=bpm, duration=duration, key=key,
        )
        neg = session.null_conditioning(cond)
        prompt_packs.append((cond, neg, tags))
    print(f"  {len(prompt_packs)} prompts ready")

    T = int(round(duration * 25))
    guidance = Curve(tensor=torch.full((T,), args.cfg, dtype=torch.bfloat16))

    saved = 0
    seed = 1000
    print(f"\nGenerating {args.num_latents} latents...")
    while saved < args.num_latents:
        cond, neg, tags = prompt_packs[saved % len(prompt_packs)]

        t0 = time.time()
        latent = session.generate(
            conditioning=cond,
            negative=neg,
            guidance_curve=guidance,
            seed=seed,
            duration=duration,
            steps=args.steps,
            shift=args.shift,
            denoise=1.0,
        )
        gen_dt = time.time() - t0

        lat_bdt = latent.tensor.transpose(1, 2).contiguous()  # [B, D, T]
        if lat_bdt.shape[1] != 64:
            raise RuntimeError(f"Expected D=64, got {lat_bdt.shape[1]}")
        if lat_bdt.shape[2] < args.frames:
            raise RuntimeError(
                f"Latent has {lat_bdt.shape[2]} frames, need >= {args.frames}"
            )

        chunk = lat_bdt[:, :, :args.frames].contiguous().float().cpu()
        path = out_dir / f"latent_{saved:04d}.pt"
        torch.save(chunk, path)

        prompt_short = tags[:40] + ("..." if len(tags) > 40 else "")
        print(f"  [{saved + 1:>3}/{args.num_latents}] seed={seed:<5} dt={gen_dt:5.1f}s  "
              f"{path.name}  ({prompt_short})")

        saved += 1
        seed += 1

    print(f"\nSaved {saved} latents to {out_dir}")


if __name__ == "__main__":
    main()
