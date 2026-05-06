"""Standalone bench for the 5Hz-LM hint generation pipeline.

Loads the 5Hz LM, generates audio codes for a fixed-duration prompt,
runs them through the DiT's FSQ quantizer + detokenizer to produce a
25Hz hint latent, optionally VAE-decodes that latent to a wav so you
can ear-test it. Reports load/generate/dequant timings + VRAM peak.

This is the test that should have run *before* wiring LM hints into
the demo; if shapes / dtypes / vocabulary ranges don't line up, this
fails loud and isolated rather than mid-stream.

Usage::

    python tests/benchmarks/bench_lm_hints.py
    python tests/benchmarks/bench_lm_hints.py --duration 30 --decode
    python tests/benchmarks/bench_lm_hints.py --tags "ambient drone, slow" --duration 60 --decode
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

# Avoid Windows cp1252 crash when transformers prints emoji deprecation
# warnings during model load.
os.environ.setdefault("PYTHONUTF8", "1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", default="heavy thrash metal, distorted guitars, fast double kick, aggressive vocals")
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--bpm", type=int, default=160)
    parser.add_argument("--key", default="E minor")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="seconds of audio to plan for (5 codes per second)")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1528)
    parser.add_argument("--decode", action="store_true",
                        help="VAE-decode the LM-hint latent to a wav for ear testing")
    parser.add_argument("--out-dir", default="bench_outputs/lm_hints")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- LM load ----
    from acestep.lm_hints import LMHintGenerator

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU free before LM load: {free_before / 1024**3:.2f} / "
          f"{total / 1024**3:.2f} GiB")

    print("\n[1] Loading 5Hz LM...")
    t0 = time.perf_counter()
    gen = LMHintGenerator()
    lm_load_s = time.perf_counter() - t0
    lm_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  load: {lm_load_s:.2f} s")
    print(f"  VRAM peak after LM load: {lm_vram:.2f} GiB")

    # ---- Code generation ----
    print(f"\n[2] Generating codes for {args.duration:.1f}s...")
    target_T_5Hz = int(round(args.duration * 5))
    print(f"  target T_5Hz = {target_T_5Hz}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    codes = gen.generate_codes(
        tags=args.tags,
        lyrics=args.lyrics,
        bpm=args.bpm,
        key=args.key,
        duration=args.duration,
        temperature=args.temperature,
        seed=args.seed,
    )
    gen_s = time.perf_counter() - t0
    gen_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  generate: {gen_s:.2f} s")
    print(f"  codes shape: {tuple(codes.shape)} dtype={codes.dtype}")
    print(f"  codes range: [{codes.min().item()}, {codes.max().item()}]")
    print(f"  unique codes: {codes.unique().numel()} of {target_T_5Hz}")
    print(f"  first 16: {codes[0, :16].tolist()}")
    print(f"  last 16:  {codes[0, -16:].tolist()}")
    print(f"  VRAM peak during generate: {gen_vram:.2f} GiB")

    # ---- Dequant via DiT FSQ + detokenizer ----
    print("\n[3] Dequantizing via DiT FSQ codebook + detokenizer...")
    from acestep.engine.session import Session

    # Spin up a Session purely to access the DiT main model. We don't
    # need TRT engines here — just the DiT's nn.Module .tokenizer and
    # .detokenizer. Using eager backend avoids loading any TRT engines.
    session = Session(
        decoder_backend="eager",
        vae_backend="eager",
    )

    handler = session.handler
    target_T_25Hz = int(round(args.duration * 25))
    t0 = time.perf_counter()
    with handler._load_model_context("model"):
        dit = handler.model
        quantizer = dit.tokenizer.quantizer
        codebook_size = quantizer.codebooks.shape[1]
        print(f"  codebook size: {codebook_size}")
        codes_clamped = codes.clamp(0, codebook_size - 1).unsqueeze(-1)
        print(f"  codes_clamped shape: {tuple(codes_clamped.shape)}")
        lm_hints_5Hz = quantizer.get_output_from_indices(codes_clamped)
        print(f"  lm_hints_5Hz shape: {tuple(lm_hints_5Hz.shape)} "
              f"abs_mean={lm_hints_5Hz.abs().mean().item():.4f}")
        lm_hints_25Hz = dit.detokenizer(lm_hints_5Hz)
    lm_hints_25Hz = lm_hints_25Hz[:, :target_T_25Hz, :]
    dequant_s = time.perf_counter() - t0
    print(f"  dequant: {dequant_s:.2f} s")
    print(f"  lm_hints_25Hz shape: {tuple(lm_hints_25Hz.shape)} "
          f"abs_mean={lm_hints_25Hz.abs().mean().item():.4f}")

    # ---- Optional ear test: VAE-decode the hint latent ----
    decoded_path = None
    decode_s = None
    if args.decode:
        print("\n[4] VAE-decoding the LM-hint latent (ear test)...")
        from acestep.nodes.types import Latent
        from acestep.nodes.vae_nodes import VAEDecodeAudio
        import soundfile as sf

        latent = Latent(tensor=lm_hints_25Hz)
        t0 = time.perf_counter()
        audio = VAEDecodeAudio().execute(vae=session.vae, latent=latent)["audio"]
        decode_s = time.perf_counter() - t0
        print(f"  decode: {decode_s:.2f} s")
        wav = audio.waveform.detach().float().cpu().numpy()
        if wav.ndim == 3:
            wav = wav[0]
        wav = wav.T  # [T, C] for soundfile
        decoded_path = out_dir / f"lm_hints_{int(args.duration)}s.wav"
        sf.write(str(decoded_path), wav, audio.sample_rate)
        print(f"  wrote: {decoded_path}")

    # ---- Summary ----
    total_vram = torch.cuda.max_memory_allocated() / 1024**3
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  LM load:    {lm_load_s:.2f} s")
    print(f"  Generate:   {gen_s:.2f} s   ({target_T_5Hz} codes)")
    print(f"  Dequant:    {dequant_s:.2f} s")
    if decode_s is not None:
        print(f"  VAE decode: {decode_s:.2f} s")
    print(f"  Peak VRAM:  {total_vram:.2f} GiB")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "tags": args.tags,
            "duration": args.duration,
            "target_T_5Hz": target_T_5Hz,
            "load_seconds": lm_load_s,
            "generate_seconds": gen_s,
            "dequant_seconds": dequant_s,
            "decode_seconds": decode_s,
            "peak_vram_gib": total_vram,
            "codes_first_16": codes[0, :16].tolist(),
            "codes_last_16": codes[0, -16:].tolist(),
            "codes_unique": int(codes.unique().numel()),
            "decoded_wav": str(decoded_path) if decoded_path else None,
        }, indent=2), encoding="utf-8")
        print(f"\nJSON summary -> {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
