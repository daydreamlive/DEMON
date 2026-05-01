"""Generate audio with the base model and save to disk.

Generates 2 wavs: PyTorch CFG and TRT CFG (batched).
Same seed so differences are purely from backend/precision.

Usage:
    uv run python scripts/benchmarks/gen_base_trt.py
    uv run python scripts/benchmarks/gen_base_trt.py --tags "jazz piano trio" --steps 50 --cfg 7.5
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import soundfile as sf
import numpy as np
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Curve


def generate_and_save(session, label, out_dir, *, tags, lyrics, duration,
                      steps, shift, seed, cfg_scale):
    """Encode, generate with CFG, decode, save. Returns timing dict."""
    cond = session.encode_text(
        tags=tags, lyrics=lyrics, duration=duration,
        instruction=TASK_INSTRUCTIONS["text2music"],
    )

    # Null conditioning using the model's learned null_condition_emb
    neg_cond = session.null_conditioning(cond)
    T = int(duration * 25)
    gc = Curve(tensor=torch.full((T,), cfg_scale, dtype=torch.bfloat16))

    print(f"  [{label}] generating ({steps} steps, cfg={cfg_scale})...", end=" ", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    latent = session.generate(
        conditioning=cond, seed=seed, steps=steps, shift=shift,
        denoise=1.0, negative=neg_cond, guidance_curve=gc,
    )
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio = session.decode(latent)
    torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) * 1000

    filename = f"{label}_{steps}s_cfg{cfg_scale}_seed{seed}.wav"
    out_path = os.path.join(out_dir, filename)
    wav = audio.waveform.squeeze(0).cpu().numpy().T
    sf.write(out_path, wav, audio.sample_rate)

    print(f"gen={gen_ms:.0f}ms  decode={dec_ms:.0f}ms  -> {filename}")
    return {"label": label, "gen_ms": gen_ms, "dec_ms": dec_ms,
            "total_ms": gen_ms + dec_ms, "file": filename}


def main():
    parser = argparse.ArgumentParser(description="Generate base model audio: PT vs TRT with CFG")
    parser.add_argument("--tags", default="jazz piano trio, brushed drums, walking bass, 140 bpm")
    parser.add_argument("--lyrics", default="[instrumental]")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="test_output/base_trt")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_root = os.path.join(project_root, "checkpoints")
    os.makedirs(args.output_dir, exist_ok=True)

    gen_kwargs = dict(
        tags=args.tags, lyrics=args.lyrics, duration=args.duration,
        steps=args.steps, shift=args.shift, seed=args.seed,
        cfg_scale=args.cfg,
    )
    results = []

    # ── PyTorch ──
    print("=" * 60)
    print("PYTORCH (base model)")
    print("=" * 60)
    session_pt = Session(
        project_root=ckpt_root,
        config_path="acestep-v15-base",
        use_flash_attention=True,
    )
    results.append(generate_and_save(
        session_pt, "pt", args.output_dir, **gen_kwargs))
    del session_pt
    torch.cuda.empty_cache()

    # ── TRT ──
    trt_engine = os.path.join(project_root, "trt_engines",
                              "decoder_base_mixed_b8_60s", "decoder_base_mixed_b8_60s.engine")
    vae_enc = os.path.join(project_root, "trt_engines", "vae_encode_fp16_60s", "vae_encode_fp16_60s.engine")
    vae_dec = os.path.join(project_root, "trt_engines", "vae_decode_fp16_60s", "vae_decode_fp16_60s.engine")

    trt_engines = {"decoder": trt_engine}
    has_vae_trt = os.path.isfile(vae_enc) and os.path.isfile(vae_dec)
    if has_vae_trt:
        trt_engines["vae_encode"] = vae_enc
        trt_engines["vae_decode"] = vae_dec

    print("\n" + "=" * 60)
    print("TRT (base model)")
    print("=" * 60)
    session_trt = Session(
        project_root=ckpt_root,
        config_path="acestep-v15-base",
        decoder_backend="tensorrt",
        vae_backend="tensorrt" if has_vae_trt else "eager",
        use_flash_attention=True,
        trt_engines=trt_engines,
    )
    results.append(generate_and_save(
        session_trt, "trt", args.output_dir, **gen_kwargs))
    del session_trt
    torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Label':<25s} {'Gen(ms)':>8s} {'Dec(ms)':>8s} {'Total':>8s}  File")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<25s} {r['gen_ms']:>8.0f} {r['dec_ms']:>8.0f} "
              f"{r['total_ms']:>8.0f}  {r['file']}")

    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
