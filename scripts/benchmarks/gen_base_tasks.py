"""Test all base-model exclusive tasks with TRT: extract, lego, complete.

Generates a source track via text2music first, then runs each task on it.
All output saved as wav files for listening comparison.

Task mechanics (from upstream generate_audio):
  - extract: context=semantic hints (is_covers=1), chunk_mask=all 1s, generates
    a specific isolated track from the full mix
  - lego: context=raw source latent (is_covers=0), chunk_mask with time range,
    generates a new track fitting the existing audio in a time window
  - complete: context=raw source latent (is_covers=0), chunk_mask=all 1s,
    adds specified instruments to a partial track

Usage:
    uv run python scripts/benchmarks/gen_base_tasks.py
    uv run python scripts/benchmarks/gen_base_tasks.py --extract-track drums --lego-track bass
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

from acestep.constants import TASK_INSTRUCTIONS, TRACK_NAMES
from acestep.engine.session import Session
from acestep.nodes.types import Audio, Curve, Latent, Mask


def save_audio(audio, path):
    wav = audio.waveform.squeeze(0).cpu().numpy().T
    sf.write(path, wav, audio.sample_rate)


def timed_generate(session, cond, *, seed, steps, shift, cfg_scale,
                   context_latent=None, chunk_mask=None):
    """Generate with CFG using the model's learned null embedding."""
    neg_cond = session.null_conditioning(cond)
    if context_latent is not None:
        gc_T = context_latent.tensor.shape[1]
    else:
        gc_T = 1500
    gc = Curve(tensor=torch.full((gc_T,), cfg_scale, dtype=torch.bfloat16))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    latent = session.generate(
        conditioning=cond, seed=seed, steps=steps, shift=shift,
        denoise=1.0, negative=neg_cond, guidance_curve=gc,
        context_latent=context_latent, chunk_mask=chunk_mask,
    )
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t0) * 1000
    return latent, gen_ms


def main():
    parser = argparse.ArgumentParser(description="Test base model exclusive tasks with TRT")
    parser.add_argument("--tags", default="jazz piano trio, brushed drums, walking bass, 140 bpm")
    parser.add_argument("--lyrics", default="[instrumental]")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-track", default="drums",
                        choices=TRACK_NAMES, help="Track to extract")
    parser.add_argument("--lego-track", default="bass",
                        choices=TRACK_NAMES, help="Track to generate with lego")
    parser.add_argument("--complete-tracks", default="drums,bass",
                        help="Comma-separated tracks for complete task")
    parser.add_argument("--lego-start", type=float, default=0.0,
                        help="Lego repainting start time in seconds")
    parser.add_argument("--lego-end", type=float, default=-1,
                        help="Lego repainting end time in seconds (-1 = full)")
    parser.add_argument("--output-dir", default="test_output/base_tasks")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_root = os.path.join(project_root, "checkpoints")
    os.makedirs(args.output_dir, exist_ok=True)

    T = int(args.duration * 25)

    # Load session with TRT
    trt_engine = os.path.join(project_root, "trt_engines",
                              "decoder_base_mixed_b8_60s", "decoder_base_mixed_b8_60s.engine")
    vae_enc = os.path.join(project_root, "trt_engines", "vae_encode_fp16_60s", "vae_encode_fp16_60s.engine")
    vae_dec = os.path.join(project_root, "trt_engines", "vae_decode_fp16_60s", "vae_decode_fp16_60s.engine")

    trt_engines = {"decoder": trt_engine}
    has_vae_trt = os.path.isfile(vae_enc) and os.path.isfile(vae_dec)
    if has_vae_trt:
        trt_engines["vae_encode"] = vae_enc
        trt_engines["vae_decode"] = vae_dec

    print("Loading base model with TRT engines...")
    session = Session(
        project_root=ckpt_root,
        config_path="acestep-v15-base",
        decoder_backend="tensorrt",
        vae_backend="tensorrt" if has_vae_trt else "eager",
        use_flash_attention=True,
        trt_engines=trt_engines,
    )

    results = []

    # ── Step 1: Generate source audio via text2music ──
    print("\n" + "=" * 60)
    print("TEXT2MUSIC (generating source track)")
    print("=" * 60)
    cond = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=args.duration,
        instruction=TASK_INSTRUCTIONS["text2music"],
    )
    latent, gen_ms = timed_generate(
        session, cond, seed=args.seed, steps=args.steps,
        shift=args.shift, cfg_scale=args.cfg,
    )
    audio = session.decode(latent)
    src_path = os.path.join(args.output_dir, "source_text2music.wav")
    save_audio(audio, src_path)
    print(f"  gen={gen_ms:.0f}ms  -> {src_path}")
    results.append(("text2music", gen_ms, "source_text2music.wav"))

    # Prepare source for subsequent tasks
    print("  Preparing source (VAE encode + semantic extract)...")
    source = session.prepare_source(audio)

    # ── Step 2: Extract ──
    # Context = semantic hints (is_covers=1 equivalent)
    # Chunk mask = all 1s (generate everything, model isolates the track)
    print("\n" + "=" * 60)
    print(f"EXTRACT (isolating '{args.extract_track}' track)")
    print("=" * 60)
    extract_instruction = TASK_INSTRUCTIONS["extract"].replace(
        "{TRACK_NAME}", args.extract_track)
    cond_extract = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=args.duration,
        instruction=extract_instruction,
        refer_latent=source.latent,
    )
    latent_extract, gen_ms = timed_generate(
        session, cond_extract, seed=args.seed, steps=args.steps,
        shift=args.shift, cfg_scale=args.cfg,
        context_latent=source.context_latent,
    )
    audio_extract = session.decode(latent_extract)
    fname = f"extract_{args.extract_track}.wav"
    save_audio(audio_extract, os.path.join(args.output_dir, fname))
    print(f"  gen={gen_ms:.0f}ms  -> {fname}")
    results.append(("extract", gen_ms, fname))

    # ── Step 3: Lego ──
    # Context = raw source latent (is_covers=0 equivalent)
    # Chunk mask = 1s in time range to generate, 0s to preserve
    print("\n" + "=" * 60)
    print(f"LEGO (generating '{args.lego_track}' track from context)")
    print("=" * 60)
    lego_instruction = TASK_INSTRUCTIONS["lego"].replace(
        "{TRACK_NAME}", args.lego_track)
    cond_lego = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=args.duration,
        instruction=lego_instruction,
        refer_latent=source.latent,
    )

    # Build lego chunk mask: 1s in the repainting window, 0s elsewhere
    lego_start_frame = int(args.lego_start * 25)
    lego_end_frame = T if args.lego_end < 0 else int(args.lego_end * 25)
    lego_mask_tensor = torch.zeros(T)
    lego_mask_tensor[lego_start_frame:lego_end_frame] = 1.0
    lego_chunk_mask = Mask(tensor=lego_mask_tensor)

    latent_lego, gen_ms = timed_generate(
        session, cond_lego, seed=args.seed, steps=args.steps,
        shift=args.shift, cfg_scale=args.cfg,
        context_latent=source.latent,
        chunk_mask=lego_chunk_mask,
    )
    audio_lego = session.decode(latent_lego)
    fname = f"lego_{args.lego_track}.wav"
    save_audio(audio_lego, os.path.join(args.output_dir, fname))
    print(f"  gen={gen_ms:.0f}ms  -> {fname}")
    results.append(("lego", gen_ms, fname))

    # ── Step 4: Complete ──
    # Context = raw source latent (is_covers=0 equivalent)
    # Chunk mask = all 1s (model uses instruction to know what to add)
    print("\n" + "=" * 60)
    print(f"COMPLETE (adding '{args.complete_tracks}' to source)")
    print("=" * 60)
    complete_instruction = TASK_INSTRUCTIONS["complete"].replace(
        "{TRACK_CLASSES}", args.complete_tracks)
    cond_complete = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=args.duration,
        instruction=complete_instruction,
        refer_latent=source.latent,
    )
    latent_complete, gen_ms = timed_generate(
        session, cond_complete, seed=args.seed, steps=args.steps,
        shift=args.shift, cfg_scale=args.cfg,
        context_latent=source.latent,
    )
    audio_complete = session.decode(latent_complete)
    fname = f"complete_{args.complete_tracks.replace(',', '_')}.wav"
    save_audio(audio_complete, os.path.join(args.output_dir, fname))
    print(f"  gen={gen_ms:.0f}ms  -> {fname}")
    results.append(("complete", gen_ms, fname))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':<15s} {'Gen(ms)':>8s}  File")
    print("-" * 50)
    for task, ms, fname in results:
        print(f"{task:<15s} {ms:>8.0f}  {fname}")
    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
