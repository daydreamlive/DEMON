"""Compare upstream generate_audio() vs our Session pipeline (PyTorch only).

Calls the raw model.generate_audio() with the exact upstream parameter setup
for each task, then runs the same task through our Session pipeline (no TRT).
Saves both outputs for A/B listening comparison.

Usage:
    uv run python scripts/benchmarks/compare_upstream.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Audio, Curve, Latent, Mask


DEVICE = "cuda"
DTYPE = torch.bfloat16
DURATION = 60.0
SEED = 42
SHIFT = 1.0
INFER_STEPS = 50
CFG_SCALE = 7.5
SAMPLE_RATE = 48000


def save_audio(audio, path):
    if isinstance(audio, Audio):
        wav = audio.waveform.squeeze(0).cpu().numpy().T
        sf.write(path, wav, audio.sample_rate)
    else:
        # Raw numpy array [2, samples]
        sf.write(path, audio.T, SAMPLE_RATE)


# ── Upstream path: call model.generate_audio() directly ──

@torch.no_grad()
def upstream_text_encode(handler, tags, lyrics, instruction):
    """Encode text/lyrics the same way as our TextEncode node."""
    meta_cap = (
        f"- bpm: 120\n"
        f"- timesignature: 4\n"
        f"- keyscale: C major\n"
        f"- duration: {DURATION}\n"
    )
    text_prompt = (
        f"# Instruction\n{instruction}\n\n"
        f"# Caption\n{tags}\n\n"
        f"# Metas\n{meta_cap}"
        f"<|endoftext|>\n"
    )
    lyrics_prompt = f"# Languages\nen\n\n# Lyric\n{lyrics}<|endoftext|><|endoftext|>"

    with handler._load_model_context("text_encoder"):
        tokens = handler.text_tokenizer(
            text_prompt, return_tensors="pt", add_special_tokens=False
        )
        text_hidden = handler.infer_text_embeddings(
            tokens["input_ids"].to(DEVICE)
        )
        text_mask = tokens["attention_mask"].to(DEVICE).bool()

        lyric_tokens = handler.text_tokenizer(
            lyrics_prompt, return_tensors="pt", add_special_tokens=False
        )
        lyric_hidden = handler.infer_lyric_embeddings(
            lyric_tokens["input_ids"].to(DEVICE)
        )
        lyric_mask = torch.ones(
            lyric_hidden.shape[:2], device=DEVICE, dtype=torch.bool
        )

    return text_hidden, text_mask, lyric_hidden, lyric_mask


@torch.no_grad()
def upstream_generate(model, text_hidden, text_mask, lyric_hidden, lyric_mask,
                      silence_latent, *,
                      src_latents, chunk_masks, is_covers,
                      refer_packed, refer_order_mask,
                      precomputed_lm_hints_25Hz=None,
                      seed=SEED):
    """Call model.generate_audio() with exact upstream parameters."""
    result = model.generate_audio(
        text_hidden_states=text_hidden.to(DTYPE),
        text_attention_mask=text_mask,
        lyric_hidden_states=lyric_hidden.to(DTYPE),
        lyric_attention_mask=lyric_mask,
        refer_audio_acoustic_hidden_states_packed=refer_packed,
        refer_audio_order_mask=refer_order_mask,
        src_latents=src_latents,
        chunk_masks=chunk_masks,
        is_covers=is_covers,
        silence_latent=silence_latent,
        seed=seed,
        infer_steps=INFER_STEPS,
        infer_method="ode",
        use_cache=True,
        shift=SHIFT,
        diffusion_guidance_sale=CFG_SCALE,
        precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
        use_progress_bar=True,
    )
    return result["target_latents"]


def decode_latent(session, latents):
    """Decode latents through our Session VAE (same VAE either way)."""
    return session.decode(Latent(tensor=latents))


# ── Session path: our pipeline in PyTorch ──

def session_generate(session, cond, *, context_latent=None, chunk_mask=None):
    """Run generation through our Session pipeline (no TRT).

    Uses noise_on_cpu=False and use_cache=True to match upstream's
    model.generate_audio() defaults exactly.
    """
    from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate

    neg_cond = session.null_conditioning(cond)
    T_gc = context_latent.tensor.shape[1] if context_latent is not None else int(DURATION * 25)
    gc = Curve(tensor=torch.full((T_gc,), CFG_SCALE, dtype=DTYPE))

    config = DiffusionConfigNode().execute(
        steps=INFER_STEPS, shift=SHIFT, seed=SEED, denoise=1.0,
        method="ode",
        noise_on_cpu=False,   # match upstream: GPU noise in [B,T,D]
        use_cache=True,       # match upstream: KV caching enabled
    )["config"]

    return Generate().execute(
        model=session.model,
        config=config,
        positive=cond,
        negative=neg_cond,
        guidance_curve=gc,
        context_latent=context_latent,
        chunk_mask=chunk_mask,
    )["latent"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default="jazz piano trio, brushed drums, walking bass, 140 bpm")
    parser.add_argument("--lyrics", default="[instrumental]")
    parser.add_argument("--extract-track", default="drums")
    parser.add_argument("--lego-track", default="bass")
    parser.add_argument("--complete-tracks", default="drums,bass")
    parser.add_argument("--output-dir", default="test_output/compare_upstream")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_root = os.path.join(project_root, "checkpoints")
    T = int(DURATION * 25)

    # ── Load Session (PyTorch, no TRT) ──
    print("Loading base model (PyTorch, no TRT)...")
    session = Session(
        project_root=ckpt_root,
        config_path="acestep-v15-base",
        use_flash_attention=True,
    )
    handler = session.handler
    model = handler.model
    handler._ensure_silence_latent_on_device()
    silence_latent = handler.silence_latent

    # ── Step 1: text2music source generation (upstream) ──
    print("\n" + "=" * 60)
    print("UPSTREAM: text2music (generating source)")
    print("=" * 60)

    text_hidden, text_mask, lyric_hidden, lyric_mask = upstream_text_encode(
        handler, args.tags, args.lyrics, TASK_INSTRUCTIONS["text2music"]
    )

    # Timbre reference = silence (no source for text2music)
    refer_packed = silence_latent[:, :750, :].to(DTYPE)
    refer_order_mask = torch.zeros(1, device=DEVICE, dtype=torch.long)

    # Source latents = silence
    src = silence_latent[:, :T, :].clone().to(DTYPE)
    if src.shape[1] < T:
        src = F.pad(src, (0, 0, 0, T - src.shape[1]))

    chunk_masks = torch.ones(1, T, 64, device=DEVICE, dtype=DTYPE)
    is_covers = torch.zeros(1, device=DEVICE, dtype=DTYPE)

    with handler._load_model_context("model"):
        t0 = time.perf_counter()
        source_latent_upstream = upstream_generate(
            model, text_hidden, text_mask, lyric_hidden, lyric_mask,
            silence_latent,
            src_latents=src,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            refer_packed=refer_packed,
            refer_order_mask=refer_order_mask,
        )
        t_gen = (time.perf_counter() - t0) * 1000
    print(f"  gen={t_gen:.0f}ms  latent range=[{source_latent_upstream.min():.3f}, {source_latent_upstream.max():.3f}]")

    source_audio_upstream = decode_latent(session, source_latent_upstream)
    save_audio(source_audio_upstream, os.path.join(args.output_dir, "source_upstream.wav"))

    # Prepare source audio for downstream tasks
    source = session.prepare_source(source_audio_upstream)

    # Pre-compute semantic hints for upstream extract path
    with handler._load_model_context("model"):
        source_for_hints = source.latent.tensor.to(DTYPE).to(DEVICE)
        hints_upstream = model.tokenizer.tokenize(source_for_hints)
        lm_hints_5Hz = hints_upstream[0]
        precomputed_hints = model.detokenizer(lm_hints_5Hz)[:, :T, :]
    print(f"  Precomputed hints: shape={precomputed_hints.shape}")

    # ── Step 2: Session text2music (for comparison) ──
    print("\n" + "=" * 60)
    print("SESSION: text2music")
    print("=" * 60)

    cond_t2m = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=DURATION,
        instruction=TASK_INSTRUCTIONS["text2music"],
    )
    t0 = time.perf_counter()
    latent_t2m = session_generate(session, cond_t2m)
    t_gen = (time.perf_counter() - t0) * 1000
    audio_t2m = session.decode(latent_t2m)
    save_audio(audio_t2m, os.path.join(args.output_dir, "source_session.wav"))
    print(f"  gen={t_gen:.0f}ms")

    # ================================================================
    # EXTRACT
    # ================================================================
    extract_instr = TASK_INSTRUCTIONS["extract"].replace("{TRACK_NAME}", args.extract_track)

    # ── Upstream extract ──
    print("\n" + "=" * 60)
    print(f"UPSTREAM: extract '{args.extract_track}'")
    print("=" * 60)

    text_hidden_ext, text_mask_ext, lyric_hidden_ext, lyric_mask_ext = upstream_text_encode(
        handler, args.tags, args.lyrics, extract_instr
    )

    # For extract: is_covers=1, precomputed_lm_hints provided, src_latents=raw source
    # Timbre reference = source latent (not silence)
    refer_packed_src = source.latent.tensor[:, :, :].to(DTYPE).to(DEVICE)
    is_covers_extract = torch.ones(1, device=DEVICE, dtype=DTYPE)

    with handler._load_model_context("model"):
        t0 = time.perf_counter()
        latent_extract_up = upstream_generate(
            model, text_hidden_ext, text_mask_ext,
            lyric_hidden_ext, lyric_mask_ext, silence_latent,
            src_latents=source.latent.tensor.to(DTYPE).to(DEVICE),
            chunk_masks=chunk_masks,
            is_covers=is_covers_extract,
            refer_packed=refer_packed_src,
            refer_order_mask=refer_order_mask,
            precomputed_lm_hints_25Hz=precomputed_hints,
        )
        t_gen = (time.perf_counter() - t0) * 1000
    print(f"  gen={t_gen:.0f}ms  latent range=[{latent_extract_up.min():.3f}, {latent_extract_up.max():.3f}]")

    audio_extract_up = decode_latent(session, latent_extract_up)
    save_audio(audio_extract_up, os.path.join(args.output_dir, f"extract_{args.extract_track}_upstream.wav"))

    # ── Session extract ──
    print(f"\nSESSION: extract '{args.extract_track}'")
    cond_extract = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=DURATION,
        instruction=extract_instr,
        refer_latent=source.latent,
    )
    t0 = time.perf_counter()
    latent_extract_sess = session_generate(
        session, cond_extract,
        context_latent=source.context_latent,
    )
    t_gen = (time.perf_counter() - t0) * 1000
    audio_extract_sess = session.decode(latent_extract_sess)
    save_audio(audio_extract_sess, os.path.join(args.output_dir, f"extract_{args.extract_track}_session.wav"))
    print(f"  gen={t_gen:.0f}ms")

    # ================================================================
    # LEGO
    # ================================================================
    lego_instr = TASK_INSTRUCTIONS["lego"].replace("{TRACK_NAME}", args.lego_track)

    # ── Upstream lego ──
    print("\n" + "=" * 60)
    print(f"UPSTREAM: lego '{args.lego_track}'")
    print("=" * 60)

    text_hidden_lego, text_mask_lego, lyric_hidden_lego, lyric_mask_lego = upstream_text_encode(
        handler, args.tags, args.lyrics, lego_instr
    )

    # Lego: is_covers=0, context=raw source, chunk_mask with time window
    lego_mask = torch.zeros(1, T, 64, device=DEVICE, dtype=DTYPE)
    lego_mask[:, :, :] = 1.0  # Full duration for now (same as test)

    with handler._load_model_context("model"):
        t0 = time.perf_counter()
        latent_lego_up = upstream_generate(
            model, text_hidden_lego, text_mask_lego,
            lyric_hidden_lego, lyric_mask_lego, silence_latent,
            src_latents=source.latent.tensor.to(DTYPE).to(DEVICE),
            chunk_masks=lego_mask,
            is_covers=is_covers,  # 0
            refer_packed=refer_packed_src,
            refer_order_mask=refer_order_mask,
        )
        t_gen = (time.perf_counter() - t0) * 1000
    print(f"  gen={t_gen:.0f}ms  latent range=[{latent_lego_up.min():.3f}, {latent_lego_up.max():.3f}]")

    audio_lego_up = decode_latent(session, latent_lego_up)
    save_audio(audio_lego_up, os.path.join(args.output_dir, f"lego_{args.lego_track}_upstream.wav"))

    # ── Session lego ──
    print(f"\nSESSION: lego '{args.lego_track}'")
    cond_lego = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=DURATION,
        instruction=lego_instr,
        refer_latent=source.latent,
    )
    t0 = time.perf_counter()
    latent_lego_sess = session_generate(
        session, cond_lego,
        context_latent=source.latent,
    )
    t_gen = (time.perf_counter() - t0) * 1000
    audio_lego_sess = session.decode(latent_lego_sess)
    save_audio(audio_lego_sess, os.path.join(args.output_dir, f"lego_{args.lego_track}_session.wav"))
    print(f"  gen={t_gen:.0f}ms")

    # ================================================================
    # COMPLETE
    # ================================================================
    complete_instr = TASK_INSTRUCTIONS["complete"].replace("{TRACK_CLASSES}", args.complete_tracks)

    # ── Upstream complete ──
    print("\n" + "=" * 60)
    print(f"UPSTREAM: complete '{args.complete_tracks}'")
    print("=" * 60)

    text_hidden_comp, text_mask_comp, lyric_hidden_comp, lyric_mask_comp = upstream_text_encode(
        handler, args.tags, args.lyrics, complete_instr
    )

    with handler._load_model_context("model"):
        t0 = time.perf_counter()
        latent_comp_up = upstream_generate(
            model, text_hidden_comp, text_mask_comp,
            lyric_hidden_comp, lyric_mask_comp, silence_latent,
            src_latents=source.latent.tensor.to(DTYPE).to(DEVICE),
            chunk_masks=chunk_masks,  # all ones
            is_covers=is_covers,  # 0
            refer_packed=refer_packed_src,
            refer_order_mask=refer_order_mask,
        )
        t_gen = (time.perf_counter() - t0) * 1000
    print(f"  gen={t_gen:.0f}ms  latent range=[{latent_comp_up.min():.3f}, {latent_comp_up.max():.3f}]")

    audio_comp_up = decode_latent(session, latent_comp_up)
    save_audio(audio_comp_up, os.path.join(args.output_dir, f"complete_{args.complete_tracks.replace(',','_')}_upstream.wav"))

    # ── Session complete ──
    print(f"\nSESSION: complete '{args.complete_tracks}'")
    cond_comp = session.encode_text(
        tags=args.tags, lyrics=args.lyrics, duration=DURATION,
        instruction=complete_instr,
        refer_latent=source.latent,
    )
    t0 = time.perf_counter()
    latent_comp_sess = session_generate(
        session, cond_comp,
        context_latent=source.latent,
    )
    t_gen = (time.perf_counter() - t0) * 1000
    audio_comp_sess = session.decode(latent_comp_sess)
    save_audio(audio_comp_sess, os.path.join(args.output_dir, f"complete_{args.complete_tracks.replace(',','_')}_session.wav"))
    print(f"  gen={t_gen:.0f}ms")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DONE - compare upstream vs session WAVs in:")
    print(f"  {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
