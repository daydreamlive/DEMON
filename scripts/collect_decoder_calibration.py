#!/usr/bin/env python3
"""Dump XL decoder calibration tensors for FP8 PTQ.

Runs short streaming sessions over a curated prompt mix at the
ringbuffer depth used in production (depth=4) and captures the four
DiT inputs at every TRT decoder call:

  - hidden_states          [B, T, 64]
  - timestep               [B]
  - encoder_hidden_states  [B, L_enc, 2048]   (padded to ENC_LEN)
  - context_latents        [B, T, 128]

The captured tensors are stacked and saved as a single .npz that the
``modelopt.onnx.quantization.quantize`` calibrator can consume. The
ONNX-side FP8 quantizer feeds this through onnxruntime to compute
per-tensor amaxes that become the FP8 scales.

Usage::

    uv run python scripts/collect_decoder_calibration.py
    uv run python scripts/collect_decoder_calibration.py --num-prompts 8 --max-calls 16

Output: ``<MODELS_DIR>/calibration/decoder_xl_fp8/calibration.npz`` plus
a sidecar ``manifest.json`` recording prompt list, engine paths, and
capture shapes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.paths import models_dir, trt_engines_dir


# Curated prompt mix. The first 8 are the original calibration set (the
# spread inherited from the INT8 VAE work in commit 0131b08, plus a few
# extras for harmonic / vocal content). Prompts 9-24 expand coverage
# along axes the original set was thin on:
#   - prominent lead vocals (drives encoder_hidden_states distribution)
#   - extreme transients (percussion-focused, glitch)
#   - wide dynamic range (very quiet to very loud within one prompt)
#   - very-low-frequency dominant content (drum&bass, dub)
#   - very-high-frequency content (sparkly, glassy, shimmering)
#   - dense polyphony (choir, orchestral tutti, layered electronic)
#   - sparse content (solo piano, solo violin, ASMR-quiet)
# These broaden the activation distribution seen by mlp.down_proj and
# attention projections — the layers with structural massive-activation
# patterns. More breadth lets per-tensor and per-channel scales settle
# on the actual amax / p99.9 / per-channel ceiling rather than a
# distribution-of-8 estimate.
PROMPTS: List[tuple[str, int, str]] = [
    # --- Original 8 (preserve so the baseline is reproducible) ---
    ("dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads", 128, "F minor"),
    ("jazz piano trio, brushed drums, walking bass", 140, "Bb major"),
    ("ambient electronic, slow pads, evolving textures", 80, "C minor"),
    ("metal, aggressive guitar riffs, fast double kick, growling vocals", 180, "E minor"),
    ("hip hop beat, 808 bass, trap hi-hats, dark synths", 140, "F# minor"),
    ("classical orchestral, sweeping strings, brass, timpani", 90, "D major"),
    ("acoustic folk, fingerpicked guitar, soft harmonica, brushes", 100, "G major"),
    ("synthwave, retro drum machine, analog synth, neon", 120, "A minor"),
    # --- Vocal-forward (encoder side coverage) ---
    ("pop ballad, emotive female lead vocal, soft piano, strings swell", 72, "Eb major"),
    ("r&b soul, smooth male vocal, electric piano, finger snaps, warm bass", 95, "Db major"),
    ("opera aria, dramatic soprano, lush orchestra, dynamic crescendo", 65, "A major"),
    ("rap verse, fast-flow vocal, minimal beat, sub bass drops", 90, "G minor"),
    # --- Transient-heavy and percussion-extreme ---
    ("breakbeat, chopped amen break, fast snares, dense fills", 165, "B minor"),
    ("idm glitch, granular percussion, micro edits, stutter rhythms", 150, "D minor"),
    ("flamenco, palmas claps, fast strumming, foot stomps", 130, "E phrygian"),
    # --- Very-low-frequency dominant ---
    ("drum and bass, deep sub bass, fast amen drums, jungle atmosphere", 174, "F minor"),
    ("dub reggae, deep bass, sparse skank, tape delay, smoky", 78, "A minor"),
    # --- Very-high-frequency and shimmering ---
    ("glassy ambient, shimmering bells, high-register pads, no bass", 60, "E major"),
    ("post rock crescendo, layered guitars, bright cymbals, soaring leads", 110, "C major"),
    # --- Dense polyphony ---
    ("choral, full SATB choir, rich harmony, cathedral reverb", 60, "F major"),
    ("big band swing, full brass, drums, bass, piano comping, ensemble hits", 160, "Eb major"),
    # --- Sparse / quiet ---
    ("solo piano nocturne, intimate, room tone, minimal ornament", 55, "Bb minor"),
    ("solo violin sonata, expressive vibrato, sparse texture", 70, "G minor"),
    # --- Genre extremes the original missed ---
    ("country americana, slide guitar, fiddle, brushed snare, warm tones", 100, "C major"),
    ("noise drone, harsh distortion, sustained dissonance, no rhythm", 60, "atonal"),
]

# Target encoder length to pad all captures to. Matches the canonical
# enc_opt=200 in TRTBuildConfig; the engine's profile spans [32, 512].
ENC_LEN = 200

# Default seq length (latent frames) for calibration. 1500 matches the
# 60s XL profile. Override per-duration with --seq-len so calibration
# distributions reflect the actual sequence length each engine will see.
SEQ_LEN_DEFAULT = 1500


def _engine_candidates_for(checkpoint: str) -> dict[str, list[str]]:
    if "xl" in checkpoint:
        return {
            "decoder": [
                "decoder_xl-turbo_mixed_refit_b4_60s",
                "decoder_xl-turbo_mixed_refit_b8_60s",
                "decoder_xl-turbo_bf16mix_dynbatch_b8_60s",
            ],
            "vae_encode": ["vae_encode_fp16_60s"],
            "vae_decode": [
                "dreamvae_decode_fp16_60s",
                "vae_decode_fp16_60s",
            ],
        }
    # 2B turbo (acestep-v15-turbo). Prefer the standard fp16 VAE because
    # the dreamvae variant is built less often, so it's more likely to be
    # stale against the current TRT runtime.
    return {
        "decoder": ["decoder_mixed_refit_b8_60s"],
        "vae_encode": ["vae_encode_fp16_60s"],
        "vae_decode": [
            "vae_decode_fp16_60s",
            "dreamvae_decode_fp16_60s",
        ],
    }


def _resolve_engines(
    checkpoint: str,
    decoder_engine_override: str | None = None,
) -> dict[str, str]:
    """Find the TRT engines needed to drive calibration for a checkpoint.

    When ``decoder_engine_override`` is set, the candidate list for the
    decoder is bypassed and the supplied path is used verbatim. VAE
    engines are still picked from the candidate list because they're
    shared across decoder durations.
    """
    root = trt_engines_dir()
    candidates = _engine_candidates_for(checkpoint)
    out: dict[str, str] = {}
    for key, names in candidates.items():
        if key == "decoder" and decoder_engine_override:
            engine_path = Path(decoder_engine_override)
            if not engine_path.exists():
                raise FileNotFoundError(
                    f"Decoder engine not found: {engine_path}"
                )
            out[key] = str(engine_path)
            continue
        for name in names:
            path = root / name / f"{name}.engine"
            if path.exists():
                out[key] = str(path)
                break
        if key not in out:
            raise FileNotFoundError(
                f"Could not locate any {key} engine in {root}; "
                f"tried {names}"
            )
    return out


def _calibration_subdir(checkpoint: str) -> str:
    return "decoder_xl_fp8" if "xl" in checkpoint else "decoder_2b_fp8"


def _patch_pipe_capture(pipe, capture: list[dict]):
    """Wrap StreamPipeline._trt_forward to dump inputs to ``capture``.

    The streaming pipeline owns its own TRT execution context and
    bypasses ``engine._trt_decoder_step`` entirely once it's loaded
    (see ``acestep/engine/stream.py``). Patching the engine method
    therefore captures nothing during streaming. We patch
    ``pipe._trt_forward`` instead, which is the real call site.

    The wrapped call still executes TRT inference so the pipeline
    produces realistic noisy_latent / timestep trajectories at each
    subsequent step. Inputs are converted to numpy on the caller's
    stride before the TRT call writes them, so the snapshot is the
    decoder input distribution as it will appear in production.
    """
    original = pipe._trt_forward

    def _wrap(xt_batch, timestep_list, enc_batch, ctx_batch):
        # Capture full snapshot regardless of batch — the caller filters
        # warmup batches (B < depth) before stacking.
        capture.append({
            "hidden_states": xt_batch.detach().to(torch.float32).cpu().numpy().copy(),
            "timestep": np.asarray(timestep_list, dtype=np.float32),
            "encoder_hidden_states": enc_batch.detach().to(torch.float32).cpu().numpy().copy(),
            "context_latents": ctx_batch.detach().to(torch.float32).cpu().numpy().copy(),
        })
        return original(
            xt_batch=xt_batch,
            timestep_list=timestep_list,
            enc_batch=enc_batch,
            ctx_batch=ctx_batch,
        )

    pipe._trt_forward = _wrap
    return original


def _pad_encoder(enc_np: np.ndarray, target_len: int) -> np.ndarray:
    """Pad encoder_hidden_states along axis=1 to ``target_len`` (zero-pad)
    or truncate if longer. Matches what condition_embedder + attention
    masking handle at inference time.
    """
    L = enc_np.shape[1]
    if L == target_len:
        return enc_np
    if L > target_len:
        return enc_np[:, :target_len, :]
    pad = np.zeros(
        (enc_np.shape[0], target_len - L, enc_np.shape[2]),
        dtype=enc_np.dtype,
    )
    return np.concatenate([enc_np, pad], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=8,
                    help="Number of prompts to cycle through (default: 8).")
    ap.add_argument("--depth", type=int, default=4,
                    help="Ringbuffer pipeline depth (default: 4, "
                         "matches production setup).")
    ap.add_argument("--infer-steps", type=int, default=8,
                    help="Denoise steps per slot (default: 8).")
    ap.add_argument("--shift", type=float, default=3.5)
    ap.add_argument("--max-calls", type=int, default=16,
                    help="Stop after this many DiT calls have been "
                         "captured (default: 16 -> 64 samples at "
                         "batch=4).")
    ap.add_argument("--enc-len", type=int, default=ENC_LEN,
                    help=f"Pad all encoder_hidden_states to this length "
                         f"(default: {ENC_LEN}).")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT,
                    help=f"Latent sequence length T (frames at 25 Hz) to "
                         f"capture. 1500/3000/6000 correspond to the "
                         f"60s/120s/240s engine profiles. Default: "
                         f"{SEQ_LEN_DEFAULT}.")
    ap.add_argument("--decoder-engine", type=str, default=None,
                    help="Full path to the decoder .engine to drive "
                         "calibration through. When set, the candidate "
                         "lookup is bypassed. Use this to match the cal "
                         "run to the engine duration (e.g., the 120s "
                         "bf16 engine for --seq-len 3000).")
    ap.add_argument("--checkpoint", type=str, default="acestep-v15-xl-turbo",
                    help="Model checkpoint directory name "
                         "(default: acestep-v15-xl-turbo; use "
                         "'acestep-v15-turbo' for the 2B production "
                         "decoder).")
    ap.add_argument("--output", type=str, default=None,
                    help="Destination .npz. Default: "
                         "<MODELS_DIR>/calibration/decoder_<2b|xl>_fp8/"
                         "calibration.npz, picked from --checkpoint.")
    args = ap.parse_args()

    subdir = _calibration_subdir(args.checkpoint)
    out_path = Path(args.output) if args.output else (
        models_dir() / "calibration" / subdir / "calibration.npz"
    )
    # `models_dir()` already ends in .../models/demon; the calibration
    # subdir hangs off that root next to trt_engines/.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Checkpoint:    {args.checkpoint}")
    print(f"[setup] Output target: {out_path}")

    print(f"[setup] Locating TRT engines for {args.checkpoint}...")
    engines = _resolve_engines(
        args.checkpoint,
        decoder_engine_override=args.decoder_engine,
    )
    for k, v in engines.items():
        print(f"  {k}: {v}")

    print(f"[setup] Loading {args.checkpoint} session...")
    session = Session(
        config_path=args.checkpoint,
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=engines,
    )
    handler = session.handler
    engine = handler._diffusion_engine
    device = handler.device
    dtype = handler.dtype
    print(f"  device={device} dtype={dtype}")

    cfg = DiffusionConfig(
        infer_steps=args.infer_steps,
        shift=args.shift,
        noise_on_cpu=True,
    )
    pipe = StreamPipeline(engine, cfg, pipeline_depth=args.depth)
    print(f"[setup] StreamPipeline depth={pipe.depth}, infer_steps={cfg.infer_steps}")

    capture: list[dict] = []
    _patch_pipe_capture(pipe, capture)

    prompts = PROMPTS[:args.num_prompts]
    print(f"[capture] Running {len(prompts)} prompts until {args.max_calls} "
          f"DiT calls are captured...")
    print(f"  (hard cap: {args.max_calls * 4 + args.depth + 8} ticks; bails "
          f"if the patch isn't being hit)")

    # Zero context_latents matches how the user runs cover-graph today
    # (no audio prompt in the simplest case). The DiT activations come
    # primarily from the noisy latent + text encoder output; context is
    # the reference latent path which the runtime feeds zeros into when
    # no source audio is set. This is fine for calibration coverage of
    # the bulk MatMul activations.
    T = args.seq_len
    D_ctx = 64
    ctx_lat = torch.zeros(1, T, D_ctx, device=device, dtype=dtype)
    ctx_mask = torch.ones(1, T, D_ctx, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, ctx_mask], dim=-1)

    t0 = time.perf_counter()
    submitted = 0
    ticks = 0
    next_prompt = 0
    # We count steady-state (B==depth) captures, not warmup captures.
    # Warmup takes ``depth`` ticks before the ringbuffer is saturated;
    # everything after is at batch=depth.
    tick_cap = args.max_calls + args.depth * 3 + 8

    def _steady_count() -> int:
        return sum(
            1 for c in capture
            if c["hidden_states"].shape[0] == args.depth
        )

    while _steady_count() < args.max_calls:
        if ticks >= tick_cap:
            raise RuntimeError(
                f"Hard cap hit: {ticks} ticks, {len(capture)} captures, "
                f"{_steady_count()} steady-state. _trt_forward patch may "
                f"not be applied to the right pipe instance."
            )
        # Submit one request per tick, cycling through prompts. The
        # depth=4 ringbuffer ensures every tick's DiT call sees 4 slots
        # at different stages of their schedules; this is exactly the
        # heterogeneous batch=4 activation distribution we want to
        # calibrate against.
        prompt, bpm, key = prompts[next_prompt % len(prompts)]
        next_prompt += 1
        cond = session.encode_text(
            tags=prompt,
            instruction=TASK_INSTRUCTIONS["text2music"],
            bpm=bpm, duration=60.0, key=key,
        )
        entry = cond.to_entries()[0]
        pipe.submit(SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=42 + submitted,
        ))
        submitted += 1
        pipe.tick()
        ticks += 1
        if ticks % 4 == 0:
            print(f"  ticks={ticks}  captured={len(capture)}  "
                  f"submitted={submitted}")

    elapsed = time.perf_counter() - t0
    print(f"[capture] Done: {len(capture)} DiT calls in {elapsed:.1f}s "
          f"({elapsed / len(capture) * 1000:.1f} ms/call)")

    # Drop warmup captures (B < depth). During the first ``depth``
    # ticks the ring-buffer is not yet saturated, so the DiT runs at
    # batch=1, 2, 3, then 4. Calibration must see only steady-state
    # batch=depth activations so the per-tensor amax reflects what
    # production will see.
    pre = len(capture)
    capture = [c for c in capture if c["hidden_states"].shape[0] == args.depth]
    dropped = pre - len(capture)
    if dropped:
        print(f"[stack] Dropped {dropped} warmup captures with B<{args.depth}; "
              f"{len(capture)} steady-state samples remaining")
    if not capture:
        raise RuntimeError(
            f"No steady-state captures (batch={args.depth}). "
            f"Either max_calls is too small or the ringbuffer never warmed up."
        )

    # Stack captured tensors. encoder_hidden_states must be padded to a
    # uniform length so they stack cleanly; everything else already has
    # uniform shape from the fixed depth + seq profile.
    print(f"[stack] Padding encoder_hidden_states to L_enc={args.enc_len}")
    hs = np.stack([c["hidden_states"] for c in capture], axis=0)
    ts = np.stack([c["timestep"] for c in capture], axis=0)
    enc = np.stack(
        [_pad_encoder(c["encoder_hidden_states"], args.enc_len) for c in capture],
        axis=0,
    )
    ctx = np.stack([c["context_latents"] for c in capture], axis=0)
    # Stacking adds a leading dim; collapse it back into batch axis so
    # ModelOpt sees axis-0 as the calibration sample axis.
    hs = hs.reshape(-1, *hs.shape[2:])
    ts = ts.reshape(-1, *ts.shape[2:])
    enc = enc.reshape(-1, *enc.shape[2:])
    ctx = ctx.reshape(-1, *ctx.shape[2:])
    print(f"  hidden_states={hs.shape} dtype={hs.dtype}")
    print(f"  timestep={ts.shape} dtype={ts.dtype}")
    print(f"  encoder_hidden_states={enc.shape} dtype={enc.dtype}")
    print(f"  context_latents={ctx.shape} dtype={ctx.dtype}")

    # ModelOpt's CalibrationDataProvider asserts that batch sizes
    # match the model's expected input shapes after splitting along
    # axis 0. We save the stacked arrays as-is; the build script
    # decides the per-iteration batch.
    np.savez_compressed(
        str(out_path),
        hidden_states=hs.astype(np.float32),
        timestep=ts.astype(np.float32),
        encoder_hidden_states=enc.astype(np.float32),
        context_latents=ctx.astype(np.float32),
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[save] Wrote {out_path} ({size_mb:.1f} MB)")

    manifest = {
        "schema_version": 1,
        "engine_paths": engines,
        "prompts": [{"text": p, "bpm": b, "key": k} for (p, b, k) in prompts],
        "depth": args.depth,
        "infer_steps": args.infer_steps,
        "shift": args.shift,
        "captures": len(capture),
        "stacked_samples": int(hs.shape[0]),
        "shapes": {
            "hidden_states": list(hs.shape),
            "timestep": list(ts.shape),
            "encoder_hidden_states": list(enc.shape),
            "context_latents": list(ctx.shape),
        },
        "enc_len": args.enc_len,
        "seq_len": args.seq_len,
    }
    (out_path.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"[save] Wrote manifest -> {out_path.parent / 'manifest.json'}")


if __name__ == "__main__":
    main()
