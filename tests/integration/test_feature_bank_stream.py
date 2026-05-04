"""GPU smoke test: StreamPipeline with feature bank enabled (eager path).

Runs the pipeline twice with identical seeds and prompts -- once
without the feature bank, once with it -- and prints a per-song
diff between the two latent streams.

Expected:
- Song 1 should match between bank-off and bank-on runs to within
  SDPA numeric noise. (Bank reads are no-ops on empty entries; bank
  writes happen but don't affect this song.)
- Songs 2+ should diverge measurably under bank-on, because each
  attention call now sees the prior song's K/V at the same denoise
  step.

This is a runnability + plumbing check, not a quality check. Whether
the divergence is *musically* meaningful (timbre/identity transfer)
is the open empirical question and requires listening, not diffing.

Run from repo root:

    .venv/Scripts/python.exe tests/integration/test_feature_bank_stream.py

Saves the resulting latents to ``./feature_bank_smoke_out/`` so the
caller can decode and listen separately.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.session import Session
from acestep.engine.stream import SlotRequest, StreamPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent / "feature_bank_smoke_out"

PROMPTS: List[str] = [
    "ambient drone, slow swell, dark cinematic",
    "ambient drone, slow swell, dark cinematic",
    "ambient drone, slow swell, dark cinematic",
]
SEEDS: List[int] = [1234, 5678, 9012]

DURATION_SEC = 10.0
STEPS = 8
DEPTH = 8
SHIFT = 3.0
DENOISE = 1.0  # full text-to-music; no source latent

# Bank strengths to A/B. None = bank disabled. Other values are
# softmax-mass scalars for the banked tokens (1.0 = equal weighting
# with current K/V, 0.5 = banked tokens get half the weight current
# would, 0.0 = effectively disabled).
STRENGTHS_TO_RUN: List[float] = [1.5, 2.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_request(
    handler,
    cond_entry,
    T: int,
    device,
    dtype,
    seed: int,
) -> SlotRequest:
    """Build one SlotRequest with silence context and a fixed seed."""
    handler._ensure_silence_latent_on_device()
    ctx_lat = (
        handler.silence_latent[:, :T, :]
        .clone()
        .to(
            device=device,
            dtype=dtype,
        )
    )
    D = ctx_lat.shape[2]
    chunk_mask = torch.ones(1, T, D, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, chunk_mask], dim=-1)

    return SlotRequest(
        encoder_hidden_states=cond_entry.encoder_hidden_states,
        encoder_attention_mask=cond_entry.encoder_attention_mask,
        context_latents=context_latents,
        seed=seed,
        denoise=DENOISE,
    )


def _run_stream(
    engine: DiffusionEngine,
    config: DiffusionConfig,
    requests: List[SlotRequest],
    *,
    bank_strength: Optional[float],
    label: str,
):
    """Drive a fresh StreamPipeline through ``requests`` and collect outputs.

    Submits one request per tick (the actual streaming pattern) so the
    ring buffer ends up with slots at staggered step_idx values. Lockstep
    submission (all-up-front) defeats the bank: every slot is always at
    the same step as its peers, so reads happen on bank keys that no
    prior slot has written.

    ``bank_strength=None`` disables the bank. A float enables it at
    that softmax-mass scalar.
    """
    print(f"\n--- run: {label} (bank_strength={bank_strength}) ---")
    pipe = StreamPipeline(engine, config, pipeline_depth=DEPTH)
    bank = None
    if bank_strength is not None:
        bank = pipe.enable_feature_bank(strength=bank_strength)

    finished: List[torch.Tensor] = []
    tick_ms: List[float] = []
    max_ticks = len(requests) + STEPS + 4
    pending = list(requests)

    for i in range(max_ticks):
        if pending:
            pipe.submit(pending.pop(0))

        t0 = time.time()
        out = pipe.tick()
        dt = (time.time() - t0) * 1000
        tick_ms.append(dt)

        active_steps = sorted(s.step_idx for s in pipe._slots if s is not None)
        status = "FINISH" if out is not None else "tick"
        bank_n = bank.num_entries() if bank is not None else 0
        print(
            f"  tick {i:02d}: {dt:7.1f}ms  [{status}]  "
            f"active={pipe.active_slots} steps={active_steps}  "
            f"queue={len(pipe._queue)}  bank={bank_n}"
        )
        if out is not None:
            finished.append(out.detach().clone())
        if (
            not pending
            and pipe.active_slots == 0
            and not pipe._queue
            and len(finished) >= len(requests)
        ):
            break

    if bank_strength is not None:
        pipe.disable_feature_bank()

    return finished, tick_ms


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Feature Bank Smoke Test (eager path, no TRT, no torch.compile)")
    print("=" * 70)
    print(f"prompts ({len(PROMPTS)}): {PROMPTS}")
    print(f"seeds: {SEEDS}")
    print(f"duration={DURATION_SEC}s steps={STEPS} depth={DEPTH} shift={SHIFT}")

    # ------------------------------------------------------------------
    # Load session in eager mode
    # ------------------------------------------------------------------
    print("\n[1] Loading session (eager decoder, eager VAE)...")
    session = Session(
        decoder_backend="eager",
        vae_backend="eager",
        use_flash_attention=False,  # SDPA throughout; cleaner numerics
    )
    handler = session.handler
    device = handler.device
    dtype = handler.dtype
    attn_impl = getattr(handler.config, "_attn_implementation", "?")
    print(f"  device={device} dtype={dtype} attn={attn_impl}")
    print(f"  decoder type: {type(handler.model.decoder).__name__}")

    # ------------------------------------------------------------------
    # Encode conditioning(s)
    # ------------------------------------------------------------------
    print("\n[2] Encoding text conditioning per prompt...")
    cond_entries = []
    for prompt in PROMPTS:
        cond = session.encode_text(
            tags=prompt,
            lyrics="[instrumental]",
            duration=DURATION_SEC,
        )
        cond_entries.append(cond.to_entries()[0])

    # ------------------------------------------------------------------
    # Build SlotRequests (shared between runs so seeds match exactly)
    # ------------------------------------------------------------------
    T = int(DURATION_SEC * 25)  # acestep latent rate
    print(f"\n[3] Building {len(PROMPTS)} SlotRequests (T={T})...")
    requests: List[SlotRequest] = []
    for entry, seed in zip(cond_entries, SEEDS):
        requests.append(_build_request(handler, entry, T, device, dtype, seed))

    # ------------------------------------------------------------------
    # DiffusionEngine (no TRT). Reused across both runs so model state
    # is identical — we just rebuild the StreamPipeline per run.
    # ------------------------------------------------------------------
    print("\n[4] Building DiffusionEngine (no TRT, compile_loops=False)...")
    engine = DiffusionEngine(handler.model, compile_loops=False)
    config = DiffusionConfig(
        infer_steps=STEPS,
        shift=SHIFT,
        noise_on_cpu=True,
    )

    # ------------------------------------------------------------------
    # Run baseline (bank disabled) plus one run per requested strength
    # ------------------------------------------------------------------
    runs: dict[str, tuple[List[torch.Tensor], List[float]]] = {}

    out_off, ticks_off = _run_stream(
        engine,
        config,
        requests,
        bank_strength=None,
        label="bank OFF",
    )
    runs["off"] = (out_off, ticks_off)

    for s in STRENGTHS_TO_RUN:
        tag = f"s{s:.2f}".replace(".", "p")  # e.g. "s1p00", "s0p50"
        out, ticks = _run_stream(
            engine,
            config,
            requests,
            bank_strength=s,
            label=f"bank ON strength={s:g}",
        )
        runs[tag] = (out, ticks)

    # ------------------------------------------------------------------
    # Within-run pairwise distance: smaller under bank ON => identity
    # carryover. Strength sweep should show monotone trend if the bank
    # is doing what we think.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Within-run song-to-song distance (lower => more carryover):")
    print("=" * 70)

    def _pairwise(latents):
        out = []
        for i in range(len(latents) - 1):
            a = latents[i].float()
            b = latents[i + 1].float()
            cos = torch.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0),
                b.flatten().unsqueeze(0),
            ).item()
            rel_l2 = (a - b).norm().item() / a.norm().item()
            out.append((i + 1, i + 2, cos, rel_l2))
        return out

    for tag, (latents, _) in runs.items():
        print(f"  run [{tag}]:")
        for i, j, cos, rel_l2 in _pairwise(latents):
            print(
                f"    song {i} <-> song {j}:  cos = {cos:+.4f}   rel-L2 = {rel_l2:.4e}"
            )

    # ------------------------------------------------------------------
    # Save latents
    # ------------------------------------------------------------------
    print(f"\n[5] Saving latents to {OUT_DIR} ...")
    for tag, (latents, _) in runs.items():
        for i, lat in enumerate(latents):
            torch.save(lat.cpu(), OUT_DIR / f"song{i + 1}_{tag}.pt")

    # ------------------------------------------------------------------
    # VAE decode -> WAV for A/B listening
    # ------------------------------------------------------------------
    print(f"\n[6] VAE-decoding latents and writing WAVs to {OUT_DIR} ...")
    import soundfile as sf

    from acestep.nodes.types import Latent

    def _decode_save(lat_tensor: torch.Tensor, path: Path) -> None:
        audio = session.decode(Latent(tensor=lat_tensor))
        wav = audio.waveform.detach().cpu().float().squeeze(0)
        sf.write(str(path), wav.t().numpy(), audio.sample_rate)
        print(f"    wrote {path.name}  ({wav.shape[1] / audio.sample_rate:.2f}s)")

    for tag, (latents, _) in runs.items():
        for i, lat in enumerate(latents):
            _decode_save(lat, OUT_DIR / f"song{i + 1}_{tag}.wav")

    # ------------------------------------------------------------------
    # Tick time summary
    # ------------------------------------------------------------------
    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\nMean tick per run:")
    for tag, (_, ticks) in runs.items():
        print(f"  [{tag}]: {_avg(ticks):.1f}ms")

    print("\nDone.")


if __name__ == "__main__":
    main()
