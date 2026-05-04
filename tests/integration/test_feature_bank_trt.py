"""GPU smoke test: bank-aware TRT engine end-to-end.

Loads the bank-aware TRT engine produced by
``scripts/build_bank_engine.py`` and drives the pipeline through the
same A/B as the eager smoke test (``test_feature_bank_stream.py``).
The point is to confirm the engine binds, executes, and produces
sane output -- timing comparison vs eager is the bonus.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.session import Session
from acestep.engine.stream import SlotRequest, StreamPipeline


OUT_DIR = Path(__file__).resolve().parent / "feature_bank_trt_out"
ENGINE_PATH = (
    Path(__file__).resolve().parents[2] /
    "trt_engines" / "bank_decoder_b8" / "bank_decoder_b8.engine"
)

PROMPTS: List[str] = [
    "ambient drone, slow swell, dark cinematic",
    "ambient drone, slow swell, dark cinematic",
    "ambient drone, slow swell, dark cinematic",
]
SEEDS: List[int] = [1234, 5678, 9012]

# The engine was built with seq_len=250 (10s at 25Hz latent, halved to
# T_lat=125 inside via patch_size=2). Match that.
DURATION_SEC = 10.0
STEPS = 8
DEPTH = 8
SHIFT = 3.0
DENOISE = 1.0
STRENGTH = 0.0  # diagnostic: 0 = bank fully masked, should match no-bank baseline


def _build_request(handler, cond_entry, T, device, dtype, seed):
    handler._ensure_silence_latent_on_device()
    ctx_lat = handler.silence_latent[:, :T, :].clone().to(device=device, dtype=dtype)
    D = ctx_lat.shape[2]
    chunk_mask = torch.ones(1, T, D, device=device, dtype=dtype)
    return SlotRequest(
        encoder_hidden_states=cond_entry.encoder_hidden_states,
        encoder_attention_mask=cond_entry.encoder_attention_mask,
        context_latents=torch.cat([ctx_lat, chunk_mask], dim=-1),
        seed=seed, denoise=DENOISE,
    )


def _run_stream(engine, config, requests, *, label):
    print(f"\n--- {label} ---")
    pipe = StreamPipeline(engine, config, pipeline_depth=DEPTH)
    bank = pipe.enable_feature_bank_trt(
        engine_path=ENGINE_PATH, num_steps=STEPS, strength=STRENGTH,
    )

    finished, tick_ms = [], []
    pending = list(requests)
    max_ticks = len(requests) + STEPS + 4
    for i in range(max_ticks):
        if pending:
            pipe.submit(pending.pop(0))
        t0 = time.time()
        out = pipe.tick()
        dt = (time.time() - t0) * 1000
        tick_ms.append(dt)
        active_steps = sorted(s.step_idx for s in pipe._slots if s is not None)
        bank_n = bank.num_entries()
        status = "FINISH" if out is not None else "tick"
        print(
            f"  tick {i:02d}: {dt:7.1f}ms  [{status}]  "
            f"active={pipe.active_slots} steps={active_steps}  bank={bank_n}"
        )
        if out is not None:
            finished.append(out.detach().clone())
        if (
            not pending
            and pipe.active_slots == 0
            and len(finished) >= len(requests)
        ):
            break
    return finished, tick_ms


def main():
    if not ENGINE_PATH.exists():
        raise SystemExit(f"Bank engine not found at {ENGINE_PATH}")

    OUT_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("Bank TRT Smoke Test")
    print("=" * 70)
    print(f"engine: {ENGINE_PATH}")

    print("\n[1] Loading session (eager VAE / text encoder; bank engine swap below)...")
    session = Session(
        decoder_backend="eager",  # PT decoder loaded so we have weights for fallback
        vae_backend="eager",
        use_flash_attention=False,
    )
    handler = session.handler
    device = handler.device
    dtype = handler.dtype
    print(f"  device={device} dtype={dtype}")

    print(f"\n[2] Encoding {len(PROMPTS)} prompts...")
    cond_entries = []
    for prompt in PROMPTS:
        cond = session.encode_text(
            tags=prompt, lyrics="[instrumental]", duration=DURATION_SEC,
        )
        cond_entries.append(cond.to_entries()[0])

    T = int(DURATION_SEC * 25)  # 250
    requests = [
        _build_request(handler, ent, T, device, dtype, seed)
        for ent, seed in zip(cond_entries, SEEDS)
    ]

    print("\n[3] Building DiffusionEngine (PT, no compile_loops)...")
    engine = DiffusionEngine(handler.model, compile_loops=False)
    config = DiffusionConfig(
        infer_steps=STEPS, shift=SHIFT, noise_on_cpu=True,
    )

    out_trt, ticks_trt = _run_stream(
        engine, config, requests, label="bank TRT, strength=1.0",
    )

    print("\n[4] Decoding to WAVs...")
    import soundfile as sf
    from acestep.nodes.types import Latent
    for i, lat in enumerate(out_trt):
        audio = session.decode(Latent(tensor=lat))
        wav = audio.waveform.detach().cpu().float().squeeze(0).t().numpy()
        path = OUT_DIR / f"song{i+1}_trt.wav"
        sf.write(str(path), wav, audio.sample_rate)
        print(f"    wrote {path.name}")

    print(
        f"\nMean tick: {sum(ticks_trt)/len(ticks_trt):.1f}ms "
        f"(n_finished={len(out_trt)})"
    )

    # Cross-song similarity (sanity: should be > eager's no-bank baseline).
    print("\nWithin-run pairwise cos:")
    for i in range(len(out_trt) - 1):
        a = out_trt[i].float().flatten().unsqueeze(0)
        b = out_trt[i + 1].float().flatten().unsqueeze(0)
        cos = torch.nn.functional.cosine_similarity(a, b).item()
        print(f"  song {i+1} <-> {i+2}: cos={cos:+.4f}")


if __name__ == "__main__":
    main()
