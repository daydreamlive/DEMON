"""Spike test for StreamDiffusion-style batched denoising pipeline.

Proves the concept: batch multiple generations at different denoising
stages into a single forward pass. After warmup, every tick produces
a finished generation.

Compares output quality against sequential (standard) generation to
verify the batched approach doesn't corrupt results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
import numpy as np

from acestep.engine.session import Session
from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.engine.conditions import PreparedCondition, ConditionSet


def main():
    print("=" * 60)
    print("StreamPipeline Spike Test")
    print("=" * 60)

    # Load model (no TRT, no compile for clean spike)
    print("\n[1] Loading model...")
    session = Session(
        project_root="checkpoints",
        use_flash_attention=True,
    )
    handler = session.handler
    device = handler.device
    dtype = handler.dtype

    # Encode a test prompt
    print("[2] Encoding conditioning...")
    cond = session.encode_text(
        tags="electronic ambient, 120 bpm, dreamy pads",
        lyrics="[instrumental]",
        duration=30.0,
    )
    entry = cond.to_entries()[0]

    # Build context latents (silence + chunk mask)
    handler._ensure_silence_latent_on_device()
    T = int(30.0 * 25)  # 30 seconds
    ctx_lat = handler.silence_latent[:, :T, :].clone().to(device=device, dtype=dtype)
    D = ctx_lat.shape[2]
    cm = torch.ones(1, T, D, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, cm], dim=-1)

    config = DiffusionConfig(
        infer_steps=8,
        shift=3.0,
        seed=42,
        noise_on_cpu=True,
    )

    # -------------------------------------------------------
    # Test 1: Sequential baseline (manual loop, no torch.compile)
    # -------------------------------------------------------
    print("\n[3] Running sequential baseline (8 steps, seed=42)...")
    engine = DiffusionEngine(handler.model)
    decoder = engine.decoder

    prepared = PreparedCondition(
        encoder_hidden_states=entry.encoder_hidden_states,
        encoder_attention_mask=entry.encoder_attention_mask,
        context_latents=context_latents,
    )

    # Build schedule and noise (same as StreamPipeline will do)
    t_schedule = engine._build_timestep_schedule(config, device, dtype)
    torch.manual_seed(42)
    D = context_latents.shape[-1] // 2
    noise_bdt = torch.randn(1, D, T, device="cpu", dtype=torch.float32)
    xt = noise_bdt.movedim(-1, -2).to(device=device, dtype=dtype)
    attn_mask = torch.ones(1, T, device=device, dtype=dtype)

    t0 = time.time()
    with torch.no_grad():
        for i in range(config.infer_steps):
            t_curr = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_next - t_curr
            t_vec = t_curr.unsqueeze(0)  # [1]

            vt = decoder(
                hidden_states=xt,
                timestep=t_vec,
                timestep_r=t_vec,
                attention_mask=attn_mask,
                encoder_hidden_states=prepared.encoder_hidden_states,
                encoder_attention_mask=prepared.encoder_attention_mask,
                context_latents=prepared.context_latents,
                use_cache=False,
                past_key_values=None,
            )[0]
            xt = xt + dt * vt
    baseline_ms = (time.time() - t0) * 1000
    baseline_latent = xt
    print(f"  Baseline: {baseline_ms:.1f}ms, shape={list(baseline_latent.shape)}")

    # -------------------------------------------------------
    # Test 2: Stream pipeline - single request (should match baseline)
    # -------------------------------------------------------
    print("\n[4] Running stream pipeline (single request, same seed)...")
    pipe = StreamPipeline(engine, config)

    request = SlotRequest(
        encoder_hidden_states=entry.encoder_hidden_states,
        encoder_attention_mask=entry.encoder_attention_mask,
        context_latents=context_latents,
        seed=42,
    )
    pipe.submit(request)

    # Tick through all steps + 1 to collect result
    stream_result = None
    tick_times = []
    for i in range(config.infer_steps + 1):
        t0 = time.time()
        result = pipe.tick()
        tick_ms = (time.time() - t0) * 1000
        tick_times.append(tick_ms)
        status = "DONE" if result is not None else f"step {i}"
        print(f"  tick {i}: {tick_ms:.1f}ms  [{status}]  "
              f"active={pipe.active_slots}")
        if result is not None:
            stream_result = result

    if stream_result is not None:
        diff = (baseline_latent - stream_result).abs().max().item()
        print(f"\n  Max diff vs baseline: {diff:.6f}")
        if diff < 1e-3:
            print("  PASS: Stream output matches baseline")
        else:
            print(f"  WARNING: Outputs differ (max={diff})")
    else:
        print("  FAIL: No result produced")

    # -------------------------------------------------------
    # Test 3: Stream pipeline - continuous feed (throughput test)
    # -------------------------------------------------------
    print("\n[5] Throughput test: feeding 16 requests continuously...")
    pipe2 = StreamPipeline(engine, config)

    # Submit 16 requests with different seeds
    for i in range(16):
        pipe2.submit(SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=1000 + i,
        ))

    completed = []
    total_start = time.time()
    max_ticks = 16 + config.infer_steps + 2  # generous bound

    for tick_num in range(max_ticks):
        t0 = time.time()
        result = pipe2.tick()
        tick_ms = (time.time() - t0) * 1000

        if result is not None:
            completed.append((tick_num, tick_ms))
            print(f"  tick {tick_num:3d}: {tick_ms:6.1f}ms  "
                  f"-> COMPLETED #{len(completed)}  "
                  f"active={pipe2.active_slots} queue={len(pipe2._queue)}")
        else:
            if tick_num < 12 or tick_num % 5 == 0:
                print(f"  tick {tick_num:3d}: {tick_ms:6.1f}ms  "
                      f"active={pipe2.active_slots} queue={len(pipe2._queue)}")

        if pipe2.active_slots == 0 and not pipe2._queue:
            break

    total_ms = (time.time() - total_start) * 1000
    print(f"\n  Total: {total_ms:.0f}ms for {len(completed)} generations")
    if len(completed) > 1:
        # Steady-state throughput: time between completions after warmup
        warmup_completions = completed[0:1]
        steady = completed[1:]
        if steady:
            avg_ms = total_ms / len(completed)
            print(f"  Avg per generation: {avg_ms:.1f}ms")
            print(f"  Throughput: {1000/avg_ms:.1f} gen/sec")

    # -------------------------------------------------------
    # Test 4: Interactive simulation (parameter changes)
    # -------------------------------------------------------
    print("\n[6] Interactive simulation: user changes params mid-stream...")

    # Encode a second prompt (different tags)
    cond2 = session.encode_text(
        tags="heavy metal, distorted guitars, 160 bpm, aggressive",
        lyrics="[instrumental]",
        duration=30.0,
    )
    entry2 = cond2.to_entries()[0]

    pipe3 = StreamPipeline(engine, config)

    # Feed initial prompt for warmup
    for i in range(8):
        pipe3.submit(SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=2000 + i,
        ))

    results_before = []
    results_after = []
    switched_at = None

    for tick_num in range(24):
        # At tick 12, "user" switches to new prompt
        if tick_num == 12 and switched_at is None:
            switched_at = tick_num
            print(f"\n  --- USER CHANGES PROMPT AT TICK {tick_num} ---\n")

        # Always feed new requests to keep pipeline full
        if tick_num >= 8:
            if switched_at is not None and tick_num >= switched_at:
                # New prompt
                pipe3.submit(SlotRequest(
                    encoder_hidden_states=entry2.encoder_hidden_states,
                    encoder_attention_mask=entry2.encoder_attention_mask,
                    context_latents=context_latents,
                    seed=3000 + tick_num,
                ))
            else:
                pipe3.submit(SlotRequest(
                    encoder_hidden_states=entry.encoder_hidden_states,
                    encoder_attention_mask=entry.encoder_attention_mask,
                    context_latents=context_latents,
                    seed=2000 + tick_num,
                ))

        result = pipe3.tick()
        if result is not None:
            label = "NEW" if switched_at and tick_num >= switched_at + 8 else "OLD"
            print(f"  tick {tick_num:3d}: completed ({label} prompt)  "
                  f"active={pipe3.active_slots}")
            if label == "OLD":
                results_before.append(result)
            else:
                results_after.append(result)
        else:
            if tick_num < 10 or tick_num == switched_at:
                print(f"  tick {tick_num:3d}: processing  "
                      f"active={pipe3.active_slots}")

    print(f"\n  Completions with old prompt: {len(results_before)}")
    print(f"  Completions with new prompt: {len(results_after)}")
    if results_before and results_after:
        diff = (results_before[-1] - results_after[0]).abs().mean().item()
        print(f"  Mean diff between last old / first new: {diff:.4f}")
        print("  (Should be significantly different = prompt change worked)")

    print("\n" + "=" * 60)
    print("Spike complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
