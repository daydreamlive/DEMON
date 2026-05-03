"""Test shared mutable SDE curves that bypass ring-buffer drain.

Stagger submissions (one per tick) so slots are at different denoising
stages. After switching, the slot at step 0 has ALL remaining steps
with the new curve and should show maximum effect. The slot at step 7
has only 1 remaining step and minimal effect.

In per-slot mode: ALL old slots use CURVE_A regardless of step.
In shared mode: old slots use CURVE_B, with effect proportional to
remaining steps. The earliest-stage slot should approach pure CURVE_B.
"""
if __name__ != "__main__":
    import sys; sys.exit(0)

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import numpy as np

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionConfig, DiffusionEngine
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.fixtures import audio_fixture

PROJECT_ROOT = Path(__file__).parent.parent.parent
SOURCE_AUDIO = audio_fixture("inside_confusion_loop_60s_gsm.wav")

SAMPLE_RATE = 48000
SEED = 1528
T = 1500
DEPTH = 8

TRT_ENGINE = PROJECT_ROOT / "trt_engines" / "decoder_mixed_refit_b8_240s" / "decoder_mixed_refit_b8_240s.engine"
VAE_ENCODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_encode_fp16_240s" / "vae_encode_fp16_240s.engine"
VAE_DECODE_ENGINE = PROJECT_ROOT / "trt_engines" / "vae_decode_fp16_240s" / "vae_decode_fp16_240s.engine"

CURVE_A = torch.full((1, T, 1), 0.1)   # heavy source preservation
CURVE_B = torch.full((1, T, 1), 0.95)  # near-full generation


def load_audio(path, duration=60.0):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(duration * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    from acestep.nodes.types import Audio
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


def cosine_sim(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
print("=" * 60)
print("Shared Mutable SDE Curve - Ring Buffer Bypass Test")
print("=" * 60)

print("\n[Setup] Loading model...")
session = Session(
    project_root=str(PROJECT_ROOT / "checkpoints"),
    decoder_backend="tensorrt",
    vae_backend="tensorrt",
    trt_engines={
        "decoder": str(TRT_ENGINE),
        "vae_encode": str(VAE_ENCODE_ENGINE),
        "vae_decode": str(VAE_DECODE_ENGINE),
    },
)

handler = session.handler
device = handler.device
dtype = handler.dtype

audio = load_audio(SOURCE_AUDIO)
latent = session.encode_audio(audio)
context_latent = session.extract_hints(latent)
source = PreparedSource(latent=latent, context_latent=context_latent)

cond = session.encode_text(
    tags="deathstep, heavy bass, dark atmosphere",
    instruction=TASK_INSTRUCTIONS["cover"],
    refer_latent=source.latent,
    bpm=136, duration=60.0, key="G# minor",
)
entry = cond.to_entries()[0]

ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
D_ctx = ctx_lat.shape[2]
cm = torch.ones(1, T, D_ctx, device=device, dtype=dtype)
context_latents = torch.cat([ctx_lat, cm], dim=-1)
source_latents = source.latent.tensor.to(device=device, dtype=dtype)

engine = handler._diffusion_engine
config = DiffusionConfig(infer_steps=8, shift=3.0, noise_on_cpu=True)


def make_request(curve):
    return SlotRequest(
        encoder_hidden_states=entry.encoder_hidden_states,
        encoder_attention_mask=entry.encoder_attention_mask,
        context_latents=context_latents,
        seed=SEED,
        source_latents=source_latents,
        denoise=0.75,
        sde_denoise_curve=curve,
    )


def run_test(mode: str):
    """Fill pipeline with staggered CURVE_A slots, switch, measure.

    Staggered: submit one request per tick so slots end up at
    steps [7, 6, 5, 4, 3, 2, 1, 0] when the pipeline is full.
    """
    pipe = StreamPipeline(engine, config, pipeline_depth=DEPTH)

    # Phase 1: Stagger CURVE_A submissions (one per tick).
    # First DEPTH ticks fill the pipeline with slots at different stages.
    for i in range(DEPTH):
        pipe.submit(make_request(CURVE_A))
        r = pipe.tick()
        # No completions yet during warmup (first slot needs DEPTH ticks)
        if r is not None:
            print(f"    [warmup] unexpected completion at tick {i}")

    # Now pipeline is full. Slot stages: [7, 6, 5, 4, 3, 2, 1, 0]
    # (first submitted has had 7 steps, last submitted has had 0)
    print(f"  Pipeline full: {pipe.active_slots} slots, staggered steps 7..0")
    for i, slot in enumerate(pipe._slots):
        if slot is not None:
            print(f"    slot {i}: step_idx={slot.step_idx}")

    # Phase 2: Switch to CURVE_B
    if mode == 'shared':
        pipe.set_shared_sde_curve(CURVE_B.to(device=device, dtype=dtype))
        print("  >>> Set shared CURVE_B (all in-flight slots affected)")

    # Phase 3: Continue ticking with new submissions.
    # In per-slot: new submissions use CURVE_B.
    # In shared: new submissions use CURVE_A but shared overrides to CURVE_B.
    new_curve = CURVE_B if mode == 'per_slot' else CURVE_A
    results = []
    for i in range(DEPTH + 8):
        pipe.submit(make_request(new_curve))
        r = pipe.tick()
        if r is not None:
            sim = cosine_sim(r, source_latents)
            results.append(sim)
            tag = "OLD" if sim > 0.8 else "NEW"
            print(f"  completion {len(results):2d}: source_sim={sim:.4f}  [{tag}]")

    return results


# ------------------------------------------------------------------
# Run both modes
# ------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"A) Per-slot: old slots keep CURVE_A until they drain")
print("=" * 60)
results_a = run_test('per_slot')

print(f"\n{'=' * 60}")
print(f"B) Shared: old slots use CURVE_B on next tick")
print("=" * 60)
results_b = run_test('shared')


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("COMPARISON (first 12 completions after switch)")
print("=" * 60)

print(f"\n  {'#':<4} {'Per-slot':<12} {'Shared':<12} {'Delta':<10} {'Notes'}")
print(f"  {'-'*55}")
for i in range(min(12, len(results_a), len(results_b))):
    delta = results_a[i] - results_b[i]
    note = ""
    if i < DEPTH:
        note = f"old slot (had {DEPTH - 1 - i}/8 steps left)"
    else:
        note = "new slot (fully CURVE_B)"
    print(f"  {i:<4} {results_a[i]:<12.4f} {results_b[i]:<12.4f} {delta:<+10.4f} {note}")

# Quantify the bypass effect
old_slots_a = results_a[:DEPTH]
old_slots_b = results_b[:DEPTH]
new_slots_a = results_a[DEPTH:DEPTH+4] if len(results_a) > DEPTH else []
new_slots_b = results_b[DEPTH:DEPTH+4] if len(results_b) > DEPTH else []

avg_old_a = np.mean(old_slots_a) if old_slots_a else 0
avg_old_b = np.mean(old_slots_b) if old_slots_b else 0
avg_new_a = np.mean(new_slots_a) if new_slots_a else 0
avg_new_b = np.mean(new_slots_b) if new_slots_b else 0

print(f"\n  Old slots (in-flight during switch):")
print(f"    Per-slot avg: {avg_old_a:.4f}  (should be ~1.0, still CURVE_A)")
print(f"    Shared avg:   {avg_old_b:.4f}  (should trend toward ~0.35)")
print(f"    Delta:        {avg_old_a - avg_old_b:+.4f}")

print(f"\n  New slots (submitted after switch):")
print(f"    Per-slot avg: {avg_new_a:.4f}  (should be ~0.35, CURVE_B)")
print(f"    Shared avg:   {avg_new_b:.4f}  (should be ~0.35, CURVE_B)")

if avg_old_a - avg_old_b > 0.1:
    print(f"\n  PASS: Shared curve affected in-flight slots.")
    print(f"  Old slots shifted {avg_old_a - avg_old_b:.4f} toward CURVE_B behavior.")
else:
    print(f"\n  FAIL: Shared curve had negligible effect on old slots.")
