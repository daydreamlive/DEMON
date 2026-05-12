"""RCFG smoke test — Residual CFG paths through the streaming pipeline.

Inspired by StreamDiffusion's RCFG (Kodaira et al.). Two RCFG variants:

  - rcfg_mode="initialize" : run the uncond pass once per slot at step 0,
                              cache the velocity, reuse it for every
                              remaining step. One extra forward per slot.
  - rcfg_mode="self"       : skip the uncond forward entirely; approximate
                              ``v_uncond`` with the slot's initial noise
                              tensor. Flow-matching identity ``v = noise - x0``
                              with ``x0_uncond ~ 0`` gives ``v_uncond ~ noise``.
                              Zero extra forwards.

The point: turbo is CFG-distilled (skips guidance by default). RCFG lets
us put guidance back at inference time, with cost much closer to no-CFG
than to standard CFG.

This test runs four configurations against identical noise + conditioning:

  baseline_no_cfg : turbo's normal path, no guidance applied
  full_cfg        : standard two-pass CFG, w=guidance_scale every step
  rcfg_initialize : RCFG-Onetime
  rcfg_self       : RCFG-Self-Negative

Reports:
  - Per-tick latency (RCFG should be near baseline; full_cfg ~2x)
  - Latent / mel-spec / chroma / onset similarity to baseline and to full_cfg
    (so we can tell whether RCFG approximates full CFG)

Usage:
    uv run python demos/test_rcfg.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

import soundfile as sf
import librosa

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session, PreparedSource
from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import StreamPipeline, SlotRequest, SlotCondition
from acestep.nodes.types import Audio, Latent
from acestep.paths import project_root, checkpoints_dir, select_trt_engines
from acestep.fixtures import audio_fixture


PROJECT_ROOT = project_root()
OUT_DIR = PROJECT_ROOT / "_output" / "rcfg"
SAMPLE_RATE = 48000

T_FRAMES = 1500
N_GENS = 3
INFER_STEPS = 8
SHIFT = 3.0
SEED_BASE = 1528
GUIDANCE_SCALE = 7.0   # CFG strength. Turbo treats >1.0 as no-op without
                       # something running uncond+APG (which RCFG provides).


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    n = (af.norm() * bf.norm()).clamp_min(1e-12)
    return float((af @ bf) / n)


def mel_full(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    mel = librosa.feature.melspectrogram(
        y=wav.astype(np.float32), sr=sr, n_mels=64,
        hop_length=512, n_fft=2048,
    )
    return torch.from_numpy(librosa.power_to_db(mel + 1e-10).flatten())


def chroma_full(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    chroma = librosa.feature.chroma_stft(
        y=wav.astype(np.float32), sr=sr, hop_length=512, n_fft=2048,
    )
    return torch.from_numpy(chroma.flatten())


def onset(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    o = librosa.onset.onset_strength(y=wav.astype(np.float32), sr=sr, hop_length=512)
    return torch.from_numpy(o)


def build_neg_conditions(session: Session, T_seconds: float) -> list[SlotCondition]:
    """Empty-prompt negative conditioning, the standard CFG uncond."""
    neg_cond = session.encode_text(
        tags="", instruction=TASK_INSTRUCTIONS["text2music"],
        refer_latent=None, bpm=120, duration=T_seconds, key="C major",
    )
    entries = neg_cond.to_entries()
    return [
        SlotCondition(
            encoder_hidden_states=e.encoder_hidden_states,
            encoder_attention_mask=e.encoder_attention_mask,
        )
        for e in entries
    ]


def run_config(
    label: str,
    *,
    engine,
    config: DiffusionConfig,
    entry,
    context_latents: torch.Tensor,
    neg_conditions: list[SlotCondition],
    guidance_curve: torch.Tensor | None,
    rcfg_mode: str | None,
    session: Session,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Build a fresh pipeline, run N_GENS slots, return latents + audio + timing.

    ``_make_noise`` is patched to a deterministic per-call source so all
    configurations see byte-identical noise tensors and any output
    difference is purely from the RCFG branch logic."""
    print(f"\n[{label}]  rcfg_mode={rcfg_mode}  guidance={guidance_curve is not None}")
    pipe = StreamPipeline(engine, config)

    rng = torch.Generator(device=device).manual_seed(SEED_BASE)
    call_count = {"n": 0}

    def make_noise(request: SlotRequest) -> torch.Tensor:
        # Deterministic per-call, shared across configs.
        T = request.context_latents.shape[1]
        D = request.context_latents.shape[-1] // 2
        gen = torch.Generator(device=device).manual_seed(SEED_BASE + call_count["n"])
        call_count["n"] += 1
        return torch.randn(1, T, D, device=device, dtype=dtype, generator=gen)

    pipe._make_noise = make_noise  # type: ignore[method-assign]

    # Submit N_GENS requests with the requested guidance / rcfg config.
    for _ in range(N_GENS):
        req = SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=None,
            denoise=1.0,
            neg_conditions=neg_conditions if rcfg_mode != "self" else [],
            guidance_curve=guidance_curve,
            rcfg_mode=rcfg_mode,
        )
        pipe.submit(req)

    finished: list[torch.Tensor] = []
    tick_ms_per: list[float] = []
    max_ticks = N_GENS + pipe.depth + 5
    for _ in range(max_ticks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipe.tick()
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - t0) * 1000
        tick_ms_per.append(tick_ms)
        if result is not None:
            finished.append(result.detach().cpu().clone())
            print(f"  gen {len(finished)}/{N_GENS}  tick={tick_ms:.2f}ms")
        if len(finished) >= N_GENS:
            break

    audios, mels, chromas, onsets = [], [], [], []
    out_dir = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, lat in enumerate(finished):
        lat_dev = lat.to(device=device, dtype=dtype)
        audio_out = session.decode(Latent(tensor=lat_dev))
        wav = audio_out.waveform.detach().cpu().float().squeeze(0).numpy()
        audios.append(wav)
        mels.append(mel_full(wav))
        chromas.append(chroma_full(wav))
        onsets.append(onset(wav))
        sf.write(str(out_dir / f"gen{i+1}.wav"), wav.T, SAMPLE_RATE)

    # Per-tick dump (helps catch zero-time / async-mismeasurement issues).
    print(f"  all ticks (ms): " + ", ".join(f"{t:.1f}" for t in tick_ms_per))

    # Steady-state tick time. The schedule runs N_GENS submissions through
    # an 8-step ring; submissions enter on ticks 0..N_GENS-1, then drain.
    # Drop tick 0 (TRT-context first-call overhead) and the trailing
    # zero-work drain ticks, take the median of what remains.
    nonzero = [t for t in tick_ms_per[1:] if t > 1.0]
    steady = nonzero
    return {
        "label": label,
        "latents": finished,
        "audios": audios,
        "mels": mels,
        "chromas": chromas,
        "onsets": onsets,
        "tick_ms_all": tick_ms_per,
        "tick_ms_steady_median": float(np.median(steady)) if steady else float("nan"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"RCFG smoke test  T={T_FRAMES} frames ({T_FRAMES/25:.0f}s)  "
          f"N_GENS={N_GENS}  guidance={GUIDANCE_SCALE}")
    print("=" * 78)

    # --- Load model ---
    print("\n[Setup] loading model (TRT 60s)...")
    t0 = time.time()
    trt = select_trt_engines(duration_s=T_FRAMES / 25.0)
    session = Session(
        project_root=str(checkpoints_dir()),
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=trt,
    )
    handler = session.handler
    device, dtype = handler.device, handler.dtype
    engine = handler._diffusion_engine
    print(f"  loaded in {time.time()-t0:.1f}s")

    # --- Source audio for semantic context ---
    print("[Setup] source audio + text encode...")
    audio_path = audio_fixture("inside_confusion_loop_60s_gsm.wav")
    data, sr = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, : int(60.0 * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, : waveform.shape[-1] - rem]
    audio_in = Audio(waveform=waveform, sample_rate=SAMPLE_RATE)
    latent = session.encode_audio(audio_in)
    context_latent = session.extract_hints(latent)
    source = PreparedSource(latent=latent, context_latent=context_latent)

    pos_cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["text2music"],
        refer_latent=source.latent,
        bpm=136, duration=T_FRAMES / 25.0, key="G# minor",
    )
    entry = pos_cond.to_entries()[0]
    neg_conditions = build_neg_conditions(session, T_FRAMES / 25.0)

    ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
    T_actual = ctx_lat.shape[1]
    D = ctx_lat.shape[2]
    cm = torch.ones(1, T_actual, D, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, cm], dim=-1)
    print(f"  context T={T_actual}, D={D}")

    config = DiffusionConfig(infer_steps=INFER_STEPS, shift=SHIFT, noise_on_cpu=True)
    guidance_curve = torch.full((1, T_actual, 1), GUIDANCE_SCALE, device=device, dtype=dtype)

    # --- Run the four configurations ---
    configs = [
        ("baseline_no_cfg", None,                None),  # turbo default
        ("full_cfg",        guidance_curve,      None),  # standard CFG
        ("rcfg_initialize", guidance_curve,      "initialize"),
        ("rcfg_self",       guidance_curve,      "self"),
    ]
    results = []
    for label, gc, mode in configs:
        results.append(run_config(
            label,
            engine=engine, config=config, entry=entry,
            context_latents=context_latents,
            neg_conditions=neg_conditions,
            guidance_curve=gc,
            rcfg_mode=mode,
            session=session, device=device, dtype=dtype,
        ))

    # --- Timing report ---
    print("\n" + "=" * 78)
    print("STEADY-STATE TICK LATENCY (median over the post-warmup window)")
    print("=" * 78)
    base = next(r for r in results if r["label"] == "baseline_no_cfg")
    base_ms = base["tick_ms_steady_median"]
    print(f"  {'config':>18s}  {'tick (ms)':>10s}  {'vs no-CFG':>10s}")
    for r in results:
        m = r["tick_ms_steady_median"]
        ratio = m / base_ms if base_ms > 0 else float("nan")
        print(f"  {r['label']:>18s}  {m:>10.2f}  {ratio:>9.2f}x")

    # --- Output similarity ---
    print("\n" + "=" * 78)
    print("PAIRWISE OUTPUT SIMILARITY  (averaged over gens, vs baseline_no_cfg)")
    print("  Higher = more similar to baseline. Lower means guidance changed the output.")
    print("=" * 78)
    base_lat = base["latents"]
    base_mel = base["mels"]
    base_chr = base["chromas"]
    base_ons = base["onsets"]
    print(f"  {'config':>18s}  {'latent':>10s}  {'mel-full':>10s}  "
          f"{'chroma':>10s}  {'onset':>10s}")
    for r in results:
        latc = float(np.mean([cos_sim(r["latents"][i], base_lat[i]) for i in range(N_GENS)]))
        melc = float(np.mean([cos_sim(r["mels"][i], base_mel[i]) for i in range(N_GENS)]))
        chrc = float(np.mean([cos_sim(r["chromas"][i], base_chr[i]) for i in range(N_GENS)]))
        onsc = float(np.mean([cos_sim(r["onsets"][i], base_ons[i]) for i in range(N_GENS)]))
        print(f"  {r['label']:>18s}  {latc:>10.3f}  {melc:>10.3f}  "
              f"{chrc:>10.3f}  {onsc:>10.3f}")

    print("\n" + "=" * 78)
    print("AGREEMENT WITH full_cfg  (does each RCFG approximate standard CFG?)")
    print("  Higher = RCFG output more closely matches full CFG output.")
    print("=" * 78)
    full = next(r for r in results if r["label"] == "full_cfg")
    print(f"  {'config':>18s}  {'latent':>10s}  {'mel-full':>10s}  "
          f"{'chroma':>10s}  {'onset':>10s}")
    for r in results:
        if r["label"] == "full_cfg":
            continue
        latc = float(np.mean([cos_sim(r["latents"][i], full["latents"][i]) for i in range(N_GENS)]))
        melc = float(np.mean([cos_sim(r["mels"][i], full["mels"][i]) for i in range(N_GENS)]))
        chrc = float(np.mean([cos_sim(r["chromas"][i], full["chromas"][i]) for i in range(N_GENS)]))
        onsc = float(np.mean([cos_sim(r["onsets"][i], full["onsets"][i]) for i in range(N_GENS)]))
        print(f"  {r['label']:>18s}  {latc:>10.3f}  {melc:>10.3f}  "
              f"{chrc:>10.3f}  {onsc:>10.3f}")

    print(f"\nDone. Audio at {OUT_DIR}")


if __name__ == "__main__":
    main()
