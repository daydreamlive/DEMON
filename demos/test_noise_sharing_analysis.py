"""Empirical analysis of noise-sharing strategies through the real model.

Probes how noise-space correlation actually translates to latent-space
and audio-space correlation under text conditioning. Tests:

  - fresh        : independent randn each gen (true 'no sharing' baseline)
  - fixed_seed   : same noise every gen (max correlation ceiling)
  - ema_a0.70    : current scheme (acestep/engine/stream.py:391-396)
  - ema_a0.99    : current scheme with the strongest still-non-degenerate
                   alpha — should be the most musical IF the mechanism works
  - anchor_a0.70 : proposed fix — blend with frozen first-gen noise
                   (stable alpha, no exponential decay)

For each strategy, runs N_GENS pure-t2m generations with identical text
conditioning and reports three similarity matrices:

  noise-cos    : cos(input_noise_i, input_noise_j)        [validates math]
  latent-cos   : cos(output_latent_i, output_latent_j)    [DiT carry-through]
  mel-cos      : cos(mel_envelope_i, mel_envelope_j)      [audio similarity,
                                                           time-averaged]
  chroma-cos   : cos(chroma_mean_i, chroma_mean_j)        [harmonic content]

The point: if noise-cos > 0.7 but mel-cos ≈ chroma-cos ≈ what 'fresh' produces,
then noise-space correlation does not translate to musical correlation.

Usage:
    uv run python demos/test_noise_sharing_analysis.py
"""

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
from acestep.engine.stream import StreamPipeline, SlotRequest
from acestep.nodes.types import Audio, Latent
from acestep.paths import project_root, checkpoints_dir, select_trt_engines
from acestep.fixtures import audio_fixture


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = project_root()
OUT_DIR = PROJECT_ROOT / "_output" / "noise_sharing_analysis"
SAMPLE_RATE = 48000

T_FRAMES = 1500              # 60s @ 25fps — matches realtime_motion deployment
N_GENS = 5                   # generations per strategy
INFER_STEPS = 8
SHIFT = 3.0
SEED_BASE = 1528
SAVE_AUDIO = True            # also write WAVs so we can listen


# ---------------------------------------------------------------------------
# Noise strategies
# ---------------------------------------------------------------------------
@dataclass
class NoiseStrategy:
    name: str
    fn: Callable[[int, tuple, torch.device, torch.dtype, torch.Generator], torch.Tensor]


def make_fresh() -> NoiseStrategy:
    """Independent randn each gen — true 'noise sharing OFF, fresh seed' baseline."""
    def fn(idx, shape, device, dtype, gen):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)
    return NoiseStrategy("fresh", fn)


def make_fixed_seed(seed: int) -> NoiseStrategy:
    """Same noise every gen — ceiling for what noise-space sharing can achieve."""
    cache = {}

    def fn(idx, shape, device, dtype, gen):
        key = (tuple(shape), device, dtype)
        if key not in cache:
            g = torch.Generator(device=device).manual_seed(seed)
            cache[key] = torch.randn(*shape, device=device, dtype=dtype, generator=g)
        return cache[key].clone()

    return NoiseStrategy(f"fixed_seed_{seed}", fn)


def make_ema(alpha: float) -> NoiseStrategy:
    """Current scheme: noise = alpha*last + sqrt(1-a^2)*fresh; last = noise.clone().
    Replicates acestep/engine/stream.py:391-396 exactly."""
    state = {"last": None}

    def fn(idx, shape, device, dtype, gen):
        fresh = torch.randn(*shape, device=device, dtype=dtype, generator=gen)
        if state["last"] is None or alpha <= 0.0:
            noise = fresh
        else:
            noise = alpha * state["last"] + math.sqrt(1.0 - alpha**2) * fresh
        state["last"] = noise.clone()
        return noise

    return NoiseStrategy(f"ema_a{alpha:.2f}", fn)


def make_anchor(alpha: float) -> NoiseStrategy:
    """Proposed fix: anchor = first noise; every later gen mixes anchor + fresh.
    Stable alpha; pairwise cos(gen_i, gen_j) = alpha**2 for i,j > 0; cos(gen_0, gen_k) = alpha."""
    state = {"anchor": None}

    def fn(idx, shape, device, dtype, gen):
        fresh = torch.randn(*shape, device=device, dtype=dtype, generator=gen)
        if state["anchor"] is None:
            state["anchor"] = fresh.clone()
            return fresh
        return alpha * state["anchor"] + math.sqrt(1.0 - alpha**2) * fresh

    return NoiseStrategy(f"anchor_a{alpha:.2f}", fn)


def make_lowfreq(alpha: float, smooth_frames: int = 25) -> NoiseStrategy:
    """Proposed fix: share only the LOW-FREQUENCY part of the noise along T.

    Hypothesis: noise correlation in the slow component should produce
    'phrase-level kinship' — same macroscopic envelope, fresh micro detail.
    Implementation: separate each noise into low-pass (moving average along T)
    and high-pass (residual) components. Share the low-pass via anchor blend
    at strength alpha; keep high-pass fully fresh.

    smooth_frames=25 == 1s at 25 fps, so the shared component is roughly
    'slow envelope of about a second' while everything faster stays fresh."""
    state = {"anchor_low": None}

    def lowpass(x: torch.Tensor) -> torch.Tensor:
        # Box filter along T (dim=1) with window = smooth_frames.
        # x: [1, T, D]; pad reflectively to keep T length.
        pad = smooth_frames // 2
        xp = torch.nn.functional.pad(x.transpose(1, 2), (pad, pad), mode="reflect")
        kern = torch.full((x.shape[-1], 1, smooth_frames),
                          1.0 / smooth_frames, device=x.device, dtype=x.dtype)
        lp = torch.nn.functional.conv1d(xp, kern, groups=x.shape[-1])
        return lp.transpose(1, 2)[:, : x.shape[1], :]

    def fn(idx, shape, device, dtype, gen):
        fresh = torch.randn(*shape, device=device, dtype=dtype, generator=gen)
        fresh_low = lowpass(fresh)
        fresh_high = fresh - fresh_low
        if state["anchor_low"] is None:
            state["anchor_low"] = fresh_low.clone()
            return fresh
        # Blend the low band toward the anchor, keep high band fresh.
        blended_low = alpha * state["anchor_low"] + math.sqrt(1.0 - alpha**2) * fresh_low
        return blended_low + fresh_high

    return NoiseStrategy(f"lowfreq_a{alpha:.2f}_s{smooth_frames}", fn)


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------
def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten().float()
    bf = b.flatten().float()
    n = (af.norm() * bf.norm()).clamp_min(1e-12)
    return float((af @ bf) / n)


def pair_matrix(items: list[torch.Tensor]) -> np.ndarray:
    n = len(items)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            v = cos_sim(items[i], items[j])
            M[i, j] = v
            M[j, i] = v
    return M


def mel_envelope(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Time-averaged mel-spectrogram (shift-invariant timbre fingerprint).
    Captures global timbre but NOT temporal musical content — two distinct
    pieces in the same genre have ~1.0 cosine here."""
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    mel = librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=sr, n_mels=64,
                                          hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel + 1e-10)
    return torch.from_numpy(mel_db.mean(axis=1))


def mel_full(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Full mel-spectrogram flattened (time-aligned timbral fingerprint).
    Sensitive to 'same content at same time positions' — the right metric
    for detecting musical continuity between consecutive generations."""
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    mel = librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=sr, n_mels=64,
                                          hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel + 1e-10)
    return torch.from_numpy(mel_db.flatten())


def chroma_full(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Full chromagram flattened (time-aligned harmonic fingerprint).
    Detects 'same notes/chords at same times'."""
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    chroma = librosa.feature.chroma_stft(y=wav.astype(np.float32), sr=sr,
                                         hop_length=512, n_fft=2048)
    return torch.from_numpy(chroma.flatten())


def onset_strength(wav: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """1D onset-strength envelope — detects rhythmic alignment between gens."""
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    onset = librosa.onset.onset_strength(y=wav.astype(np.float32), sr=sr,
                                          hop_length=512)
    return torch.from_numpy(onset)


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def run_strategy(
    strategy: NoiseStrategy,
    engine,
    config: DiffusionConfig,
    entry,
    context_latents: torch.Tensor,
    session: Session,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Run N_GENS generations using ``strategy`` for the input noise.

    The pipeline's built-in noise_sharing is disabled; we replace
    ``_make_noise`` so each submitted slot gets exactly the noise our
    strategy produced. ``request.seed`` is ignored by the patched
    function (would otherwise reseed torch globally).
    """
    print(f"\n[{strategy.name}]")
    pipe = StreamPipeline(engine, config, noise_sharing=0.0)

    # Per-strategy RNG so all strategies see the same random draws.
    rng = torch.Generator(device=device).manual_seed(SEED_BASE)

    noises: list[torch.Tensor] = []  # captured in order of submission

    def patched_make_noise(request: SlotRequest) -> torch.Tensor:
        # Compute deterministically next strategy noise, push into history.
        T = request.context_latents.shape[1]
        D = request.context_latents.shape[-1] // 2
        noise = strategy.fn(len(noises), (1, T, D), device, dtype, rng)
        noises.append(noise.cpu().clone())
        return noise

    pipe._make_noise = patched_make_noise  # type: ignore[method-assign]

    for i in range(N_GENS):
        req = SlotRequest(
            encoder_hidden_states=entry.encoder_hidden_states,
            encoder_attention_mask=entry.encoder_attention_mask,
            context_latents=context_latents,
            seed=None,           # ignored by patched _make_noise
            denoise=1.0,         # full t2m, no source dominance
        )
        pipe.submit(req)

    finished: list[torch.Tensor] = []
    max_ticks = N_GENS + pipe.depth + 5
    for _ in range(max_ticks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = pipe.tick()
        torch.cuda.synchronize()
        tick_ms = (time.perf_counter() - t0) * 1000
        if result is not None:
            finished.append(result.detach().cpu().clone())
            print(f"  gen {len(finished)}/{N_GENS}  tick={tick_ms:.0f}ms")
        if len(finished) >= N_GENS:
            break

    audios: list[np.ndarray] = []
    mel_envs: list[torch.Tensor] = []
    mel_fulls: list[torch.Tensor] = []
    chroma_fulls: list[torch.Tensor] = []
    onsets: list[torch.Tensor] = []
    out_dir = OUT_DIR / strategy.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, lat_cpu in enumerate(finished):
        lat = lat_cpu.to(device=device, dtype=dtype)
        audio_out = session.decode(Latent(tensor=lat))
        wav = audio_out.waveform.detach().cpu().float().squeeze(0).numpy()  # [C, S]
        audios.append(wav)
        mel_envs.append(mel_envelope(wav))
        mel_fulls.append(mel_full(wav))
        chroma_fulls.append(chroma_full(wav))
        onsets.append(onset_strength(wav))
        if SAVE_AUDIO:
            sf.write(str(out_dir / f"gen{i+1}.wav"), wav.T, SAMPLE_RATE)

    return {
        "name": strategy.name,
        "noises": noises[:N_GENS],
        "latents": finished,
        "audios": audios,
        "mel_envs": mel_envs,
        "mel_fulls": mel_fulls,
        "chroma_fulls": chroma_fulls,
        "onsets": onsets,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_matrix(name: str, M: np.ndarray) -> None:
    n = M.shape[0]
    print(f"    {name}:")
    for i in range(n):
        print("      " + " ".join(f"{M[i,j]:+.3f}" for j in range(n)))


def summarize_offdiag(M: np.ndarray) -> dict:
    """Mean/std/min/max of off-diagonal cells (i != j, upper triangle only)."""
    n = M.shape[0]
    vals = [M[i, j] for i in range(n) for j in range(i + 1, n)]
    if not vals:
        return {"mean": float("nan"), "std": 0.0, "min": float("nan"), "max": float("nan")}
    a = np.array(vals)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Noise sharing analysis — model-side")
    print(f"  T={T_FRAMES} frames ({T_FRAMES/25:.0f}s), N_GENS={N_GENS}, steps={INFER_STEPS}")
    print(f"  output dir: {OUT_DIR}")
    print("=" * 78)

    # ----- Setup -----
    print("\n[Setup] loading model (TRT 60s engines)...")
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

    print("[Setup] loading source audio (for semantic context only)...")
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

    print("[Setup] text encode...")
    cond = session.encode_text(
        tags="deathstep, heavy bass, dark atmosphere",
        instruction=TASK_INSTRUCTIONS["text2music"],
        refer_latent=source.latent,
        bpm=136, duration=T_FRAMES / 25.0, key="G# minor",
    )
    entry = cond.to_entries()[0]
    # context_latents shape: [1, T, 2*D] = [context, mask]. T is whatever
    # the VAE encoder produced for the source audio (may differ from
    # T_FRAMES due to the encoder's internal pooling).
    ctx_lat = source.context_latent.tensor.to(device=device, dtype=dtype)
    T_actual = ctx_lat.shape[1]
    D = ctx_lat.shape[2]
    cm = torch.ones(1, T_actual, D, device=device, dtype=dtype)
    context_latents = torch.cat([ctx_lat, cm], dim=-1)
    print(f"  context T = {T_actual} (D = {D})")

    config = DiffusionConfig(infer_steps=INFER_STEPS, shift=SHIFT, noise_on_cpu=True)

    # ----- Strategies -----
    strategies = [
        make_fresh(),
        make_fixed_seed(SEED_BASE),
        make_ema(0.70),
        make_ema(0.99),
        make_anchor(0.70),
        make_anchor(0.99),
        make_lowfreq(0.99, smooth_frames=25),   # share 1-second-scale envelope
        make_lowfreq(0.99, smooth_frames=75),   # share 3-second-scale envelope (~phrase)
    ]

    results = []
    for s in strategies:
        results.append(run_strategy(s, engine, config, entry, context_latents,
                                     session, device, dtype))

    # ----- Per-strategy similarity matrices -----
    print("\n" + "=" * 78)
    print("PER-STRATEGY SIMILARITY MATRICES (N_GENS x N_GENS, cosine)")
    print("  noise/latent: at the model interface")
    print("  mel-env     : time-AVERAGED mel envelope (genre/timbre, NOT temporal)")
    print("  mel-full    : flattened mel-spec (time-aligned: same content at same t?)")
    print("  chroma-full : flattened chroma   (time-aligned harmonic content)")
    print("  onset       : onset-strength envelope (rhythmic alignment)")
    print("=" * 78)
    summary_rows = []
    for r in results:
        Mn = pair_matrix(r["noises"])
        Ml = pair_matrix(r["latents"])
        Mme = pair_matrix(r["mel_envs"])
        Mmf = pair_matrix(r["mel_fulls"])
        Mcf = pair_matrix(r["chroma_fulls"])
        Mo = pair_matrix(r["onsets"])
        print(f"\n  [{r['name']}]")
        print_matrix("noise-cos  ", Mn)
        print_matrix("latent-cos ", Ml)
        print_matrix("mel-env-cos", Mme)
        print_matrix("mel-full-cos", Mmf)
        print_matrix("chroma-full-cos", Mcf)
        print_matrix("onset-cos  ", Mo)
        summary_rows.append((
            r["name"],
            summarize_offdiag(Mn), summarize_offdiag(Ml),
            summarize_offdiag(Mme), summarize_offdiag(Mmf),
            summarize_offdiag(Mcf), summarize_offdiag(Mo),
        ))

    # ----- Cross-strategy summary -----
    print("\n" + "=" * 78)
    print("OFF-DIAGONAL MEAN +/- STD  (pairwise similarity between distinct gens)")
    print("=" * 78)
    print(f"  {'strategy':>18s}  {'noise':>13s}  {'latent':>13s}  "
          f"{'mel-env':>13s}  {'mel-full':>13s}  {'chroma':>13s}  {'onset':>13s}")
    for name, sn, sl, sme, smf, scf, so in summary_rows:
        print(f"  {name:>18s}  "
              f"{sn['mean']:+.3f}+/-{sn['std']:.3f}  "
              f"{sl['mean']:+.3f}+/-{sl['std']:.3f}  "
              f"{sme['mean']:+.3f}+/-{sme['std']:.3f}  "
              f"{smf['mean']:+.3f}+/-{smf['std']:.3f}  "
              f"{scf['mean']:+.3f}+/-{scf['std']:.3f}  "
              f"{so['mean']:+.3f}+/-{so['std']:.3f}")

    # ----- Key question: does noise-space correlation predict audio similarity? -----
    print("\n" + "=" * 78)
    print("DELTA over 'fresh' baseline (positive = noise sharing carried into audio)")
    print("  fresh = independent randn each gen; sets the conditioning floor.")
    print("  fixed_seed = same noise every gen; this is the achievable CEILING.")
    print("  Useful gain only if a strategy's delta is well above noise floor and")
    print("  comparable to the fixed_seed delta.")
    print("=" * 78)
    fresh = next(r for r in summary_rows if r[0] == "fresh")
    _, _, _, fresh_mme, fresh_mmf, fresh_mcf, fresh_mo = fresh
    print(f"  baseline 'fresh' off-diag mean:")
    print(f"     mel-env={fresh_mme['mean']:+.3f}  mel-full={fresh_mmf['mean']:+.3f}  "
          f"chroma={fresh_mcf['mean']:+.3f}  onset={fresh_mo['mean']:+.3f}")
    print()
    print(f"  {'strategy':>18s}  {'noise-cos':>10s}  {'lat d':>8s}  "
          f"{'mel-env d':>10s}  {'mel-full d':>11s}  {'chroma d':>10s}  {'onset d':>10s}")
    for name, sn, sl, sme, smf, scf, so in summary_rows:
        dl = sl['mean']  # absolute (latent shows clear signal already)
        dme = sme['mean'] - fresh_mme['mean']
        dmf = smf['mean'] - fresh_mmf['mean']
        dcf = scf['mean'] - fresh_mcf['mean']
        do = so['mean'] - fresh_mo['mean']
        print(f"  {name:>18s}  {sn['mean']:>+10.3f}  {dl:>+8.3f}  "
              f"{dme:>+10.3f}  {dmf:>+11.3f}  {dcf:>+10.3f}  {do:>+10.3f}")

    print(f"\nDone. Audio saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
