#!/usr/bin/env python3
"""Run all workflow variants in a single session with internal/external timing."""

import os
import sys
import time

import soundfile as sf
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes import Audio, Latent, Mask
from acestep.nodes.cond_nodes import TextEncode, ConditioningZeroOut, ConditioningAverage, ConditioningCombine
from acestep.nodes.curve_nodes import CurveRamp, CurveWave
from acestep.nodes.diffusion_nodes import DiffusionConfigNode, Generate
from acestep.nodes.lora_nodes import LoadLoRA, ApplyLoRA, RemoveLoRA
from acestep.nodes.mask_nodes import TemporalMask, SetLatentNoiseMask
from acestep.nodes.vae_nodes import VAEDecodeAudio, LatentBlend
from acestep.fixtures import audio_fixture

SOURCE_AUDIO = str(audio_fixture("inside_confusion_loop_60s_gsm.wav"))
LORA_PATH = ""
OUTPUT_DIR = os.path.join(project_root, "test_output", "session_all")


def load_audio(path: str, duration: float = 60.0) -> Audio:
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != 48000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    waveform = waveform[:2, : int(duration * 48000)]
    return Audio(waveform=waveform, sample_rate=48000)


def save_audio(audio: Audio, name: str) -> None:
    wav = audio.waveform
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    sf.write(path, wav.detach().cpu().float().numpy().T, audio.sample_rate)


class Timer:
    def __init__(self):
        self.steps = {}
        self._t0 = None
        self._name = None

    def start(self, name):
        if self._name:
            self.stop()
        self._name = name
        self._t0 = time.perf_counter()

    def stop(self):
        if self._name:
            self.steps[self._name] = time.perf_counter() - self._t0
            self._name = None

    def report(self):
        parts = [f"{k}={v:.3f}s" for k, v in self.steps.items()]
        total = sum(self.steps.values())
        return f"{' | '.join(parts)} | total={total:.3f}s"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Session init ===
    print("Creating session...", flush=True)
    t0 = time.perf_counter()
    s = Session(project_root=project_root, decoder_backend="compile", vae_backend="compile")
    print(f"Session ready: {time.perf_counter() - t0:.1f}s\n")

    # === Shared source prep ===
    print("Preparing source...", flush=True)
    t0 = time.perf_counter()
    audio = load_audio(SOURCE_AUDIO)
    source = s.prepare_source(audio)
    T = source.latent.tensor.shape[1]
    print(f"Source ready: {time.perf_counter() - t0:.2f}s  (T={T})\n")

    # Common conditioning (deathstep)
    deathstep_cond = s.encode_text(
        tags="deathstep death deaht deaht",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )

    # === Warmup (first generate triggers torch.compile) ===
    print("Warmup (torch.compile)...", flush=True)
    t0 = time.perf_counter()
    _ = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=0,
    )
    print(f"Warmup done: {time.perf_counter() - t0:.1f}s\n")

    results = {}

    # ------------------------------------------------------------------
    # 1. cover_basic (3 denoise levels)
    # ------------------------------------------------------------------
    for dn in [0.5, 0.75, 1.0]:
        name = f"cover_basic_d{int(dn*100)}"
        tm = Timer()
        t_ext = time.perf_counter()
        tm.start("generate")
        out = s.generate(
            conditioning=deathstep_cond,
            context_latent=source.context_latent,
            source_latent=source.latent,
            seed=1528, denoise=dn,
        )
        tm.start("decode")
        audio_out = s.decode(out)
        tm.stop()
        save_audio(audio_out, name)
        results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 2. velocity_scaling
    # ------------------------------------------------------------------
    name = "velocity_scaling"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("curve")
    vel_curve = CurveRamp().execute(start=0.2, end=1.5, length=T)["curve"]
    tm.start("generate")
    out = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528, velocity_scale=vel_curve,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 3. sde_denoise_curve
    # ------------------------------------------------------------------
    name = "sde_denoise_curve"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("curve")
    sde_curve = CurveRamp().execute(start=0.3, end=1.0, length=T)["curve"]
    tm.start("generate")
    out = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528, steps=8, shift=3.0,
        sde_denoise_curve=sde_curve,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 4. initial_noise_curve
    # ------------------------------------------------------------------
    name = "initial_noise_curve"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("curve")
    noise_curve = CurveRamp().execute(start=0.3, end=1.0, length=T)["curve"]
    tm.start("generate")
    out = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528, initial_noise_curve=noise_curve,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 5. ode_noise_injection
    # ------------------------------------------------------------------
    name = "ode_noise_injection"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("curve")
    inject_curve = CurveWave().execute(
        wave_type="sine", frames_per_cycle=25,
        amplitude=0.25, offset=0.25, length=T,
    )["curve"]
    tm.start("generate")
    out = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528, ode_noise_curve=inject_curve,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 6. guidance_curve (CFG with negative)
    # ------------------------------------------------------------------
    name = "guidance_curve"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("zero_out")
    neg_cond = ConditioningZeroOut().execute(conditioning=deathstep_cond)["conditioning"]
    tm.start("curve")
    cfg_curve = CurveRamp().execute(start=1.0, end=2.0, length=T)["curve"]
    tm.start("generate")
    out = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528, negative=neg_cond, guidance_curve=cfg_curve,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 7. conditioning_average (two prompts, 50/50 blend)
    # ------------------------------------------------------------------
    name = "conditioning_average"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("encode_b")
    cond_b = s.encode_text(
        tags="ambiet angelic synths a lot of synths",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )
    tm.start("average")
    blended = ConditioningAverage().execute(
        conditioning_a=deathstep_cond, conditioning_b=cond_b, weight=0.5,
    )["conditioning"]
    tm.start("generate")
    out = s.generate(
        conditioning=blended,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 8. prompt_blend (ConditioningCombine with temporal weight)
    # ------------------------------------------------------------------
    name = "prompt_blend"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("encode_a")
    cond_a = s.encode_text(
        tags="daft punk style electronic french house",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )
    tm.start("encode_b")
    cond_b = s.encode_text(
        tags="heavy demon techno, growling bass, industrial",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )
    tm.start("curve+combine")
    blend_mask = CurveWave().execute(
        wave_type="pulse", frames_per_cycle=151,
        amplitude=0.5, offset=0.5, length=T,
    )["curve"]
    combined = ConditioningCombine().execute(
        conditioning_a=cond_a, conditioning_b=cond_b,
        temporal_weight_b=Mask(tensor=blend_mask.tensor),
    )["conditioning"]
    tm.start("generate")
    out = s.generate(
        conditioning=combined,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1118,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 9. latent_noise_mask
    # ------------------------------------------------------------------
    name = "latent_noise_mask"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("mask")
    mask_curve = CurveWave().execute(
        wave_type="pulse", frames_per_cycle=300,
        amplitude=0.5, offset=0.5, length=T,
    )["curve"]
    mask = TemporalMask().execute(latent=source.latent, curve=mask_curve)["mask"]
    masked_latent = SetLatentNoiseMask().execute(
        latent=source.latent, mask=mask,
    )["latent"]
    tm.start("encode_text")
    cond_techno = s.encode_text(
        tags="driving techno with insane synths",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0,
    )
    tm.start("generate")
    out = s.generate(
        conditioning=cond_techno,
        context_latent=source.context_latent,
        source_latent=masked_latent,
        seed=632, denoise=0.85,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 10. cover_semantic_blend (two sources, blended hints)
    # ------------------------------------------------------------------
    name = "cover_semantic_blend"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("blend_hints")
    blend_curve = CurveWave().execute(
        wave_type="pulse", frames_per_cycle=150,
        amplitude=0.5, offset=0.5, length=T,
    )["curve"]
    ctx_blended = LatentBlend().execute(
        latent_a=source.context_latent, latent_b=source.context_latent,
        blend_curve=blend_curve,
    )["latent"]
    tm.start("encode_text")
    cond_jazz = s.encode_text(
        tags="jazz piano cover with swing rhythm",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="C major",
    )
    tm.start("generate")
    out = s.generate(
        conditioning=cond_jazz,
        context_latent=ctx_blended,
        source_latent=source.latent,
        seed=990,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 11. x0_target_blend (two-pass)
    # ------------------------------------------------------------------
    name = "x0_target_blend"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("encode_target")
    target_cond = s.encode_text(
        tags="daft punk style electronic french house",
        instruction=TASK_INSTRUCTIONS["cover"],
        refer_latent=source.latent,
        bpm=136, duration=60.0, key="G# minor",
    )
    tm.start("gen_target")
    target_latent = s.generate(
        conditioning=target_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528,
    )
    tm.start("curve")
    blend_curve = CurveRamp().execute(start=0.0, end=0.8, length=T)["curve"]
    tm.start("gen_blend")
    out = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528,
        x0_target=target_latent, x0_target_curve=blend_curve,
    )
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # ------------------------------------------------------------------
    # 12. lora_generation
    # ------------------------------------------------------------------
    name = "lora_generation"
    tm = Timer()
    t_ext = time.perf_counter()
    tm.start("load_lora")
    lora = LoadLoRA().execute(path=LORA_PATH, scale=1.3)["lora"]
    ApplyLoRA().execute(model=s.model, lora=lora)
    tm.start("generate")
    out = s.generate(
        conditioning=deathstep_cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=1528,
    )
    tm.start("remove_lora")
    RemoveLoRA().execute(model=s.model, lora=lora)
    tm.start("decode")
    audio_out = s.decode(out)
    tm.stop()
    save_audio(audio_out, name)
    results[name] = (time.perf_counter() - t_ext, tm)

    # === Report ===
    print("\n" + "=" * 80)
    print(f"{'WORKFLOW':<28} {'EXTERNAL':>10}  INTERNAL BREAKDOWN")
    print("=" * 80)
    for name, (ext, tm) in results.items():
        print(f"{name:<28} {ext:>8.3f}s  {tm.report()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
