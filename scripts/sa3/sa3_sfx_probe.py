"""Stable Audio 3 small-sfx: capability clips and TRT DiT parity.

Two subcommands, both on the GPU:

``clips``  Generate short clips eager across prompt types (impacts,
           ambiences, foley, UI, risers) and durations, save WAVs, and
           print/save a per-clip analysis from the signal itself: level,
           where the energy sits in time (attack, decay, how much of the
           clip is above -30 dB of peak), spectral centroid, and a
           stationarity score (RMS-envelope variation; low = steady bed,
           high = one-shot). A spectrogram PNG is written per clip.

``parity`` Per-step TRT-vs-eager velocity cosine on REAL trajectory
           states: an eager 8-step generation runs with a forward hook on
           the DiT that records every (x, t) the sampler feeds it; each
           recorded state is then pushed through the eager DiT and the
           built ``sa3_sfx_dit`` engine with the same production cond
           bundle, so both see identical inputs.

Run (repo root):
    python scripts/sa3/sa3_sfx_probe.py clips --out E:/Projects/sa3-variants/sfx/probe
    python scripts/sa3/sa3_sfx_probe.py parity --duration 54
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "sa3"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sa3_reference_generate import checkpoint_dir, load_local_model  # noqa: E402

MODEL_ID = "small-sfx"
SR = 44100

# (slug, prompt, seconds) -- prompt types the runbook asks for, plus two
# seconds_total variations of the same prompt.
CLIPS = [
    ("impact_door", "Heavy metal door slam, single impact, reverberant warehouse", 4),
    ("impact_boom", "Deep cinematic boom impact with long sub tail", 6),
    ("ambience_rain", "Steady heavy rain on a tin roof", 10),
    ("ambience_crowd", "Crowd murmur in a large indoor hall", 10),
    ("loop_engine", "Idling diesel engine, constant rumble", 10),
    ("foley_footsteps", "Footsteps on gravel, a few steps", 6),
    ("foley_glass", "Glass bottle shattering on concrete", 4),
    ("ui_chime", "Short UI confirm chime, clean bright notification", 2),
    ("ui_click", "Soft UI click, minimal interface tap", 1),
    ("riser", "Rising tension riser, building synth swell", 8),
    ("laser_3s", "Futuristic laser blast, sharp energy pulse, arcade style", 3),
    ("laser_12s", "Futuristic laser blast, sharp energy pulse, arcade style", 12),
]


def _load(device="cuda"):
    sam = load_local_model(checkpoint_dir(MODEL_ID), device=device, model_half=True)
    sam.model.eval()
    return sam


def _db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-9))


def analyse(audio: np.ndarray, sr: int = SR) -> dict:
    """``audio`` [C, N] float. Describes the signal, nothing else."""
    mono = audio.mean(axis=0)
    n = mono.shape[0]
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(mono ** 2)))
    hop = int(0.01 * sr)
    frames = n // hop
    env = np.sqrt(np.mean(mono[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    env_db = 20 * np.log10(env)
    top = env_db.max()
    loud = env_db > top - 30.0
    t_peak = float(np.argmax(env) * hop / sr)
    first = int(np.argmax(loud)) if loud.any() else 0
    last = int(len(loud) - 1 - np.argmax(loud[::-1])) if loud.any() else 0
    # attack: first frame within 3 dB of the max envelope, from first -30 dB frame
    near = np.nonzero(env_db > top - 3.0)[0]
    attack_ms = float((near[0] - first) * 10.0) if near.size else float("nan")
    # decay: from peak to the last frame above -30 dB
    decay_s = float((last - int(np.argmax(env))) * hop / sr)
    active_frac = float(loud.mean())
    # stationarity: std of envelope dB over the active region
    act = env_db[first: last + 1]
    env_std_db = float(np.std(act)) if act.size else 0.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    centroid = float((spec * freqs).sum() / max(spec.sum(), 1e-9))
    lo = float(spec[freqs < 200].sum() / max(spec.sum(), 1e-9))
    hi = float(spec[freqs > 5000].sum() / max(spec.sum(), 1e-9))
    tail_db = float(env_db[-20:].mean()) if env_db.size >= 20 else float(env_db.mean())
    return {
        "dur_s": round(n / sr, 2),
        "peak_dbfs": round(_db(peak), 1),
        "rms_dbfs": round(_db(rms), 1),
        "t_peak_s": round(t_peak, 2),
        "active_s": [round(first * hop / sr, 2), round(last * hop / sr, 2)],
        "active_frac": round(active_frac, 2),
        "attack_ms": round(attack_ms, 0),
        "decay_s": round(decay_s, 2),
        "env_std_db": round(env_std_db, 1),
        "centroid_hz": round(centroid, 0),
        "energy_below_200hz": round(lo, 2),
        "energy_above_5khz": round(hi, 2),
        "last_200ms_db": round(tail_db, 1),
        "lr_corr": round(float(np.corrcoef(audio[0], audio[1])[0, 1]), 3)
        if audio.shape[0] == 2 and np.std(audio[0]) > 0 and np.std(audio[1]) > 0 else None,
    }


def _spectrogram(audio: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mono = audio.mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    ax.specgram(mono, NFFT=2048, Fs=SR, noverlap=1536, cmap="magma", vmin=-120)
    ax.set_ylim(0, 16000)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("s")
    ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cmd_clips(args) -> int:
    import soundfile as sf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sam = _load()
    load_s = time.perf_counter() - t0
    rows = []
    for i, (slug, prompt, secs) in enumerate(CLIPS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            audio = sam.generate(prompt=prompt, duration=secs, steps=8, seed=args.seed + i)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        a = audio[0].float().cpu().numpy() if audio.dim() == 3 else audio.float().cpu().numpy()
        a = np.clip(a, -1, 1)
        wav = out / f"{i:02d}_{slug}.wav"
        sf.write(wav, a.T, SR, subtype="PCM_16")
        _spectrogram(a, out / f"{i:02d}_{slug}.png", f"{slug}: {prompt} ({secs}s)")
        row = {"file": wav.name, "prompt": prompt, "seconds": secs,
               "seed": args.seed + i, "gen_wall_s": round(wall, 3), **analyse(a)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "probe.json").write_text(
        json.dumps({"model": MODEL_ID, "load_s": round(load_s, 1), "steps": 8,
                    "backend": "eager fp16", "clips": rows}, indent=1),
        encoding="utf-8",
    )
    return 0


def cmd_stream(args) -> int:
    """The same prompts through the PRODUCTION streaming path (what the
    demo plays): SA3Backend over the process-cached context, with the
    context's song-length label and loop-wrap pad, depth 1, first fresh
    result rendered and trimmed to the playable length (48 kHz)."""
    import soundfile as sf

    from acestep.streaming.generator_backend import TickContext
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.sa3_backend import (
        DELIVERY_SAMPLE_RATE,
        SA3Backend,
        sa3_knob_specs,
    )
    from acestep.streaming.sa3_session import get_sa3_context
    from acestep.streaming.source import SAMPLE_RATE

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    context = get_sa3_context(MODEL_ID)
    rows = []
    for i, (slug, prompt, secs) in enumerate(CLIPS):
        cond = context.prepare_cond(prompt=prompt, duration=secs, steps=8)
        src = context.encode_source(
            (SAMPLE_RATE, torch.zeros(2, int(secs * SAMPLE_RATE))), cond.audio_sample_size,
        )
        b = SA3Backend.from_context(
            context, prompt=prompt, duration_s=secs,
            knob_state=KnobState(sa3_knob_specs()), cond=cond,
            source_latent_bct=src, steps=8, depth=1,
            dit_backend=args.backend, codec_backend=args.backend,
        )
        seed = args.seed + i
        knobs = {"sa3_denoise": 1.0, "seed": seed}
        tctx = TickContext(playhead_s=0.0, buffer_duration_s=secs)
        for _ in range(200):
            if b.produce(knobs, tctx, "generate"):
                break
        pcm = b.render_full().pcm[: int(round(secs * DELIVERY_SAMPLE_RATE))]
        dit = type(b.adapter.dit).__name__
        b.close()
        a = np.clip(pcm.T.astype(np.float32), -1, 1)
        wav = out / f"{i:02d}_{slug}_stream.wav"
        sf.write(wav, a.T, DELIVERY_SAMPLE_RATE, subtype="PCM_16")
        _spectrogram(a, out / f"{i:02d}_{slug}_stream.png",
                     f"{slug} STREAM label={context.cond_seconds_total(secs):.0f}s ({secs}s)")
        row = {"file": wav.name, "prompt": prompt, "seconds": secs, "seed": seed,
               "label_s": context.cond_seconds_total(secs), "L": int(cond.latent_frames),
               "dit": dit, **analyse(a, DELIVERY_SAMPLE_RATE)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / f"stream_{args.backend}.json").write_text(
        json.dumps({"model": MODEL_ID, "backend": args.backend,
                    "song_seconds": context.song_seconds,
                    "outro_pad_s": context.outro_pad_s, "clips": rows}, indent=1),
        encoding="utf-8",
    )
    return 0


def cmd_parity(args) -> int:
    """Per-step TRT vs eager on the production trajectory: an eager
    SA3Backend (production cond: song-length label, loop-wrap pad, so
    the attention padding mask is fully valid) generates once with a
    pre-hook on the DiT recording every (x, t); each state then goes
    through the eager DiT and the TRT engine with the same bundle."""
    from acestep.engine.sa3_trt import SA3TRTDit, find_dit_engine
    from sa3_parity_tier import infer_tier, judge
    from acestep.streaming.generator_backend import TickContext
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.sa3_backend import SA3Backend, sa3_knob_specs
    from acestep.streaming.sa3_session import get_sa3_context
    from acestep.streaming.source import SAMPLE_RATE

    context = get_sa3_context(MODEL_ID)
    dit = context.dit
    results = []
    for prompt in args.prompts:
        cond = context.prepare_cond(prompt=prompt, duration=args.duration, steps=8)
        bundle = cond.cond_bundle
        L = int(cond.latent_frames)
        pm = bundle.get("padding_mask")
        valid = int(pm.sum()) if torch.is_tensor(pm) else L
        engine = find_dit_engine(MODEL_ID, L)
        if engine is None:
            raise SystemExit(f"no sa3_sfx_dit engine covers L={L}")
        trt_dit = SA3TRTDit(engine, latent_frames=L,
                            seconds_total=context.cond_seconds_total(args.duration))
        src = context.encode_source(
            (SAMPLE_RATE, torch.zeros(2, int(args.duration * SAMPLE_RATE))),
            cond.audio_sample_size,
        )
        b = SA3Backend.from_context(
            context, prompt=prompt, duration_s=args.duration,
            knob_state=KnobState(sa3_knob_specs()), cond=cond,
            source_latent_bct=src, steps=8, depth=1,
            dit_backend="eager", codec_backend="eager",
        )
        states = []

        def hook(_mod, a, kw):
            x, t = a[0], a[1]
            if x.shape[-1] == L:
                states.append((x.detach().clone(), t.detach().clone()))

        h = dit.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            tctx = TickContext(playhead_s=0.0, buffer_duration_s=args.duration)
            for _ in range(64):
                if b.produce({"sa3_denoise": 1.0, "seed": args.seed}, tctx, "generate"):
                    break
        finally:
            h.remove()
        eager_final = b._last_result_latent.detach().float().clone()
        b.close()
        # Compounded: the same seeded generation end to end on the engine.
        bt = SA3Backend.from_context(
            context, prompt=prompt, duration_s=args.duration,
            knob_state=KnobState(sa3_knob_specs()), cond=cond,
            source_latent_bct=src, steps=8, depth=1,
            dit_backend="tensorrt", codec_backend="eager",
        )
        for _ in range(64):
            if bt.produce({"sa3_denoise": 1.0, "seed": args.seed}, tctx, "generate"):
                break
        trt_final = bt._last_result_latent.detach().float().clone()
        bt.close()
        compounded = torch.nn.functional.cosine_similarity(
            eager_final.flatten(), trt_final.flatten(), dim=0).item()
        per_step = []
        ref, ref_bundle = dit, bundle
        if args.fp32_ref:
            # The fp16 eager DiT carries its own error vs fp32 at the
            # chaotic early steps; an fp32 copy is the cleaner reference.
            import copy

            ref = copy.deepcopy(dit).float()
            ref_bundle = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                          for k, v in bundle.items()}
        for x, t in states:
            with torch.no_grad():
                v_e = (ref(x.float(), t.float(), **ref_bundle) if args.fp32_ref
                       else ref(x, t, **ref_bundle)).float()
            v_t = trt_dit.step_bundle(x, float(t.flatten()[0]), bundle).float().clone()
            cos = torch.nn.functional.cosine_similarity(v_e.flatten(), v_t.flatten(), dim=0).item()
            rel = ((v_t - v_e).norm() / v_e.norm()).item()
            step = {"t": round(float(t.flatten()[0]), 4), "cos": round(cos, 6),
                    "rel_rms": float(f"{rel:.3e}")}
            if args.fp32_ref:
                with torch.no_grad():
                    v_h = dit(x, t, **bundle).float()
                step["cos_fp16_eager_vs_fp32"] = round(torch.nn.functional.cosine_similarity(
                    v_e.flatten(), v_h.flatten(), dim=0).item(), 6)
            per_step.append(step)
        verdict = judge([s["cos"] for s in per_step], infer_tier(engine.parent.name))
        worst = verdict["min_step_cos"]
        if args.fp32_ref:
            del ref
            torch.cuda.empty_cache()
        row = {"prompt": prompt, "duration_s": args.duration, "L": L, "valid": valid,
               "reference": "fp32 eager" if args.fp32_ref else "fp16 eager",
               "label_s": context.cond_seconds_total(args.duration),
               "engine": engine.parent.name, "n_steps": len(per_step),
               "compounded_final_latent_cos": round(compounded, 6),
               "steps": per_step, "min_cos": worst, "worst_step": verdict["worst_step"],
               "tier": verdict["tier"], "tier_label": verdict["tier_label"],
               "gate": verdict["gate"], "pass": verdict["pass"]}
        results.append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(r["pass"] for r in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clips")
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=1000)
    st = sub.add_parser("stream")
    st.add_argument("--out", required=True)
    st.add_argument("--seed", type=int, default=1000)
    st.add_argument("--backend", choices=("eager", "tensorrt"), default="eager")
    p = sub.add_parser("parity")
    p.add_argument("--duration", type=float, default=54.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json", default=None)
    p.add_argument("--fp32-ref", action="store_true",
                   help="compare against an fp32 copy of the eager DiT")
    p.add_argument("--prompts", nargs="+", default=[
        "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style",
        "Steady heavy rain on a tin roof",
        "Dog barking next to a waterfall",
    ])
    args = ap.parse_args()
    return {"clips": cmd_clips, "stream": cmd_stream, "parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
