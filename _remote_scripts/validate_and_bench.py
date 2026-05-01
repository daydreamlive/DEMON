#!/usr/bin/env python3
"""Standalone validate + benchmark for an existing TRT engine.

Usage: validate_and_bench.py <engine_path>
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ACESTEP_MODELS_DIR", "/root/.daydream-scope/models/rtmg")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
sys.path.insert(0, "/workspace/acestep")

import numpy as np
import torch
import soundfile as sf

CHECKPOINT = "acestep-v15-xl-turbo"
RESULTS_DIR = Path("/workspace/test_output_h100")
REPORT = Path("/workspace/validate_report.json")
CORR_THRESHOLD = 0.95
MIN_RMS = 1e-4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def validate_decoder(engine_path):
    """Forward decoder once with PT and TRT, compare numerically."""
    log("validation: loading PT decoder ...")
    from acestep.engine.model_context import ModelContext
    from acestep.engine.trt.runtime import TRTDecoder

    ctx = ModelContext(
        config_path=CHECKPOINT,
        device="cuda",
        compile_decoder=False,
        compile_vae=False,
        skip_vae=True,
    )
    pt_decoder = ctx.model.decoder
    pt_decoder.eval()

    log("validation: loading TRT engine ...")
    trt_decoder = TRTDecoder(str(engine_path), device=torch.device("cuda"))

    B, T, D = 1, 1450, 64
    L = 200
    torch.manual_seed(7)
    hidden_states = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    timestep = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(B, L, 2048, device="cuda", dtype=torch.bfloat16)
    context_latents = torch.randn(B, T, 128, device="cuda", dtype=torch.bfloat16)

    log("validation: running PT decoder ...")
    with torch.no_grad():
        pt_out = pt_decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            context_latents=context_latents,
            use_cache=False,
        )
    if isinstance(pt_out, (tuple, list)):
        pt_out = pt_out[0]
    pt_out = pt_out.float()

    log("validation: running TRT engine ...")
    trt_out = trt_decoder(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        context_latents=context_latents,
    ).float()

    pt_flat = pt_out.flatten()
    trt_flat = trt_out.flatten()

    cos_sim = float(torch.nn.functional.cosine_similarity(pt_flat[None], trt_flat[None]).item())
    max_abs_diff = float((pt_flat - trt_flat).abs().max().item())
    pt_max = float(pt_flat.abs().max().item())
    trt_max = float(trt_flat.abs().max().item())
    rel_diff = max_abs_diff / max(pt_max, 1e-8)
    pt_rms = float(torch.sqrt((pt_flat ** 2).mean()).item())
    trt_rms = float(torch.sqrt((trt_flat ** 2).mean()).item())
    has_nan = bool(torch.isnan(trt_out).any().item())

    result = {
        "cos_sim": cos_sim,
        "max_abs_diff": max_abs_diff,
        "rel_diff": rel_diff,
        "pt_max": pt_max,
        "trt_max": trt_max,
        "pt_rms": pt_rms,
        "trt_rms": trt_rms,
        "has_nan": has_nan,
        "shape": list(pt_out.shape),
    }
    log(f"  cos_sim={cos_sim:.4f}  rel_diff={rel_diff:.4g}")
    log(f"  pt_rms={pt_rms:.4g}  trt_rms={trt_rms:.4g}  has_nan={has_nan}")

    passed = (
        not has_nan
        and cos_sim >= CORR_THRESHOLD
        and trt_rms > MIN_RMS
        and rel_diff < 0.5
    )
    result["passed"] = passed
    log(f"  validation: {'PASS' if passed else 'FAIL'}")

    del trt_decoder, ctx, pt_decoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_t2m_benchmark():
    log("benchmark: PT vs TRT t2m comparison ...")
    from acestep.constants import TASK_INSTRUCTIONS
    from acestep.engine.session import Session
    from acestep.paths import trt_engine_path

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    TAGS = "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads"
    LYRICS = "[instrumental]"
    BPM = 128
    KEY = "F minor"
    DURATION = 58.0
    SEEDS = [1528, 42, 9999]
    INFER_STEPS = 8
    SHIFT = 3.0

    def run_pass(label, decoder_backend, trt_engines):
        log(f"  pass={label}")
        t0 = time.time()
        session = Session(
            config_path=CHECKPOINT,
            decoder_backend=decoder_backend,
            vae_backend="eager",
            trt_engines=trt_engines,
        )
        log(f"    session ready in {time.time()-t0:.1f}s")
        cond = session.encode_text(
            tags=TAGS, lyrics=LYRICS, instruction=TASK_INSTRUCTIONS["text2music"],
            bpm=BPM, duration=DURATION, key=KEY,
        )
        timings = []
        wavs = {}
        for seed in SEEDS:
            t0 = time.time()
            latent = session.generate(
                conditioning=cond, seed=seed, duration=DURATION,
                steps=INFER_STEPS, shift=SHIFT,
            )
            t_gen = time.time() - t0
            t0 = time.time()
            audio = session.decode(latent)
            t_dec = time.time() - t0
            timings.append({"seed": seed, "gen_s": t_gen, "dec_s": t_dec})
            wav = audio.waveform
            if wav.dim() == 3:
                wav = wav.squeeze(0)
            wav_np = wav.detach().cpu().float().numpy()
            wavs[seed] = wav_np
            sf.write(RESULTS_DIR / f"t2m_xl_turbo_h100_{label}_seed_{seed}.wav",
                     wav_np.T, audio.sample_rate)
            log(f"    seed={seed} gen={t_gen:.3f}s dec={t_dec:.3f}s")
        del session
        gc.collect()
        torch.cuda.empty_cache()
        return timings, wavs

    pt_t, pt_wavs = run_pass("pytorch", "eager", None)
    trt_engines = {"decoder": str(trt_engine_path("decoder_xl-turbo_bf16mix_dynbatch_b8_240s"))}
    trt_t, trt_wavs = run_pass("tensorrt", "tensorrt", trt_engines)

    log("benchmark: computing PT vs TRT audio similarity ...")
    audio_compare = []
    for seed in [1528, 42, 9999]:
        pw = pt_wavs[seed].astype(np.float64)
        tw = trt_wavs[seed].astype(np.float64)
        n = min(pw.shape[-1], tw.shape[-1])
        pw = pw[..., :n].flatten()
        tw = tw[..., :n].flatten()
        if pw.std() > 0 and tw.std() > 0:
            corr = float(np.corrcoef(pw, tw)[0, 1])
        else:
            corr = float("nan")
        pt_rms = float(np.sqrt((pw ** 2).mean()))
        trt_rms = float(np.sqrt((tw ** 2).mean()))
        max_diff = float(np.abs(pw - tw).max())
        record = {
            "seed": seed, "samples": int(n),
            "corr": corr, "pt_rms": pt_rms, "trt_rms": trt_rms,
            "max_abs_diff": max_diff,
            "has_nan_trt": bool(np.isnan(tw).any()),
        }
        audio_compare.append(record)
        log(f"  seed={seed} corr={corr:.4f} pt_rms={pt_rms:.4g} trt_rms={trt_rms:.4g}")

    return {
        "pt_timings": pt_t,
        "trt_timings": trt_t,
        "audio_compare": audio_compare,
        "all_audio_passed": all(
            (not r["has_nan_trt"]) and r["trt_rms"] > MIN_RMS and r["corr"] > CORR_THRESHOLD
            for r in audio_compare
        ),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: validate_and_bench.py <engine_path>")
        sys.exit(2)
    engine_path = Path(sys.argv[1])
    if not engine_path.exists():
        log(f"engine NOT found: {engine_path}")
        sys.exit(3)
    log(f"engine: {engine_path}  ({engine_path.stat().st_size/1e9:.2f} GB)")

    report = {"engine": str(engine_path)}
    try:
        report["validation"] = validate_decoder(engine_path)
    except Exception as e:
        import traceback
        report["validation"] = f"exception: {type(e).__name__}: {e}"
        traceback.print_exc()

    if isinstance(report["validation"], dict) and report["validation"].get("passed"):
        try:
            report["benchmark"] = run_t2m_benchmark()
            report["status"] = "success" if report["benchmark"]["all_audio_passed"] else "audio_mismatch"
        except Exception as e:
            import traceback
            report["benchmark"] = f"exception: {type(e).__name__}: {e}"
            report["status"] = "benchmark_crashed"
            traceback.print_exc()
    else:
        report["status"] = "validation_failed"

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"report saved to {REPORT}")
    log(f"status={report['status']}")


if __name__ == "__main__":
    main()
