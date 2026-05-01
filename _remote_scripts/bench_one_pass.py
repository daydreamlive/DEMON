#!/usr/bin/env python3
"""Run a single t2m pass (PT or TRT) and save wavs + timings.

Used as a subprocess so PT pass and TRT pass don't share GPU state.
Output: timings + wav file paths printed as JSON on a single line.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch

torch.set_grad_enabled(False)

CHECKPOINT = os.environ.get("BENCH_CHECKPOINT", "acestep-v15-xl-turbo")
RESULTS_DIR = os.environ.get("BENCH_OUTDIR", "test_output/bench_one_pass")
TAGS = "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads"
LYRICS = "[instrumental]"
BPM = 128
KEY = "F minor"
DURATION = float(os.environ.get("BENCH_DURATION", "58.0"))
SEEDS = [int(s) for s in os.environ.get("BENCH_SEEDS", "1528,42,9999,7,13").split(",")]
INFER_STEPS = int(os.environ.get("BENCH_INFER_STEPS", "8"))
SHIFT = 3.0
TAG = os.environ.get("BENCH_TAG", "default")


def main():
    label = sys.argv[1]  # "pytorch" or "tensorrt"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from acestep.constants import TASK_INSTRUCTIONS
    from acestep.engine.session import Session
    from acestep.paths import trt_engine_path

    decoder_engine = os.environ.get("BENCH_DECODER", "decoder_mixed_refit_b8_240s")
    bench_vae = os.environ.get("BENCH_VAE", "0") == "1"
    vae_decode_name = os.environ.get("BENCH_VAE_DECODE", "vae_decode_fp16_240s")
    vae_encode_name = os.environ.get("BENCH_VAE_ENCODE", "vae_encode_fp16_240s")
    if label == "pytorch":
        decoder_backend = "eager"
        vae_backend = "eager"
        trt_engines = None
    elif label == "tensorrt":
        decoder_backend = "tensorrt"
        if bench_vae:
            from acestep.paths import default_trt_engines as _dte
            vae_backend = "tensorrt"
            trt_engines = _dte(
                decoder=decoder_engine,
                vae_decode=vae_decode_name,
                vae_encode=vae_encode_name,
            )
        else:
            vae_backend = "eager"
            trt_engines = {"decoder": str(trt_engine_path(decoder_engine))}
    else:
        print(json.dumps({"error": f"unknown label {label}"}))
        sys.exit(2)

    print(f"[child {label}] creating session ...", flush=True)
    t0 = time.time()
    session = Session(
        config_path=CHECKPOINT,
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
        trt_engines=trt_engines,
    )
    print(f"[child {label}] session ready in {time.time()-t0:.1f}s", flush=True)

    cond = session.encode_text(
        tags=TAGS, lyrics=LYRICS, instruction=TASK_INSTRUCTIONS["text2music"],
        bpm=BPM, duration=DURATION, key=KEY,
    )

    timings = []
    wav_paths = {}
    for seed in SEEDS:
        torch.cuda.synchronize()
        t0 = time.time()
        latent = session.generate(
            conditioning=cond, seed=seed, duration=DURATION,
            steps=INFER_STEPS, shift=SHIFT,
        )
        torch.cuda.synchronize()
        t_gen = time.time() - t0
        # free intermediate state before VAE decode (large)
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        t0 = time.time()
        audio = session.decode(latent)
        torch.cuda.synchronize()
        t_dec = time.time() - t0

        timings.append({"seed": seed, "gen_s": t_gen, "dec_s": t_dec})
        wav = audio.waveform
        if wav.dim() == 3:
            wav = wav.squeeze(0)
        wav_np = wav.detach().cpu().float().numpy()
        path = os.path.join(RESULTS_DIR, f"{TAG}_{label}_seed_{seed}.wav")
        sf.write(path, wav_np.T, audio.sample_rate)
        wav_paths[seed] = path
        print(f"[child {label}] seed={seed} gen={t_gen:.3f}s dec={t_dec:.3f}s -> {path}", flush=True)

        # also free the latent + audio between seeds
        del latent, audio, wav, wav_np
        gc.collect()
        torch.cuda.empty_cache()

    out = {"label": label, "timings": timings, "wav_paths": {str(k): v for k, v in wav_paths.items()}}
    print("RESULT:" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
