"""Quick test: 2B turbo PT vs new TRT 10.16 engine, programmatic correlation."""
import gc, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_MODULES_CACHE", "C:/Users/ryanf/.cache/huggingface_modules")
os.environ.setdefault("HF_HOME", "C:/Users/ryanf/.cache/huggingface")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.paths import trt_engine_path

CHECKPOINT = os.environ.get("TEST_CHECKPOINT", "acestep-v15-turbo")
ENGINE = os.environ.get("TEST_ENGINE", "decoder_mixed_refit_b8_240s")
OUT_DIR = Path(os.environ.get("TEST_OUTDIR", "test_output/text_to_music_2b_turbo_trt10.16_test"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAGS = "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads"
LYRICS = "[instrumental]"
BPM = 128
KEY = "F minor"
DURATION = 58.0
SEEDS = [1528, 42, 9999, 7777, 2024]  # extra seeds to bracket the 1528 anomaly
INFER_STEPS = 8
SHIFT = 3.0


def run_pass(label, decoder_backend, trt_engines):
    print(f"\n=== {label} ===")
    t0 = time.time()
    session = Session(
        config_path=CHECKPOINT,
        decoder_backend=decoder_backend,
        vae_backend="eager",
        trt_engines=trt_engines,
    )
    print(f"  session ready {time.time()-t0:.1f}s")
    cond = session.encode_text(
        tags=TAGS, lyrics=LYRICS, instruction=TASK_INSTRUCTIONS["text2music"],
        bpm=BPM, duration=DURATION, key=KEY,
    )
    timings, wavs = [], {}
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
        timings.append((seed, t_gen, t_dec))
        wav = audio.waveform
        if wav.dim() == 3:
            wav = wav.squeeze(0)
        wav_np = wav.detach().cpu().float().numpy()
        wavs[seed] = wav_np
        sf.write(OUT_DIR / f"t2m_2b_turbo_{label}_seed_{seed}.wav", wav_np.T, audio.sample_rate)
        print(f"  seed={seed} gen={t_gen:.3f}s dec={t_dec:.3f}s")
    del session
    gc.collect()
    torch.cuda.empty_cache()
    return timings, wavs


pt_t, pt_wavs = run_pass("pytorch", "eager", None)
trt_t, trt_wavs = run_pass("tensorrt", "tensorrt", {"decoder": str(trt_engine_path(ENGINE))})

print("\n=== AUDIO COMPARE (PT vs TRT) ===")
all_pass = True
for seed in SEEDS:
    pw = pt_wavs[seed].astype(np.float64).flatten()
    tw = trt_wavs[seed].astype(np.float64).flatten()
    n = min(len(pw), len(tw))
    pw, tw = pw[:n], tw[:n]
    corr = float(np.corrcoef(pw, tw)[0, 1]) if pw.std() > 0 and tw.std() > 0 else float("nan")
    pt_rms = float(np.sqrt((pw ** 2).mean()))
    trt_rms = float(np.sqrt((tw ** 2).mean()))
    has_nan = bool(np.isnan(tw).any())
    passed = (not has_nan) and corr > 0.95 and trt_rms > 1e-4
    flag = "PASS" if passed else "FAIL"
    all_pass &= passed
    print(f"  seed {seed}: corr={corr:.4f}  pt_rms={pt_rms:.4g}  trt_rms={trt_rms:.4g}  {flag}")

print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

print("\n=== TIMINGS ===")
print(f"{'seed':>6} {'pt_gen':>10} {'trt_gen':>10} {'speedup':>10}")
for (s, pg, _), (_, tg, _) in zip(pt_t, trt_t):
    spd = pg / tg if tg > 0 else 0
    print(f"{s:>6} {pg:>10.3f} {tg:>10.3f} {spd:>9.2f}x")
