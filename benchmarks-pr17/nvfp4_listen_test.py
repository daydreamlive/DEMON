"""A/B listen test for NVFP4 plugin engine vs bf16, mirroring fp8_listen_test.py.

Generates same-seed/same-prompt 60-second clips on bf16 and NVFP4. The NVFP4
engine has a fixed seq profile (seq=1500), so the listen test runs at
duration=60s. Output goes under benchmarks-pr17/listen/{bf16_60s,nvfp4}/.

Run::

    python benchmarks-pr17/nvfp4_listen_test.py
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Load the NVFP4 plugin DLL BEFORE the engine deserializer needs it.
PLUGIN_DLL = Path(__file__).resolve().parent.parent / "acestep" / "engine" / "trt" / \
    "plugins" / "nvfp4_linear" / "nvfp4_linear_plugin.dll"
venv_root = Path(sys.executable).resolve().parent.parent
site_packages = venv_root / "Lib" / "site-packages"
for d in (
    site_packages / "tensorrt_libs",
    site_packages / "torch" / "lib",
    Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
):
    if d.exists():
        os.add_dll_directory(str(d))
import torch
torch.cuda.init(); _ = torch.empty(1, device="cuda")
_plugin_dll = ctypes.CDLL(str(PLUGIN_DLL))
print(f"plugin DLL loaded: {PLUGIN_DLL}")

torch.set_grad_enabled(False)
import soundfile as sf
from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.paths import trt_engines_dir


# Same prompt set as fp8_listen_test.py for direct comparison.
PROMPTS = [
    ("dance", "dance music, four on the floor, kick drum, electronic, club, energetic synth bass, bright leads", 128, "F minor"),
    ("jazz",  "jazz piano trio, brushed drums, walking bass, late-night club", 140, "Bb major"),
    ("ambient", "ambient electronic, slow evolving pads, glassy textures, no drums", 80, "C minor"),
    ("metal", "metal, aggressive guitar riffs, fast double kick, growling vocals", 180, "E minor"),
    ("orch", "classical orchestral, sweeping strings, brass, timpani", 90, "D major"),
    ("folk", "acoustic folk, fingerpicked guitar, soft harmonica, brushes", 100, "G major"),
]


ENGINE_VARIANTS = {
    "bf16": "decoder_xl-turbo_mixed_refit_b4_60s",
    "nvfp4": "decoder_xl-turbo_nvfp4_noskip_b4_60s",
}

SHARED_ENGINES = {
    "vae_encode": "vae_encode_fp16_60s",
    "vae_decode": "vae_decode_fp16_60s",
}


def _resolve_engines(decoder_dir_name: str) -> dict[str, str]:
    root = trt_engines_dir()
    out: dict[str, str] = {
        "decoder": str(root / decoder_dir_name / f"{decoder_dir_name}.engine"),
    }
    for key, name in SHARED_ENGINES.items():
        out[key] = str(root / name / f"{name}.engine")
    for k, v in out.items():
        if not Path(v).exists():
            raise FileNotFoundError(f"Missing {k} engine: {v}")
    return out


def _generate_one(
    session, *, tag, prompt, bpm, key, seed, duration, steps, shift, out_path,
):
    t0 = time.perf_counter()
    cond = session.encode_text(
        tags=prompt,
        instruction=TASK_INSTRUCTIONS["text2music"],
        bpm=bpm, duration=duration, key=key,
    )
    t_text = time.perf_counter() - t0
    t1 = time.perf_counter()
    latent = session.generate(
        conditioning=cond, seed=seed, denoise=1.0,
        steps=steps, shift=shift, method="ode",
    )
    t_diffuse = time.perf_counter() - t1
    t2 = time.perf_counter()
    audio = session.decode(latent)
    t_decode = time.perf_counter() - t2

    wav = audio.waveform.detach().cpu().float().numpy()
    if wav.ndim == 3:
        wav = wav[0]
    wav = wav.T
    sf.write(str(out_path), wav, audio.sample_rate, format="WAV")
    total = time.perf_counter() - t0
    print(
        f"  {tag:>8s}  seed={seed}  text={t_text:.2f}s  "
        f"diffuse={t_diffuse:.2f}s  decode={t_decode:.2f}s  total={total:.2f}s -> {out_path.name}"
    )
    return dict(
        tag=tag, seed=seed, text_s=t_text, diffuse_s=t_diffuse,
        decode_s=t_decode, total_s=total, wav_path=str(out_path),
    )


def _run_variant(variant, *, prompts, seed, duration, steps, shift, out_dir):
    engines = _resolve_engines(ENGINE_VARIANTS[variant])
    print(f"\n[{variant}] engines:")
    for k, v in engines.items():
        print(f"  {k}: {v} ({Path(v).stat().st_size/1e6:.0f} MB)")

    session = Session(
        config_path="acestep-v15-xl-turbo",
        decoder_backend="tensorrt",
        vae_backend="tensorrt",
        trt_engines=engines,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (tag, prompt, bpm, key) in enumerate(prompts):
        out_path = out_dir / f"{i+1:02d}_{tag}.wav"
        rec = _generate_one(
            session, tag=tag, prompt=prompt, bpm=bpm, key=key,
            seed=seed, duration=duration, steps=steps, shift=shift,
            out_path=out_path,
        )
        rec["prompt"] = prompt
        rec["bpm"] = bpm
        rec["key"] = key
        records.append(rec)
    try:
        session.close()
    except Exception as e:
        print(f"[{variant}] session.close raised: {e}")
    del session
    gc.collect()
    torch.cuda.empty_cache()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=("both", "bf16", "nvfp4"), default="both")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "listen"))
    ap.add_argument("--prompts", nargs="*", type=int, default=None)
    args = ap.parse_args()

    prompts = PROMPTS if args.prompts is None else [PROMPTS[i] for i in args.prompts]
    out_root = Path(args.out_dir).resolve()
    print(f"[setup] out={out_root} duration={args.duration}s seed={args.seed}")

    all_records = {}
    # Match INDEX.md naming: bf16 dir name is just "bf16" (60s is the standard).
    subdir_map = {"bf16": "bf16_60s", "nvfp4": "nvfp4_noskip"}
    for variant in ("bf16", "nvfp4"):
        if args.only != "both" and args.only != variant:
            continue
        print(f"\n{'=' * 60}\nVARIANT: {variant}\n{'=' * 60}")
        records = _run_variant(
            variant,
            prompts=prompts, seed=args.seed,
            duration=args.duration, steps=args.steps, shift=args.shift,
            out_dir=out_root / subdir_map[variant],
        )
        all_records[variant] = records

    if "bf16" in all_records and "nvfp4" in all_records:
        print(f"\n{'=' * 60}\nSPEEDUP SUMMARY\n{'=' * 60}")
        for bf, nv in zip(all_records["bf16"], all_records["nvfp4"]):
            assert bf["seed"] == nv["seed"]
            print(f"  {bf['tag']:>8s}  bf16 diffuse={bf['diffuse_s']:.2f}s  "
                  f"nvfp4 diffuse={nv['diffuse_s']:.2f}s  "
                  f"speedup={bf['diffuse_s']/max(nv['diffuse_s'], 1e-6):.2f}x")

    (out_root / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "duration": args.duration, "steps": args.steps,
        "shift": args.shift, "engines": ENGINE_VARIANTS, "records": all_records,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
