"""Validate + benchmark the sub-second real-VAE decode engines.

Three things:

  A. Production fidelity. Decode the kept center through the fp16
     ``vae_decode_fp16_1s_fixed`` engine (T=25) and compare to a wide-
     context fp32 PyTorch decode (ground truth). This is the total real-
     world error a 1 s streaming chunk would carry (fp16 + windowing).

  B. fp16 windowing isolation. Using one engine (``sub1s_dyn``) so the
     kernels are identical, sweep the overlap margin and compare each
     narrow decode's center to a wide decode's center on the SAME engine.
     This isolates the boundary receptive field in fp16 (does the 8-frame
     / 320 ms knee from the fp32 study survive fp16?).

  C. Latency. Per-call decode latency for the small engines vs the
     existing 3-30 s windowed engine and the 240 s canonical engine, at
     the shapes each paradigm actually runs.

Usage::

    uv run python scripts/benchmarks/vae_1s_validate_bench.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import torch

from acestep.fixtures import fixture_sidecar
from acestep.paths import checkpoints_dir, trt_engines_dir

FPS = 25
SPF = 1920  # samples per frame
MS = 1000.0 / FPS  # 40 ms/frame
TRT = trt_engines_dir()

ENGINE = {
    "1s_fixed": TRT / "vae_decode_fp16_1s_fixed" / "vae_decode_fp16_1s_fixed.engine",
    "sub1s_dyn": TRT / "vae_decode_fp16_sub1s_dyn" / "vae_decode_fp16_sub1s_dyn.engine",
    "3to30s": TRT / "vae_decode_fp16_3to30s" / "vae_decode_fp16_3to30s.engine",
    "240s": TRT / "vae_decode_fp16_240s" / "vae_decode_fp16_240s.engine",
}


# ---- minimal in-process TRT decode (mirrors runtime _trt_vae_decode) ----
_cache: dict = {}


def trt_load(path: Path):
    if path in _cache:
        return _cache[path]
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes
    eng = engine_from_bytes(bytes_from_path(str(path)))
    ctx = eng.create_execution_context()
    from polygraphy import cuda as pg_cuda
    entry = (eng, ctx, pg_cuda.Stream())
    _cache[path] = entry
    return entry


def trt_decode(path: Path, lat_bdt: torch.Tensor) -> torch.Tensor:
    _, ctx, stream = trt_load(path)
    lat = lat_bdt.to("cuda", torch.float32).contiguous()
    ctx.set_input_shape("latents", tuple(lat.shape))
    ctx.set_tensor_address("latents", lat.data_ptr())
    out_shape = tuple(ctx.get_tensor_shape("audio"))
    out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
    ctx.set_tensor_address("audio", out.data_ptr())
    ctx.execute_async_v3(stream.ptr)
    stream.synchronize()
    return out


def metrics(center: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (center - ref).abs()
    err_rms = (center - ref).pow(2).mean().sqrt().item()
    ref_rms = ref.pow(2).mean().sqrt().item()
    snr = 20.0 * torch.log10(torch.tensor(ref_rms / (err_rms + 1e-12))).item()
    cos = torch.nn.functional.cosine_similarity(
        center.reshape(1, -1), ref.reshape(1, -1)).item()
    return {"max_diff": diff.max().item(), "mean_diff": diff.mean().item(),
            "cos": cos, "snr_db": snr}


def main() -> None:
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")

    sc = fixture_sidecar("inside_confusion_loop_60s_gsm.wav")
    lat = sc.latent.float().transpose(1, 2).contiguous().to(dev)  # [1,64,T]
    T = lat.shape[-1]
    c = T // 2
    print(f"latent T={T} ({T/FPS:.1f}s), center frame {c}\n")

    # ---- fp32 ground truth (PyTorch real VAE), keep = 8 frames center ----
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(str(checkpoints_dir() / "vae"))
    vae = vae.to(dev).to(torch.float32).eval()
    keep = 8
    ks, ke = c - keep // 2, c - keep // 2 + keep
    G = 200
    gt = vae.decode(lat[:, :, ks - G:ke + G]).sample.float()
    gt_center = gt[:, :, G * SPF:(G + keep) * SPF].clone()

    # =================== A. production fidelity (1s fixed) ===================
    print("=" * 70)
    print("A. PRODUCTION FIDELITY  fp16 1s_fixed (T=25) vs fp32 wide-context GT")
    print("=" * 70)
    # 25-frame decode centered on keep: margin 8 left, 9 right.
    ml, mr = 8, 25 - keep - 8  # 8, 9
    win = lat[:, :, ks - ml:ke + mr]
    assert win.shape[-1] == 25, win.shape
    audio = trt_decode(ENGINE["1s_fixed"], win)
    center = audio[:, :, ml * SPF:(ml + keep) * SPF]
    mA = metrics(center, gt_center)
    print(f"  keep {keep} fr ({keep*MS:.0f} ms), margins L={ml} R={mr} fr, decode 25 fr")
    print(f"  max_diff={mA['max_diff']:.3e}  cos={mA['cos']:.8f}  SNR={mA['snr_db']:.1f} dB")

    # =============== B. fp16 windowing isolation (sub1s_dyn) ===============
    print("\n" + "=" * 70)
    print("B. fp16 WINDOWING ISOLATION  sub1s_dyn, same engine, keep=4 fr")
    print("   (ref = same engine @ 40 fr, margin 18 >> receptive field)")
    print("=" * 70)
    keepB = 4
    ksB, keB = c - keepB // 2, c - keepB // 2 + keepB
    ref40 = trt_decode(ENGINE["sub1s_dyn"], lat[:, :, ksB - 18:keB + 18])
    ref_center = ref40[:, :, 18 * SPF:(18 + keepB) * SPF].clone()
    print(f"{'margin_fr':>9}{'margin_ms':>10}{'decode_fr':>10}"
          f"{'max_diff':>12}{'cos':>12}{'snr_dB':>9}")
    print("-" * 62)
    rowsB = []
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16):
        a = trt_decode(ENGINE["sub1s_dyn"], lat[:, :, ksB - m:keB + m])
        cen = a[:, :, m * SPF:(m + keepB) * SPF]
        mm = metrics(cen, ref_center)
        rowsB.append({"margin_frames": m, "margin_ms": m * MS, "decode_frames": keepB + 2*m, **mm})
        print(f"{m:>9}{m*MS:>10.0f}{keepB+2*m:>10}"
              f"{mm['max_diff']:>12.2e}{mm['cos']:>12.8f}{mm['snr_db']:>9.1f}")

    # =========================== C. latency ===========================
    print("\n" + "=" * 70)
    print("C. DECODE LATENCY  (30 warmup, 200 timed; execute + sync)")
    print("=" * 70)
    plan = [
        ("1s_fixed", 25, "new 1s chunk"),
        ("sub1s_dyn", 24, "new: keep8+margin8"),
        ("sub1s_dyn", 40, "new: keep8+margin16"),
        ("3to30s", 75, "current windowed min (3s)"),
        ("3to30s", 99, "current default 3s win + .5s ovl"),
        ("240s", 125, "240s engine min (5s)"),
        ("240s", 1500, "240s engine @ 60s"),
    ]
    print(f"{'engine':>11}{'T_fr':>6}{'audio_s':>9}{'mean_ms':>9}"
          f"{'p50_ms':>8}{'min_ms':>8}{'ms/s_out':>10}  note")
    print("-" * 92)
    rowsC = []
    for label, Tf, note in plan:
        path = ENGINE[label]
        if not path.exists():
            print(f"{label:>11}{Tf:>6}   MISSING"); continue
        inp = torch.empty((1, 64, Tf), dtype=torch.float32, device=dev).normal_()
        _, ctx, stream = trt_load(path)
        ctx.set_input_shape("latents", tuple(inp.shape))
        ctx.set_tensor_address("latents", inp.data_ptr())
        out_shape = tuple(ctx.get_tensor_shape("audio"))
        out = torch.empty(out_shape, dtype=torch.float32, device=dev)
        ctx.set_tensor_address("audio", out.data_ptr())
        for _ in range(30):
            ctx.execute_async_v3(stream.ptr); stream.synchronize()
        ts = []
        for _ in range(200):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ctx.execute_async_v3(stream.ptr); stream.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        mean, p50, mn = sum(ts)/len(ts), ts[len(ts)//2], ts[0]
        audio_s = Tf / FPS
        ms_per_s = mean / audio_s
        rowsC.append({"engine": label, "T": Tf, "audio_s": audio_s,
                      "mean_ms": mean, "p50_ms": p50, "min_ms": mn, "ms_per_s_out": ms_per_s})
        print(f"{label:>11}{Tf:>6}{audio_s:>9.2f}{mean:>9.3f}"
              f"{p50:>8.3f}{mn:>8.3f}{ms_per_s:>10.2f}  {note}")

    out = Path(PROJECT_ROOT) / "scripts" / "benchmarks" / "vae_1s_validate_bench.json"
    out.write_text(json.dumps({"fidelity_A": mA, "isolation_B": rowsB, "latency_C": rowsC}, indent=2))
    print(f"\nJSON -> {out}")


if __name__ == "__main__":
    main()
