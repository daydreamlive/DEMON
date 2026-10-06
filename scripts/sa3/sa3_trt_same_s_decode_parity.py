#!/usr/bin/env python3
"""SAME-S TensorRT decode parity on REAL latents (small-music codec).

Random-input cosine is a false positive for decoders, so every latent
here is real:

* ``session``: the canvas latent of a live in-process small-music session
  (SA3Backend, production conditioning, 8-step pingpong, denoise 0.9 off
  a fixture source) after it has finished generations: exactly what the
  codec decodes per tick.
* ``encoded``: a fixture SAME-S encoded through the production
  ``encode_source`` (real audio, the a2a source path).

Odd latent lengths are reported but not gated: the codec decodes them
eagerly (see ``SA3SAMECodec.trt_serves``).

Per latent the TRT engine (latent x ``pretransform.scale``) is compared to
the eager decode with the decode noise sources disabled (bottleneck
renoise + decoder mask_noise, ``sa3_decode_noise_mode``), against both an
fp32 pretransform and the production-loaded one. Context rows: the noisy
legacy eager decode vs deterministic eager, and TRT run twice. Metrics:
cosine over the flattened waveform, SNR dB = 10 log10(|ref|^2/|ref-x|^2).
Gate: cos >= 0.9998 vs deterministic eager. Also times the decode
(median of N, CUDA-synchronized) for eager vs TRT.

    python scripts/sa3/sa3_trt_same_s_decode_parity.py --out parity.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) in sys.path:
    sys.path.remove(str(_REPO))
sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

GATE = 0.9998
SESSIONS = (
    ("low_fi_Gm_loop_60s_gnm.wav", "lofi hip hop, mellow, instrumental"),
    ("thrash_metal_loop_60s_enm.wav", "thrash metal, distorted guitars, double kick"),
    ("prog_rock_loop_60s_enm.wav", "prog rock, analog synths, odd meter"),
    ("inside_confusion_loop_60s_gsm.wav", "ambient electronica, warm pads"),
)


def _metrics(ref: torch.Tensor, x: torch.Tensor) -> dict:
    n = min(ref.shape[-1], x.shape[-1])
    a = ref[..., :n].double().flatten()
    b = x[..., :n].double().flatten()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    err = float(((a - b) ** 2).sum())
    snr = float("inf") if err == 0 else 10.0 * torch.log10((a ** 2).sum() / err).item()
    return {"cos": round(cos, 7), "snr_db": round(snr, 2), "len_ref": int(ref.shape[-1]),
            "len_x": int(x.shape[-1])}


def _load(name: str):
    import soundfile as sf

    from acestep.fixtures import audio_fixture

    data, sr = sf.read(str(audio_fixture(name)), dtype="float32", always_2d=True)
    return int(sr), torch.from_numpy(data.T.copy())


def _session_latent(context, fixture: str, prompt: str, *, duration_s: float, steps: int):
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.sa3_backend import SA3Backend, sa3_knob_specs

    be = SA3Backend.from_context(
        context, prompt=prompt, duration_s=duration_s,
        knob_state=KnobState(sa3_knob_specs()), source_audio=_load(fixture),
        dit_backend="tensorrt", codec_backend="eager", steps=steps, depth=1,
    )
    prep = {"denoise": 0.9, "seed": 1528, "steps": steps, "shift": 1.0,
            "x0_target": 0.0, "feedback": 0.0, "feedback_depth": 1}
    lat = None
    finished = 0
    for _ in range(6 * steps):
        out = be._generate(prep)
        if out is not None:
            lat, finished = out, finished + 1
            if finished >= 3:
                break
    if lat is None:
        raise RuntimeError(f"session {fixture} produced no finished generation")
    return lat.movedim(1, 2).contiguous().float()  # [1, 256, T]


def _time(fn, n: int) -> float:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(ts), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=54.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--timing-n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from acestep.engine.sa3_context import SA3Context, SA3SAMECodec
    from acestep.engine.sa3_stream_helpers import sa3_decode_noise_mode

    ctx = SA3Context("small-music")
    codec = SA3SAMECodec(ctx, use_trt=True)
    if not codec.uses_trt:
        print("no SAME-S decode engine built; run sa3_build --same-s-decode")
        return 2
    trt_dec, scale = codec._trt, codec._scale
    pt = ctx.sam.model.pretransform
    prod_dtype = next(pt.parameters()).dtype
    print(f"pretransform dtype={prod_dtype} scale={scale} engine={trt_dec.engine_path.parent.name}")

    latents = []
    for fixture, prompt in SESSIONS[:3]:
        latents.append((f"session:{fixture.split('_loop')[0]}",
                        _session_latent(ctx, fixture, prompt, duration_s=args.duration,
                                        steps=args.steps)))
    for fixture in ("low_fi_Gm_loop_60s_gnm.wav", "thrash_metal_loop_60s_enm.wav"):
        sr, wav = _load(fixture)
        n = int(args.duration * ctx.sample_rate)
        enc = ctx.encode_source((sr, wav), n).float()
        if enc.shape[-1] % 2:  # the codec serves even lengths only
            latents.append((f"odd:{fixture.split('_loop')[0]}", enc))
            enc = enc[..., :-1]
        latents.append((f"encoded:{fixture.split('_loop')[0]}", enc))
    # Other profile points: a 24 s and a ~90 s window of real encoded audio.
    sr, wav = _load("low_fi_Gm_loop_60s_gnm.wav")
    for secs in (24.0, 90.0):
        enc = ctx.encode_source((sr, wav), int(secs * ctx.sample_rate)).float()
        latents.append((f"encoded:low_fi@{secs:.0f}s", enc))

    def eager(lat, *, dtype, noise: bool, seed=None):
        with torch.no_grad(), sa3_decode_noise_mode(ctx.sam, enabled=noise):
            if seed is not None:
                torch.manual_seed(seed)
            return pt.decode(lat.to(dtype)).float().clamp(-1, 1)[0]

    rows = []
    for name, lat in latents:
        lat = lat.to(ctx.device)
        t = int(lat.shape[-1])
        trt_a = trt_dec.decode(lat * scale)
        trt_b = trt_dec.decode(lat * scale)
        pt.float()
        ref32 = eager(lat, dtype=torch.float32, noise=False)
        pt.to(prod_dtype)
        refp = eager(lat, dtype=prod_dtype, noise=False)
        noisy = eager(lat, dtype=prod_dtype, noise=True, seed=7)
        row = {
            "latent": name, "T": t,
            "trt_vs_eager_fp32_det": _metrics(ref32, trt_a),
            "trt_vs_eager_prod_det": _metrics(refp, trt_a),
            "eager_prod_det_vs_fp32_det": _metrics(ref32, refp),
            "eager_noisy_vs_det": _metrics(refp, noisy),
            "trt_run_to_run": _metrics(trt_a, trt_b),
            "rms_dbfs": round(20 * torch.log10(ref32.pow(2).mean().sqrt()).item(), 1),
        }
        # Odd lengths are informational: the codec never sends them to
        # TRT (SA3SAMECodec.trt_serves), eager decodes them.
        row["codec_path"] = "tensorrt" if codec.trt_serves(t) else "eager"
        row["gate_pass"] = (
            row["trt_vs_eager_fp32_det"]["cos"] >= GATE
            if row["codec_path"] == "tensorrt" else None
        )
        rows.append(row)
        print(f"{name:28s} T={t:5d} trt~fp32 cos={row['trt_vs_eager_fp32_det']['cos']:.7f} "
              f"snr={row['trt_vs_eager_fp32_det']['snr_db']:.1f}  trt~prod "
              f"{row['trt_vs_eager_prod_det']['cos']:.7f}  prod~fp32 "
              f"{row['eager_prod_det_vs_fp32_det']['cos']:.7f}  noisy~det "
              f"{row['eager_noisy_vs_det']['cos']:.5f}  trt~trt "
              f"{row['trt_run_to_run']['cos']:.7f}  rms {row['rms_dbfs']} dBFS  "
              f"codec={row['codec_path']}")

    lat = latents[0][1].to(ctx.device)
    timing = {
        "T": int(lat.shape[-1]),
        "eager_prod_ms": _time(lambda: eager(lat, dtype=prod_dtype, noise=True), args.timing_n),
        "trt_ms": _time(lambda: trt_dec.decode(lat * scale), args.timing_n),
    }
    print(f"decode timing T={timing['T']}: eager {timing['eager_prod_ms']} ms, "
          f"TRT {timing['trt_ms']} ms")
    result = {"engine": trt_dec.engine_path.parent.name, "gate": GATE,
              "pretransform_dtype": str(prod_dtype), "scale": scale,
              "rows": rows, "timing": timing,
              "all_pass": all(r["gate_pass"] for r in rows if r["gate_pass"] is not None)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    print("ALL PASS" if result["all_pass"] else "GATE FAIL")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
