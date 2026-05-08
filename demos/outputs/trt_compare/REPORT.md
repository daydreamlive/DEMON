# TRT 10.13 vs TRT 10.16 — 2B turbo engine comparison

**Date:** 2026-05-07
**Hardware:** RTX 5090 (Blackwell, SM 12.0), NVIDIA driver 591.86, CUDA 13.1
**OS / venv:** Windows 11 Pro 26200, Python 3.11, torch 2.9.1+cu128
**Repo state:**
- `main` @ `24ed08c` (tensorrt 10.13.3.9.post1)
- PR 17 / `marco/feat/trt-10-16` @ `8466d11` (tensorrt 10.16.1.11)

## TL;DR

**TRT 10.16 is broken on this machine.** The 2B VAE encode and VAE decode engines built under TRT 10.16 segfault inside `execute_async_v3` on the very first call, regardless of:

- TRT patch (`10.16.1.11` and `10.16.0.72` both crash)
- engine builder workspace (4 GiB and 16 GiB both crash)
- shape (`(1, 64, 125)`, the engine's `min` shape, also crashes)
- code path (PR 17's `vae_nodes._trt_vae_decode`, raw `tensorrt` Python API, and `polygraphy run` CLI all crash identically)
- builder version (engine baked-in to PR 17 as `*.engine.trt10.16.bak` and a freshly-built engine on this hardware crash the same way)

Crash signature: SIGSEGV (exit code 139), no TRT error log emitted, all stdout/stderr buffers empty. The decoder engine works fine on TRT 10.16; only the VAE engines fail. Because `prepare_source` uses TRT VAE encode, **`demos/test_stream_cover_graph.py` segfaults before generating a single tick on TRT 10.16.**

The PR's own benchmark report claims `vae_decode_fp16_60s` works under TRT 10.16.1.11 at 16.3 ms decode mean, so the fault appears to be specific to this Blackwell + Windows + driver-591.86 stack rather than a code defect in the PR. Either way, this configuration cannot ship the upgrade until the VAE crash is investigated.

## What was tested

1. **Decoder isolation** — `TRTDecoder` direct-call benchmark at 3 shapes (B1/T750, B8/T750, B8/T1500), warmup 5 + 30 measured iters each.
2. **VAE encode isolation** — `_trt_vae_encode` direct-call at 30s and 60s of audio, warmup 3 + 15 iters.
3. **VAE decode isolation** — `_trt_vae_decode` direct-call at T=750 and T=1500, warmup 3 + 15 iters.
4. **End-to-end** — `demos/test_stream_cover_graph.py` (160 generations, 60s source, depth=8, DCW on, fixed seed).
5. **Quality diff** — same fixed-seed deterministic inputs on each TRT version, output tensors saved to `.pt` and diffed.

All artifacts live under `demos/outputs/trt_compare/`.

## Decoder isolation results (works on both versions)

`decoder_mixed_refit_b8_60s.engine` — fp16 mixed precision, 2B turbo.

| Config | TRT 10.13 mean | TRT 10.13 p95 | TRT 10.16 mean | TRT 10.16 p95 | Δ mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| B=1, T=750, enc=200 | 8.62 ms | 9.02 ms | 8.42 ms | 8.79 ms | **−2.4 %** |
| B=8, T=750, enc=200 | 41.91 ms | 42.92 ms | 40.77 ms | 41.53 ms | **−2.7 %** |
| B=8, T=1500, enc=200 | 79.94 ms | 80.47 ms | 77.53 ms | 78.13 ms | **−3.0 %** |

Decoder is consistently **~2–3 % faster** on 10.16. Far below the PR's claimed 11.2 % decoder tick speedup, but the PR's number includes more than just the decoder kernel.

CUDA peak allocated memory is identical between versions (≤ 0.04 GiB for the largest config); 10.16 does not regress decoder VRAM.

### Decoder quality

Same deterministic fp32 input on both versions, output captured as fp16 tensor:

| Config | mean abs diff | max abs diff | rel L2 |
| --- | ---: | ---: | ---: |
| B=1, T=750 | 7.37e-3 | 6.05e-2 | 5.03e-3 |
| B=8, T=750 | 8.50e-3 | 1.29e-1 | 5.70e-3 |
| B=8, T=1500 | 8.49e-3 | 1.41e-1 | 5.58e-3 |

~0.5 % relative L2 drift between TRT versions. Well within the noise floor of fp16 mixed-precision tactic selection (the two builders pick different kernels with different rounding paths). **Decoder quality is acceptable.**

## VAE isolation — TRT 10.13 (baseline)

Worked cleanly with `vae_decode_fp16_60s` (180 MB) and `vae_encode_fp16_60s` (180 MB).

| Config | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| vae_decode T=750 (30 s out) | 30.57 ms | 30.67 | 31.16 |
| vae_decode T=1500 (60 s out) | 55.56 ms | 55.68 | 56.46 |
| vae_encode 30 s | 31.53 ms | 31.40 | 32.33 |
| vae_encode 60 s | 55.82 ms | 56.03 | 56.56 |

## VAE isolation — TRT 10.16 (BROKEN)

Every variant of `vae_decode_fp16_60s.engine` and `vae_encode_fp16_60s.engine` built on this system under TRT 10.16 — and the `*.engine.trt10.16.bak` and `*.engine.trt10.16-staleONNX.bak` engines shipped in the PR — segfault inside `execute_async_v3` on the first call.

Investigation log:

| Step | Result |
| --- | --- |
| Run PR's prebuilt `*.engine.trt10.16.bak` | SIGSEGV |
| Run PR's `*.engine.trt10.16-staleONNX.bak` | SIGSEGV |
| Rebuild on TRT 10.16.1.11, ws=16 GiB | builds OK (377 MB, 69 s), runtime SIGSEGV |
| Rebuild on TRT 10.16.1.11, ws=4 GiB | builds OK (377 MB), runtime SIGSEGV |
| Rebuild on TRT 10.16.0.72, ws=16 GiB | builds OK, runtime SIGSEGV |
| Try smallest profile shape `(1, 64, 125)` | SIGSEGV |
| Try torch stream / null stream / polygraphy stream | all SIGSEGV |
| Try raw `tensorrt` Python API (no polygraphy) | SIGSEGV |
| Try `polygraphy run` CLI | SIGSEGV (exit 139, no log output) |
| Try fp32 build (no FP16 flag) | **build itself fails** with `Error 10: Could not find any implementation for node {ForeignNode[ONNXTRT_unsqueezeTensor...ONNXTRT_squeezeTensor_615]}` |

The fp32 builder error is the only diagnostic TRT emits. The fp16 builder picks tactics that the runtime then crashes on. The decoder engine is unaffected, so the issue is specific to the VAE 1D-conv ONNX graph.

The VAE engine on 10.16 is also **~2× larger** than on 10.13 (377 MB vs 180 MB for vae_decode, 339 MB vs 180 MB for vae_encode), which is consistent with TRT 10.16 selecting a different kernel layout for this graph.

## End-to-end `test_stream_cover_graph.py`

Same config: `acestep-v15-turbo`, 60 s source, depth=8, 8 steps, DCW on, fixed seed 1528, 160 ticks.

| Run | Wall time | tick avg | vae_decode avg | per-gen total | Decoded | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **TRT 10.13 — full TRT** | **17.5 s** | **79.3 ms** | **54.9 ms** | **109.6 ms** | 77 / 160 | reference run |
| TRT 10.16 — full TRT | — | — | — | — | — | **SIGSEGV at first VAE encode in `prepare_source`** |
| TRT 10.16 — TRT decoder + eager VAE | 28.1 s | 76.3 ms | 197.5 ms | 175.2 ms | 77 / 160 | only achievable variant |

WAVs: `stream_cover_trt10_13.wav` and `stream_cover_trt10_16_decoder_only.wav` (10.13-full vs 10.16-with-eager-VAE — not apples-to-apples because the VAE backend differs).

## Recommendation

**Do not merge PR 17 without resolving the VAE TRT 10.16 crash on this hardware.** Even with the PR-author-built `*.engine.trt10.16.bak` engines, the system cannot reach a single tick of `test_stream_cover_graph.py` here.

Suggested next steps for the PR author:
- Confirm what NVIDIA driver and CUDA toolkit they had installed when their benchmark passed. RTX 5090 + driver 591.86 + TRT 10.16 may need a specific driver to avoid this regression.
- Try a newer driver (≥ 600.x) before further engine-rebuild experiments.
- The fp32 builder error at `ForeignNode[ONNXTRT_unsqueezeTensor...ONNXTRT_squeezeTensor_615]` suggests reporting this VAE graph to NVIDIA as a TRT 10.16 regression candidate — provide the ONNX (`_onnx_vae/vae_decode/vae_decode.onnx`) for repro.

Decoder-only stats look healthy: ~3 % faster on 10.16 with sub-1 % output drift, no VRAM regression.

## Environment restoration

After the test run the venv was restored to TRT 10.13 (`tensorrt==10.13.3.9.post1`) and the live engines under `~/.daydream-scope/models/demon/trt_engines/{decoder_mixed_refit_b8_60s,vae_decode_fp16_60s,vae_encode_fp16_60s}/*.engine` were copied back from the corresponding `*.engine.trt10.13.bak` files. `debug_vae_decode.py` confirmed end-to-end VAE decode works again under 10.13.

Broken engines from the 10.16 attempt were moved to `~/.claude-trash/` rather than deleted, in case the PR author wants to inspect them.

## Files in this directory

| File | Contents |
| --- | --- |
| `bench_trt_iso.py` | Decoder + VAE isolation bench harness |
| `probe_engine_profiles.py` | Dumps each engine's TRT optimization profile |
| `compare_outputs.py` | Diffs `outputs_*.pt` across TRT versions |
| `compare_wavs.py` | Diffs the e2e WAVs |
| `debug_vae_decode*.py` | Repro scripts for the 10.16 VAE segfault |
| `build_vae_fp32.py` | Probe build to elicit the TRT-10.16 fp32 builder error |
| `timings_trt10_13.json` | 10.13 isolation bench numbers |
| `timings_trt10_16_dec.json` | 10.16 isolation bench numbers (decoder-only; VAE skipped) |
| `outputs_trt10_13.pt` / `outputs_trt10_16_dec.pt` | Tensor captures for quality diff |
| `bench_trt10_16_full.log` | Log showing the SIGSEGV right after VAE engine load |
| `vae_build_10_16.log` | Successful 10.16 VAE engine build log |
| `stream_cover_trt10_13.wav` | Reference 10.13 e2e WAV |
| `stream_cover_trt10_16_fulltrt.log` | Crash log for full-TRT 10.16 e2e run |
| `stream_cover_trt10_16_decoder_only.wav` / `.log` | 10.16 e2e with TRT decoder + eager VAE |
| `decoder_quality_diff.txt` / `wav_quality_diff.txt` | Diff summaries |
