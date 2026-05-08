## TRT 10.16 vs 10.13 on RTX 5090 — VAE engines crash, decoder is fine

Ran a full local A/B against this branch on a 5090 and there's a blocker: the 2B VAE engines built under TRT 10.16 segfault on the first `execute_async_v3` on this hardware. The decoder is healthy and slightly faster. Likely you only tested on the local 4090 — this looks like a Blackwell-only regression.

### Hardware / stack

- RTX 5090 (Blackwell, SM 12.0)
- NVIDIA driver `591.86`, CUDA 13.1
- Windows 11, Python 3.11, torch `2.9.1+cu128`
- main @ `24ed08c` for the 10.13 baseline, this branch (`marco/feat/trt-10-16`) @ `8466d11` for 10.16

### Decoder isolation — works, ~3 % faster on 10.16

`decoder_mixed_refit_b8_60s.engine`, fp16-mixed. Warmup 5 + 30 measured iters per shape.

| Config | TRT 10.13 mean | TRT 10.16 mean | Δ |
| --- | ---: | ---: | ---: |
| B=1, T=750, enc=200 | 8.62 ms | 8.42 ms | −2.4 % |
| B=8, T=750, enc=200 | 41.91 ms | 40.77 ms | −2.7 % |
| B=8, T=1500, enc=200 | 79.94 ms | 77.53 ms | −3.0 % |

Smaller than the 11.2 % tick speedup in the PR description, but the PR's number includes more than just the decoder kernel.

**Decoder output drift** (same fp32 deterministic input, fp16 output captured on both versions):

| Config | mean abs | max abs | rel L2 |
| --- | ---: | ---: | ---: |
| B=1, T=750 | 7.4e-3 | 6.0e-2 | 5.0e-3 |
| B=8, T=750 | 8.5e-3 | 1.3e-1 | 5.7e-3 |
| B=8, T=1500 | 8.5e-3 | 1.4e-1 | 5.6e-3 |

~0.5 % relative L2 — within the noise floor of fp16 mixed-precision tactic selection. Decoder quality is fine.

### VAE isolation — broken on 10.16

Every variant of `vae_decode_fp16_60s.engine` and `vae_encode_fp16_60s.engine` under TRT 10.16 segfaults inside `execute_async_v3` on the first call. **No TRT log emitted, exit code 139, all stdout/stderr buffers empty.**

What I tried:

| Variant | Result |
| --- | --- |
| Your shipped `*.engine.trt10.16.bak` | SIGSEGV |
| Your shipped `*.engine.trt10.16-staleONNX.bak` | SIGSEGV |
| Rebuild on TRT `10.16.1.11`, ws=16 GiB | builds OK (377 MB, 69 s), runtime SIGSEGV |
| Rebuild on TRT `10.16.1.11`, ws=4 GiB | builds OK, runtime SIGSEGV |
| Rebuild on TRT `10.16.0.72`, ws=16 GiB | builds OK, runtime SIGSEGV |
| Smallest profile shape `(1, 64, 125)` | SIGSEGV |
| Torch / null / polygraphy stream | all SIGSEGV |
| Raw `tensorrt` Python API (no polygraphy) | SIGSEGV |
| `polygraphy run …` CLI | SIGSEGV (no log) |
| **fp32 build (no FP16 flag)** | **builder fails** with `Error 10: Could not find any implementation for node {ForeignNode[ONNXTRT_unsqueezeTensor...ONNXTRT_squeezeTensor_615]}` |

The fp32 builder error is the only diagnostic TRT emits. fp16 picks a different tactic that the builder accepts but the runtime can't execute on Blackwell. Decoder engine is unaffected — it's specific to the VAE 1D-conv ONNX graph.

The 10.16 VAE engine is also **~2× larger than 10.13** (377 MB vs 180 MB for vae_decode, 339 MB vs 180 MB for vae_encode), consistent with TRT 10.16 picking a different kernel layout for this graph.

**TRT 10.13 baseline VAE numbers** (for reference):

| | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| vae_decode T=750 | 30.57 ms | 30.67 | 31.16 |
| vae_decode T=1500 | 55.56 ms | 55.68 | 56.46 |
| vae_encode 30s | 31.53 ms | 31.40 | 32.33 |
| vae_encode 60s | 55.82 ms | 56.03 | 56.56 |

### End-to-end `demos/test_stream_cover_graph.py`

Same config: `acestep-v15-turbo`, 60s source, depth=8, 8 steps, DCW on, fixed seed 1528, 160 ticks.

| Run | Wall | tick avg | vae_decode avg | per-gen | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **TRT 10.13 — full TRT** | **17.5 s** | **79.3 ms** | **54.9 ms** | **109.6 ms** | reference |
| TRT 10.16 — full TRT | — | — | — | — | **SIGSEGV at first VAE encode in `prepare_source`** — zero ticks generated |
| TRT 10.16 — TRT decoder + eager VAE | 28.1 s | 76.3 ms | 197.5 ms | 175.2 ms | only achievable variant on 10.16, **60 % slower than 10.13** |

So the user-visible behavior on a 5090 with this PR is either a hard crash (full TRT) or a meaningful regression (decoder TRT + eager VAE).

### Hypothesis

The PR description's benchmarks claim VAE works at TRT `10.16.1.11`, but the hardware/driver stack you tested with isn't documented. If you only tested on a 4090 (Ada, SM 8.9), TRT 10.16 may compile a different VAE tactic for 12.0 (Blackwell) that fails at runtime.

I cannot test driver 591.86 against, e.g., 600.x here, but the smoking gun is:

- 10.16 builder picks a tactic where the **fp32 path errors out at build time on Blackwell** (`ForeignNode[…unsqueezeTensor…squeezeTensor_615]`)
- the same 10.16 builder, with FP16 enabled, picks a *different* tactic that builds successfully but runtime-segfaults
- 10.13 has neither problem

That pattern is consistent with a TRT 10.16 SM 12.0 regression on the VAE graph. Worth (a) confirming on a 5090 if you have access, (b) trying a newer driver, (c) reporting the ONNX (`_onnx_vae/vae_decode/vae_decode.onnx` in this PR's layout) to NVIDIA as a TRT 10.16 + Blackwell repro.

### XL TRT — already covered, don't bother re-testing here

Don't worry about the XL TRT path in this PR. That work has already been done in [`ryanontheinside/arch/xl`](https://github.com/ryanontheinside/DEMON/tree/ryanontheinside/arch/xl) with very particular care and thorough testing — see commit [`9af4645`](https://github.com/ryanontheinside/DEMON/commit/9af4645658ce7f5e8b4bc99cdb51b26e50065a9e) ("xl-accel: TRT 10.16 upgrade, VAE Myelin workaround, XL turbo benchmarks") and the accompanying `XL_ACCEL_TRT_NOTES.md`.

That commit message also independently describes the exact failure mode this PR is hitting on a 5090:

> Pin `builder_optimization_level=1` to avoid TRT 10.15+ Myelin segfault on the oobleck VAE graph (Snake + WeightNorm convs + ConvTranspose1d) on RTX 5090. Bisected: optlvl 0/1 produce a working ~360 MB engine; optlvl 2 introduces a Myelin fusion that segfaults inside `execute_async_v3`.

So the fix for the 2B VAE crash is the same one already on that branch: pin `builder_optimization_level=1` in `vae_export.py`. The XL work in `ryanontheinside/arch/xl` is the canonical reference for the TRT 10.16 upgrade — please align this PR with it rather than reinventing the workaround.

### Recommendation

Don't merge as-is. The decoder upgrade is healthy (~3 % faster, ~0.5 % output drift, no VRAM regression), but the VAE TRT path is unusable on a 5090 here without the Myelin workaround already proven on `ryanontheinside/arch/xl`. Pull the `builder_optimization_level=1` pin from that branch into `acestep/engine/trt/vae_export.py` and we should be good.

Full artifacts (bench scripts, raw timings, repro scripts for the segfault, decoder fp16 tensor captures for diff, e2e WAVs, build logs) are saved locally if useful for follow-up. Happy to share specific files.

— Claude via remote
