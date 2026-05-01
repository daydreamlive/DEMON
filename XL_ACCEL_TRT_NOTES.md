# XL Acceleration & TensorRT Version Notes

State of the TensorRT story for ACE-Step decoder + VAE engines on RTX 5090
(Blackwell) and Hopper H100. Captures everything we learned in the
TRT 10.13 → 10.16 upgrade attempt, the bugs we hit, the fixes that work,
the fixes that don't, and what's still open.

## TL;DR

- **TRT 10.16 is required for XL turbo on H100 (Hopper).** TRT 10.14
  has a `STRONGLY_TYPED` compiler segfault during the XL decoder build
  on H100 that is only fixed in 10.16. There is no workaround on 10.13/10.14/10.15
  that produces a working XL engine on Hopper.
- **TRT 10.15 and 10.16 introduced a Myelin codegen bug for the oobleck
  VAE graph.** The default VAE build segfaults inside `execute_async_v3`
  on RTX 5090. Worked around in `acestep/engine/trt/vae_export.py` by
  pinning `builder_optimization_level = 1`.
- **The repo's default decoder build profile is wrong for 240s engines.**
  `acestep/engine/trt/export.py` hardcodes `seq_opt = 750` (30 s) regardless of
  `seq_max`. The 60s engine (max=1500) tolerates this. The 240s engine
  (max=6000) does not on TRT 10.16: kernel selection picks something that
  is catastrophically slow at runtime and inflates GPU memory usage to the
  point of nearly OOM-ing a 32 GB 5090. The fix is to set `seq_opt = 1500`
  for the 240s build.
- **Per-input dtype binding bug in `diffusion.py`.** When the engine is
  built `STRONGLY_TYPED`, individual inputs (notably `timestep`) may be
  bf16 or fp32 depending on the export recipe. The runtime code now
  reads each input's declared dtype from the engine instead of assuming.
  Universally safe, lands regardless of TRT version.
- **VAE in-session slowdown on TRT 10.16 is real and unsolved.** The same
  VAE engine that runs in 49 ms standalone (60s) or 193 ms standalone (240s)
  runs much slower when the decoder TRT engine is also resident. On TRT 10.13
  the same engine in the same Session runs at the standalone speed. We do
  not have a build-side fix for this; it's the largest open question.

## Hardware / version matrix

|                              | TRT 10.13 | TRT 10.14 | TRT 10.15 | TRT 10.16 |
|------------------------------|-----------|-----------|-----------|-----------|
| RTX 5090 decoder build       | ✓         | ✓         | ✓         | ✓         |
| RTX 5090 decoder runtime     | ✓         | ✓         | ✓         | ✓         |
| RTX 5090 VAE build           | ✓         | ?         | ✓         | ✓         |
| RTX 5090 VAE runtime         | ✓         | ?         | **SEGV**  | **SEGV**¹ |
| H100 XL decoder build (`STRONGLY_TYPED`) | ?         | **SEGV**² | ?         | ✓         |
| H100 XL decoder runtime      | ?         | n/a       | ?         | ✓         |

¹ Fixed by pinning `builder_optimization_level = 1` in `vae_export.py`.
² The 10.14 H100 strongly_typed compiler crash is fixed in 10.16; the only
known reason for the upgrade above 10.13 in the first place.

## What's in this branch

### `pyproject.toml` — pinned to `tensorrt>=10.16,<10.17`
The whole point. H100 needs 10.16. We do not want two versions, two code
paths, or two engine sets.

### `acestep/engine/trt/vae_export.py` — Myelin segfault workaround

Added `builder_optimization_level: int = 1` to `VAETRTBuildConfig` and a
new line in `build_vae_trt_engine`:

```python
build_config.builder_optimization_level = config.builder_optimization_level
```

**Why level 1 specifically.** Bisected on 2026-04-08:

| optlvl | engine layers | build time | runtime |
|--------|---------------|------------|---------|
| 0      | 113           | 15 s       | works (slow runtime, 281 ms VAE decode at 60s)|
| **1**  | **115**       | **22 s**   | **works (49 ms VAE decode at 60s)** |
| 2      | 125           | 68 s       | **SEGV in execute_async_v3** |
| 3 (default) | 125      | 180 s      | **SEGV in execute_async_v3** |

The optlvl 1 engine is the same 360 MB and uses the same kernel set as
optlvl 3 for 115 of the 125 layers — only the ~10 extra fused kernels
that optlvl 2 introduces are different, and one of them is the
broken Myelin fusion. Tactic source bisection
(no `JIT_CONVOLUTIONS`, no `EDGE_MASK_CONVOLUTIONS`) at optlvl 3 still
segfaults; the bug is in core Myelin, not a single tactic source.

`fp32` (no FP16 flag) at optlvl 3 also segfaults, so it is not a
precision issue. The segfault is inside `libnvinfer.so.10` before any
per-kernel `[V]` log line is emitted. The compressed inspector output
shows the engine is being aggressively fused into Myelin layers with
names like `__myl_MulSinMulMulAdd_*` (Snake activation) and
`__myl_SumSqrtDivMulMul_*` (WeightNorm normalization). These pattern
fusions are what TRT 10.15+ added relative to 10.13 and what triggers
the bug.

### `acestep/engine/diffusion.py` — per-input dtype binding

Added `_trt_input_dtypes` map (built once from
`engine.get_tensor_dtype(name)` per input) and used it when allocating
the per-shape buffer set, instead of the single `_trt_io_dtype` and a
hardcoded `torch.float32` for `timestep`.

Reason: the older `bf16_mixed` export put `time_embed` in fp32; the
current `bf16_mixed` export leaves it in bf16. Reading the engine's
declared dtype avoids silent corruption when the buffer dtype mismatches
under `STRONGLY_TYPED`. No behavior change for any existing engine, lands
regardless of TRT version.

## Pitfalls — things we tried that didn't work

### `seq_opt = 5950` for the 240s decoder

Intuition was: align the decoder's optimization profile to the actual
b=1, T=5950 (238 s) workload. Result: builds successfully, but the
runtime is 7-8 s per generation (vs 442 ms with `seq_opt = 1500` and
440 ms on 10.13). TRT picks attention kernels at the high opt point that
fall off a cliff at the same shape they were "optimized" for. The same
non-monotonic pattern shows up at every other "high" opt point we tried:

| `seq_opt` | gen median (240s, b=1) on TRT 10.16 |
|-----------|-------------------------------------|
| 750 (current default in `export.py`) | 3873 ms — almost OOMs the 5090 |
| **1500**  | **442 ms — matches TRT 10.13** |
| 3000      | 3776 ms |
| 5950      | 7900 ms (worst) |

Take-away: kernel selection is non-monotonic in the opt point. Some
"middle" value gives a general kernel set that scales linearly to the
true runtime shape. The brute-force "just optimize for the actual
workload" approach is wrong.

### `decode_opt_frames = 5950` / `encode_opt_samples = 11424000` for the 240s VAE

Same intuition, same result. The VAE decode at `opt = 5950` is *worse*
in-session on TRT 10.16 than at the default `opt = 1500`:

| VAE 240s build               | vae_decode in-session (TRT 10.16) | vae_decode in-session (TRT 10.13) |
|------------------------------|-----------------------------------|-----------------------------------|
| `optlvl=1, opt=1500` (current `vae_export.py` default) | 1043 ms | 237 ms |
| `optlvl=1, opt=5950`         | 1973 ms                          | not tested |
| `optlvl=1, opt=1500`, with `--vae-window 5` | 158 ms              | 12.8 ms |

Same non-monotonic pattern. Same "VAE is much slower in-session than
standalone on 10.16, but fine on 10.13" mystery.

### Tactic source bisection for the Myelin segfault

`set_tactic_sources` with `JIT_CONVOLUTIONS` removed: still segfaults.
With `EDGE_MASK_CONVOLUTIONS` removed: still segfaults. Only
`builder_optimization_level = 1` (or 0) works.

### `fp32` build (no FP16 flag) for the VAE

Built fine. Still segfaults at runtime. Not a precision issue.

### Mixing TRT versions (decoder on 10.16, VAE on 10.13)

Explicitly forbidden by user direction and not a real option anyway:
ABI between TRT minor versions is not stable, the engines aren't
interchangeable, and there is no production benefit.

## Bench numbers (RTX 5090, 2B turbo, `demos/test_stream_cover_graph.py`)

All numbers from the canonical streaming pipeline at `depth=8`, identical
flags between versions. Times in ms.

### 60 s clip, full-clip VAE decode

| Engine            | Window  | TRT 10.13 (`.bak`) | TRT 10.16 (with fixes) |
|-------------------|---------|--------------------|------------------------|
| 2B turbo non-refit| full    | tick 76 / vae 59 / **total 108** | tick 69 / vae 52 / **total 96** |
| 2B turbo refit + LoRA | full | tick 76 / vae 62 / **total 110** | tick 76 / vae 54 / **total 106** |

### 240 s clip, full-clip VAE decode

| Engine                       | Window | TRT 10.13 (`.bak`) | TRT 10.16 (decoder `seq_opt=1500` rebuild + optlvl=1 VAE) |
|------------------------------|--------|--------------------|-------------------------------------------------------|
| 2B turbo non-refit           | full   | tick 386 / vae 237 / **total 422** | tick 360 / vae 1044 / **total 503** |
| 2B turbo non-refit           | win=5  | tick 391 / vae 13 / **total 397**  | tick 381 / vae 158 / **total 410**  |

The decoder fix alone closes the decoder gap (10.16 actually edges out
10.13 on tick). The VAE 240s in-session slowdown remains. `--vae-window 5`
amortizes the VAE cost so the per-gen total is within 3 % of TRT 10.13.

The original `decoder_mixed_b8_240s.engine.trt10.16.bak` (the build with
`seq_opt = 750`) ran the same workload at tick 2500-4200 ms with erratic
VRAM and *almost crashed the host* before being killed at tick 82. Do
**not** use the default `_build_decoder_engine` settings for 240s on
TRT 10.16.

### What's NOT measured

The matrix below is what we have. Empty cells were never run through
the canonical script in this session. **2B base** and **XL turbo** were
not benched through `test_stream_cover_graph.py` at all in this session
(only `2B turbo`).

| Engine | LoRA | Window | TRT 10.13 | TRT 10.16 |
|--------|------|--------|-----------|-----------|
| 2B turbo non-refit 60s  | -    | full   | 108 ms / gen | 96 ms / gen |
| 2B turbo refit 60s      | yes  | full   | 110 ms / gen | 106 ms / gen |
| 2B turbo non-refit 240s | -    | full   | 422 ms / gen | 503 ms / gen (with fixes) |
| 2B turbo non-refit 240s | -    | win=5  | 397 ms / gen | 410 ms / gen (with fixes) |
| 2B turbo refit 240s     | yes  | any    | not run    | not run    |
| 2B base 60s             | any  | any    | not run via canonical | not run via canonical |
| 2B base 240s            | -    | -      | engine does not exist | engine does not exist |
| XL turbo 60s            | -    | any    | not run via canonical | not run via canonical |
| XL turbo 240s           | -    | -      | engine does not exist | engine does not exist |

## What still needs to be done

1. **Land the `seq_opt` fix in `acestep/engine/trt/build.py` /
   `export.py`.** Right now the default is `seq_opt = 750` regardless of
   `seq_max`. The 240s engine produced by `python -m
   acestep.engine.trt.build --duration 240` is the broken one. The fix
   is one line: scale `seq_opt` to `min(seq_max, 1500)` in the
   `_build_decoder_engine` wrapper. This is **not** in this branch yet
   because we wanted to confirm the fix on 10.16 first.
2. **Build the missing 240s engines for 2B base and XL turbo** if
   long-clip support is required for those checkpoints.
3. **Build the missing XL turbo refit engine** if LoRA on XL is required.
4. **Investigate the in-session VAE slowdown on TRT 10.16.** The standalone
   bench is 49 ms (60s) / 193 ms (240s) but in-session it's 51-54 ms (60s)
   / 1044 ms (240s). The 60s case is essentially fine; the 240s case is
   4x slower than 10.13 in-session. Hypotheses: (a) GPU memory
   fragmentation after the decoder runs, (b) 10.16 picks tactics for the
   240s VAE that are sensitive to the post-decoder GPU state, (c) some
   shared CUDA stream contention that 10.13 does not exhibit. None proven.
5. **Build the matrix for 2B base and XL turbo** through the canonical
   script. Not done in this session.
6. **Decide whether `--vae-window` should be the production default for
   240s.** Numbers strongly suggest yes — it makes the 10.13 / 10.16 gap
   essentially zero.

## Build / test recipes worth keeping

### Rebuild a single 240s decoder with the seq_opt fix
```bash
SEQ_OPT=1500 INCLUDE_REFIT=1 python _remote_scripts/rebuild_240s_decoder.py
```

### Rebuild the 240s VAE with the optlvl=1 segfault workaround
```bash
DECODE_OPT=1500 ENCODE_OPT=2880000 python _remote_scripts/rebuild_240s_vae.py
```

### Swap engine backups on disk between TRT versions
```bash
python _remote_scripts/swap_engines.py 10.13   # restore .trt10.13.bak
python _remote_scripts/swap_engines.py 10.16   # restore .trt10.16.bak (decoders) + optlvl=1 VAE
```

### Run the canonical streaming bench
```bash
# 60s
python demos/test_stream_cover_graph.py
python demos/test_stream_cover_graph.py --lora

# 240s (windowed VAE recommended for production-realistic numbers)
python demos/test_stream_cover_graph.py --duration 240 --vae-window 5
```

### Reproduce the VAE Myelin segfault for any future regression check
```bash
python _remote_scripts/test_vae_alone.py
# or
python _remote_scripts/test_vae_encode_alone.py
```

## Files changed by this branch

- `pyproject.toml` — `tensorrt>=10.16,<10.17`
- `acestep/engine/trt/vae_export.py` — `builder_optimization_level = 1`
- `acestep/engine/diffusion.py` — per-input TRT dtype map
- `XL_ACCEL_TRT_NOTES.md` (this file)
