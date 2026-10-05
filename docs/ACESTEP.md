# ACE-Step engine reference

Operating detail for the ACE-Step v1.5 family: performance, tuning, acceleration backends, the Session API, TensorRT engine builds, and the web demo. Stable Audio 3 shares the streaming pipeline through a model adapter; the other families are documented in [FAMILIES.md](FAMILIES.md). The mechanism itself (ring buffer, per-slot schedules, shared curves, SDE source preservation, windowed decode) is in the [paper](https://arxiv.org/abs/2605.28657).

## Performance

All figures are from the paper (Tables 12, 14 and 15): RTX 5090, ACE-Step v1.5 turbo (2B), all-TensorRT, 8 steps, 60 s song (1500 latent frames at 25 Hz). Depth 4 is the production operating point.

**Pipeline depth trade-off** (paper Table 14). First effect is for a per-request change (denoise, prompt, source audio), which rides a new slot through its full 8-step schedule.

| Depth | Tick | Completion interval | Throughput | First effect |
|---|---|---|---|---|
| 1 | 14.0 ms | 112 ms | 8.9 generations/s | 112 ms |
| 2 | 24.3 ms | 97 ms | 10.3 generations/s | 219 ms |
| 4 | 42.8 ms | 88.5 ms | 11.3 generations/s | 471 ms |
| 8 | 81.1 ms | 81.1 ms | 12.3 generations/s | 649 ms |

**Parameter-change propagation at depth 8** (paper Table 12).

| Parameter change | First effect | Converged |
|---|---|---|
| Denoise, prompt switch, hint strength, source audio swap, timbre reference (per-request) | tick 8 / 648 ms | tick 8 / 648 ms |
| SDE curve (shared-mutable) | tick 1 / 81 ms | tick 5 / 405 ms |
| x0-target strength (shared-mutable) | tick 1 / 81 ms | tick 4 / 324 ms |
| LoRA refit | 1 tick (~81 ms) + 1.2 s refit | immediate |

Shared-mutable parameters (every per-frame curve in Table 2 of the paper) are read from shared state on every solver step, so a write is heard one tick later at any depth: 14 ms at depth 1, 43 ms at depth 4, 81 ms at depth 8.

**Across GPUs at depth 8** (paper Table 15).

| | RTX 5090 | RTX 4090 | RTX 3090 |
|---|---|---|---|
| Streaming tick | 81.1 ms | 112.6 ms | 240.2 ms |
| Tick + 15 s windowed decode | 102 ms | 139 ms | 287 ms |
| Throughput | 12.3 generations/s | 8.9 generations/s | 4.2 generations/s |
| Working-set VRAM | | 12.3 GB | 12.2 GB |

The demo fits a 24 GB card with the 60 s engines (see the VRAM breakdown below).

## Tuning

Three settings trade off against each other.

- **Ring buffer depth (`pipeline_depth`, 1 to 8).** Higher depth makes parameter sweeps glide through finer intermediate states at the cost of per-tick compute, VRAM and control latency; lower depth is snappier and cheaper.
- **Song duration.** TensorRT engines are profile-specific and reserve workspace sized to the profile, so a 240 s engine costs more VRAM and tick time than a 60 s engine even for a 60 s workload. Build only the durations you need.
- **VAE windowing.** The demo's default. With `vae_window > 0`, every streaming decode runs through the fixed 1 s windowed engine and only the requested window is decoded per call, which is what unlocks low-latency updates. Set it to 0 for full-length decode through the `vae_decode` engine.

Per-engine peak workspace, each measured in isolation on a 5090:

| Component       | 60s engine | 240s engine |          Δ |
|-----------------|-----------:|------------:|-----------:|
| Decoder (refit) |  13,511 MB |   15,911 MB |  +2,400 MB |
| VAE decode      |  10,547 MB |   10,814 MB |    +267 MB |
| VAE encode      |   4,178 MB |   10,614 MB |  +6,436 MB |

These are isolated peaks, not a live-runtime sum; at inference the decoder peak dominates, which is why the live demo fits a 24 GB card. Switching the three engines from 240 s to 60 s frees about 9 GB. Source: [`scripts/benchmarks/vram_60s_vs_240s_results.md`](../scripts/benchmarks/vram_60s_vs_240s_results.md).

## Acceleration backends

The DiT decoder and the VAE pick a backend independently: `tensorrt`, `compile` or `eager`.

| Component         | Backend     | Notes |
|-------------------|-------------|-------|
| Decoder           | `tensorrt`  | Fastest. Needs a built decoder engine for the target duration and checkpoint. Refit-enabled engines support LoRA swaps. |
| Decoder           | `compile`   | `torch.compile`. Long warmup, no engine to build. |
| Decoder           | `eager`     | Plain PyTorch, for debugging. |
| VAE encode/decode | `tensorrt`  | Fastest. The windowed-decode engine (`vae_decode_fp16_1s_fixed`) is built once and reused across durations. |
| VAE encode/decode | `compile`   | `torch.compile`. |
| VAE encode/decode | `eager`     | Plain PyTorch. |

From the web demo, `--accel {tensorrt|compile|eager}` sets both; `--decoder-accel` / `--vae-accel` override one:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel tensorrt
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel tensorrt --vae-accel eager
```

The cheapest worthwhile engine is the windowed VAE decoder: checkpoint- and duration-agnostic, and it unlocks the low-latency path. Pair it with `--decoder-accel compile` if you do not want to build the decoder engine yet.

## Fixtures and LoRAs

Demo audio pulls on first use from the [`daydreamlive/demon-fixtures-v2`](https://huggingface.co/datasets/daydreamlive/demon-fixtures-v2) dataset (the older `daydreamlive/demon-fixtures` is the fallback) into `<models dir>/fixtures/`; [`acestep/fixtures.py`](../acestep/fixtures.py) is the canonical set.

`demon-setup` downloads 16 genre LoRAs (jazz, phonk, lo-fi, punk, acoustic, ambient and deep house in 2B and XL variants, plus funk and deathstep; skip with `--skip-loras`). To add your own, drop a `.safetensors` file (optionally with a `<stem>.metadata.json` sidecar) anywhere under `<models dir>/loras/` and it appears on the next library refresh. See [`acestep/paths.py`](../acestep/paths.py) and [`acestep/lora_metadata.py`](../acestep/lora_metadata.py).

## Programmatic use: the Session API

Load the model once, then iterate.

```python
from acestep.engine.session import Session
from acestep.constants import TASK_INSTRUCTIONS

session = Session(
    decoder_backend="compile",  # or "tensorrt", "eager"
    vae_backend="compile",
    vae_window=0.36,            # 0 = full decode; >0 enables windowed decode
)

# Load audio, encode it, extract semantic context (cache across iterations).
source = session.prepare_source(audio)

# Encode text once. Reused across generations.
cond = session.encode_text(
    tags="deathstep death",
    instruction=TASK_INSTRUCTIONS["cover"],
    refer_latent=source.latent,
    bpm=136, duration=60.0, key="G# minor",
)

# Generate, decode, save. Cheap after warmup (~310 ms per iteration).
for seed in [1528, 9999, 42]:
    latent = session.generate(
        conditioning=cond,
        context_latent=source.context_latent,
        source_latent=source.latent,
        seed=seed,
    )
    save_audio(session.decode(latent), f"out_{seed}.wav")
```

Streaming is the same primitives wrapped in a `StreamHandle`:

```python
handle = session.stream(source=source, conditioning=cond, pipeline_depth=4)
for _ in range(N_TICKS):
    # Mutate handle.conditioning / handle.context_latent between ticks
    # to swap prompts or blend semantic hints live.
    latent = handle.tick()
    if latent is not None:
        audio = handle.decode(latent, t_start=window_start_s)

# Per-frame curve overrides bypass the ring buffer (1-tick latency):
handle.pipeline.set_shared_curve("velocity_scale", 1.2)
handle.pipeline.set_shared_curve("sde_denoise_curve", torch.tensor([...]))
```

Scripts: [`examples/session_demo.py`](../examples/session_demo.py) (persistent session, covers across seeds), [`examples/realtime_cover.py`](../examples/realtime_cover.py) (dual prompts, dual LoRAs, timbre and hint references, temporal masking, per-frame curves), and one standalone script per feature in [`examples/covers/`](../examples/covers/):

| Script | Feature |
|---|---|
| `cover_basic.py` | Standard cover pipeline (encode, condition, generate, decode) |
| `prompt_blend.py` | Two prompts blended with a temporal curve |
| `sde_denoise_curve.py` | Per-frame SDE re-noise modulation |
| `velocity_scaling.py` | Per-frame transformation rate control |
| `lora_generation.py` | LoRA-conditioned generation |
| `x0_target_blend.py` | Two-pass morphing toward a target latent |
| `conditioning_average.py` | Fuse two conditionings |
| `guidance_curve.py` | Per-frame CFG scale |
| `latent_noise_mask.py` | Latent-space inpainting |
| `initial_noise_curve.py` | Per-frame noise / source init mix |
| `ode_noise_injection.py` | Stochastic ODE step |
| `cover_semantic_blend.py` | Blend semantic hints from two sources |
| `x0_target_from_reference.py` | Pre-generate a target latent, morph toward it |

[`demos/test_stream_cover_graph.py`](../demos/test_stream_cover_graph.py) drives a streaming cover graph from Python.

## Building TensorRT engines

DEMON targets TensorRT 10.16.x. Plans are version- and GPU-specific, so rebuild after changing TensorRT, CUDA, the driver or the GPU. The minimal set for the web demo (what `demon-setup` builds) is the 60 s profile plus the fixed 1 s windowed VAE decode:

```bash
# Minimal set for the realtime web demo (what `demon-setup` builds).
uv run python -m acestep.engine.trt.build --preset minimal

# Full matrix (decoder refit + VAE encode/decode for 60s / 120s / 240s).
uv run python -m acestep.engine.trt.build --all

# 60s only.
uv run python -m acestep.engine.trt.build --all --duration 60

# Just the windowed VAE decoder (smallest, fastest to build, biggest payoff).
uv run python -m acestep.engine.trt.build --vae-only --duration 60

# Preview what would be built.
uv run python -m acestep.engine.trt.build --all --dry-run

# Force rebuild even if engines already exist.
uv run python -m acestep.engine.trt.build --all --force-rebuild

# Force ONNX re-export as well.
uv run python -m acestep.engine.trt.build --all --duration 60 --force-rebuild --force-onnx
```

ONNX intermediates are duration-agnostic and reused across builds. On disk:

```
~/.daydream-scope/models/demon/trt_engines/
  _onnx_vae/                      # shared across checkpoints, auto-reused
    vae_encode/vae_encode.onnx
    vae_decode/vae_decode.onnx
  _onnx_acestep-v15-turbo/        # checkpoint-specific
    decoder_refit/decoder_refit.onnx   # + external data shards
  spectral_decoder_mixed_refit_b8_60s/
    spectral_decoder_mixed_refit_b8_60s.engine
  vae_encode_fp16_60s/
    vae_encode_fp16_60s.engine
  vae_decode_fp16_1s_fixed/       # windowed decode, duration-independent
    vae_decode_fp16_1s_fixed.engine
  ...
```

Pass engine paths to `Session` when using the API directly (`acestep.paths.select_trt_engines` / `available_trt_engines` resolve them):

```python
from acestep.paths import available_trt_engines

engines, picked_dur = available_trt_engines(duration_s=60.0)
session = Session(
    decoder_backend="tensorrt",
    vae_backend="tensorrt",
    vae_window=0.36,
    trt_engines=engines,
)
```

The full build matrix, precision recipes, the XL/FP8 path and engine naming are in [TRT.md](TRT.md).

## The web demo: realtime_motion_graph_web

A Python backend and a Next.js front-end in one launcher. Feed it audio and a prompt, then twist knobs, draw automation curves, blend prompts, swap timbre and structure references and toggle LoRAs while the model plays.

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
# open http://localhost:6660
```

The launcher starts the backend on `:1318` and the Next.js dev server on `:6660`, installing the web app and the shared SDK (`packages/demon-client`) on first run. Backend flags go after `--`:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel tensorrt
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint xl
```

External static demos mount at runtime with `--demo <path>`; DEMON serves them as built static files plus the browser SDK at `/sdk/demon-client.js`. No-build examples live in [`daydreamlive/demon-example-apps`](https://github.com/daydreamlive/demon-example-apps):

```bash
git clone https://github.com/daydreamlive/demon-example-apps.git
uv run python -u -m demos.realtime_motion_graph_web.run --demo C:\path\to\demon-example-apps\apps\summon
```

What the page gives you: prompt A/B blending with a per-tick slider; a genre-grouped LoRA library with strength faders; timbre and structure references from fixtures, uploads or the mic; source-audio swap; drawn automation curves for denoise, hint strength, feedback, shift and LoRA strength; MIDI learn on any slider; audio-reactive WebGL visuals; audio and video recording; config import/export; and an onboard MCP server that exposes every action as a tool, so an agent can drive a live session.

Defaults live in [`demos/realtime_motion_graph_web/web/public/config.json`](../demos/realtime_motion_graph_web/web/public/config.json). Backend args, wire protocol, MCP setup and front-end architecture are in [`demos/realtime_motion_graph_web/README.md`](../demos/realtime_motion_graph_web/README.md).
