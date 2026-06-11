# DEMON

**Diffusion Engine for Musical Orchestrated Noise**

DEMON is a streaming diffusion engine for ACE-Step v1.5. Think StreamDiffusion, for audio: a ring buffer holds several in-flight generations at different denoising stages, advanced together per tick. After warmup, finished latents stream out at a steady rate of `depth/steps` generations per tick. End-to-end TensorRT keeps the tick tight; per-frame modulation knobs accept scalars or `[T]` curves and are hot-mutable mid-stream; ring buffer depth itself is hot-resizable. Streaming output is bit-identical to batch.

> Don't have a GPU, or just want to play first? Try the hosted instance at **[music.daydream.live](https://music.daydream.live)**.

## What DEMON is

The engine lives in [`acestep/`](acestep/). One process loads the model once and exposes two things:

1. A programmatic **Session API** ([`acestep/engine/session.py`](acestep/engine/session.py)) that wraps the streaming pipeline, the typed node graph, and the TRT runtime in a small set of methods (`prepare_source`, `encode_text`, `generate`, `decode`, `stream`, `apply_lora`).
2. A **typed node graph** ([`acestep/nodes/`](acestep/nodes/)) of 32 composable operations (latent / audio / conditioning / curve / mask / solver / config / DCW / channel guidance) wired through `NodeDefinition` / `NodePort` / `NodeParam`, with kwarg-validation at registration.

Anything on top, a CLI, a notebook, a VST, the bundled web demo, an MCP tool, or your own protocol, drives the same primitives. The library does not know or care which one you use.

## What the engine does

- **Streaming diffusion for ACE-Step v1.5.** `StreamPipeline` ([`acestep/engine/stream.py`](acestep/engine/stream.py)) maintains a ring buffer of in-flight generations. Each tick runs a batched decoder forward pass (two when CFG is active: positive + negative) that advances every active slot by one denoising step. The decoder dispatches to TensorRT or PyTorch through the same code path. Depth is hot-resizable mid-stream (`pipeline.set_depth(n)`); active slots drain naturally.
- **Heterogeneous slots.** Every in-flight slot carries its own `SlotRequest`: its own seed, its own `denoise` strength (with its own cached timestep schedule), its own source latent, its own per-frame curves, its own conditioning (one or more `SlotCondition`s with per-frame `temporal_weight` and per-condition `step_range`), its own CFG mode, its own x0 target, and its own latent-noise mask. A single ring buffer can mix a `denoise=1.0` regeneration, a `denoise=0.5` style transfer, and an RCFG-`self` request simultaneously and batch them in one forward pass.
- **Scalar-or-curve per-frame modulation.** Velocity scale, SDE re-noise, ODE noise injection, guidance scale, x0 target strength, x0 target curve, initial noise mix, APG momentum, CFG rescale, DCW scalers, and condition temporal weights all accept either a Python scalar or a `[T]` tensor, canonicalized through `normalize_curve` at the boundary so the kernels see one shape.
- **Channel guidance.** A `[1, T, 64]` per-channel gain applied to `xt` before each forward pass. Lives in its own surface (set via `pipeline.set_channel_gain_tensor(...)`) because its per-channel-and-per-frame shape doesn't fit the `[T]`-curve pattern.
- **Shared mutable curves.** Layered on top of the heterogeneous slots: `pipeline.set_shared_curve(name, value)` overrides one of the curve-shaped fields (`velocity_scale`, `sde_denoise_curve`, `ode_noise_curve`, `apg_momentum`, `x0_target_strength`, `cfg_rescale_curve`) for the next tick on every in-flight slot at once. The override takes effect immediately rather than waiting for new submissions to make their way through the pipeline. Pass `None` to revert that name to per-slot behavior.
- **Multi-condition compositing.** Within a single slot, the decoder runs once per active condition and velocities are blended per frame by `temporal_weight`; conditions are gated in and out of the schedule by `step_range`. `ConditioningBlend` (scalar alpha) and `ConditioningCombine` (per-frame temporal weights) are the typed entry points.
- **Three CFG modes.** Standard CFG (uncond forward every step), RCFG-`initialize` (one uncond forward per slot, cached for the rest of the schedule), and RCFG-`self` (zero uncond forwards: the slot's initial noise stands in as the virtual uncond velocity). All three layer APG momentum and an optional per-frame CFG rescale curve on top.
- **Latent-noise-mask inpainting.** Two-sided x0 blending matching ComfyUI semantics: pre-blend on `xt` (so the decoder sees correctly-noised context in preserved regions) and post-blend on the predicted `x0`. Supports a per-step strength function for progressive masking.
- **DCW post-step correction.** Wavelet-domain sampler-side correction from Yu et al. CVPR 2026, ported from upstream ACE-Step v0.1.7. Four modes (low / high / double / pix), with an optional advanced surface (`mult_blend`, `mag_phase`, `soft_thresh`) that at zero is byte-identical to the upstream reference. Hot-updatable via `pipeline.set_dcw(...)`.
- **Hot LoRA.** Register a directory once, then enable / set_strength / remove without rebuilding anything. The LoRA manager ([`acestep/engine/lora.py`](acestep/engine/lora.py)) handles the lifecycle and delta math; when the decoder is in TRT mode, applies route through a refitter against the live engine.
- **TRT acceleration end-to-end.** The DiT decoder, VAE encode, and VAE decode each pick `tensorrt | compile | eager` independently. The TRT decoder is refit-enabled, so LoRA swaps do not rebuild the engine. The VAE decode has a windowed variant (`vae_decode_fp16_3to30s`, range 3 to 30 s) that is built once and reused across all durations; the caller specifies the window start via `t_start`.
- **Bit-identical streaming vs. batch.** The streaming and one-shot paths compose the same pure step primitives from [`acestep/engine/ode_steps.py`](acestep/engine/ode_steps.py); they produce the same output.

## Tested on

NVIDIA RTX 3090, 4090, and 5090. The headline numbers below are from a 5090.

## Tuning: ring buffer depth, song duration, VAE windowing

Three knobs trade off against each other. Picking the right point on the curve is what makes DEMON run well on a given card.

**Ring buffer depth (`pipeline_depth`, 1 to 8).** The pipeline keeps `depth` in-flight generations at different denoise stages, advanced together each tick. After warmup, throughput is `depth/steps` finished generations per tick.

- Higher depth: parameter sweeps glide more smoothly (more slots in different denoise phases, so a curve change blends through finer intermediate states), at the cost of more per-tick batch compute and higher VRAM.
- Lower depth: knob changes feel snappier and more discrete (fewer slots between a parameter override and the next finished latent), with lower per-tick VRAM and compute.

**Song duration.** TRT engines are profile-specific. Each engine reserves workspace sized to its profile, so a 240 s engine costs more VRAM than a 60 s engine even when the workload is only 60 seconds. Per-engine peak workspace, each measured in isolation on a 5090:

| Component       | 60s engine | 240s engine |          Δ |
|-----------------|-----------:|------------:|-----------:|
| Decoder (refit) |  13,511 MB |   15,911 MB |  +2,400 MB |
| VAE decode      |  10,547 MB |   10,814 MB |    +267 MB |
| VAE encode      |   4,178 MB |   10,614 MB |  +6,436 MB |

These are per-engine peaks captured in separate subprocesses, not a live-runtime sum. At inference time the decoder peak dominates and the VAE workspaces do not peak alongside it, which is why the live demo fits on a 24 GB card. The comparison is what matters: switching three engines from 240 s to 60 s frees about 9 GB. Source: [`scripts/benchmarks/vram_60s_vs_240s_results.md`](scripts/benchmarks/vram_60s_vs_240s_results.md). Longer engines also pay more per-tick latency since the diffusion sequence length scales with duration. Build only the durations you need.

**VAE windowing.** Optional. When `vae_window > 0`, decode happens in overlapped time windows (range 3 to 30 s) instead of full-length, controlled by a `t_start` parameter on each decode call. This is what unlocks low-latency streaming updates: only the requested window is decoded per call rather than the full latent. Set to 0 to fall back to full-length decode.

## Performance

RTX 5090, ACE-Step v1.5 turbo (2B), all-TRT, `depth=4`, `steps=8`, `vae_window=3s`, 60 s source.

| Metric | Value |
|---|---|
| Tick (decoder forward, depth=4) | ~43 ms |
| Decode (windowed VAE, 3 s) | 4.5 ms |
| Throughput | 11.3 generations/second |
| Parameter convergence | ~248 ms |
| Per-frame control resolution | 25 Hz (40 ms latent steps) |
| Streaming vs. batch quality | bit-identical output |

## Acceleration backends

The DiT decoder and the VAE pick a backend independently. Three values each: `tensorrt`, `compile`, `eager`.

| Component         | Backend     | Notes |
|-------------------|-------------|-------|
| Decoder           | `tensorrt`  | Fastest. Requires a built decoder engine for the target duration and checkpoint. Refit-enabled engines support LoRA swaps. |
| Decoder           | `compile`   | `torch.compile`. Long warmup, no engine to build, good fallback. |
| Decoder           | `eager`     | Plain PyTorch. Useful for debugging. |
| VAE encode/decode | `tensorrt`  | Fastest. The windowed-decode engine (`vae_decode_fp16_3to30s`) is built once and reused across all durations. |
| VAE encode/decode | `compile`   | `torch.compile`. |
| VAE encode/decode | `eager`     | Plain PyTorch. |

From the bundled web demo, pass `--accel {tensorrt|compile|eager}` to set both at once, or `--decoder-accel` / `--vae-accel` to override one component at a time:

```bash
# All-TRT (recommended).
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel tensorrt

# TRT decoder, eager VAE (e.g. for debugging the decode path).
uv run python -u -m demos.realtime_motion_graph_web.run -- \
    --accel tensorrt --vae-accel eager
```

**Recommended baseline: TRT windowed VAE decoder at minimum.** It is the cheapest TRT engine to build, it is checkpoint- and duration-agnostic, and it unlocks the low-latency streaming path. Pair it with `--decoder-accel compile` if you do not want to build the decoder engine yet.

## Requirements

- Python 3.11
- NVIDIA GPU. Tested on RTX 3090, 4090, and 5090.
- ACE-Step v1.5 checkpoints in `checkpoints/` (auto-downloaded on first run)
- Node.js 20+ (only if you run the bundled web demo; first run installs `web/node_modules` automatically)

## Setup

```bash
uv sync
```

That is it for Python. Audio fixtures pull on first use from the [`daydreamlive/demon-fixtures`](https://huggingface.co/datasets/daydreamlive/demon-fixtures) Hugging Face dataset and cache under `~/.cache/huggingface/`. See [`acestep/fixtures.py`](acestep/fixtures.py) for the canonical set.

LoRAs are not auto-downloaded. Drop a `.safetensors` file into `$ACESTEP_MODELS_DIR/loras/` (defaults to `~/.daydream-scope/models/demon/loras/`) and it will appear in any consumer that scans the library on next refresh. See [`acestep/paths.py`](acestep/paths.py).

## Programmatic use: the Session API

The Session API is the engine's primary surface. Load the model once, then iterate.

```python
from acestep.engine.session import Session
from acestep.constants import TASK_INSTRUCTIONS

session = Session(
    decoder_backend="compile",  # or "tensorrt", "eager"
    vae_backend="compile",
    vae_window=3.0,             # 0 = full decode; >0 enables windowed decode
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

Quick-start scripts:

- [`examples/session_demo.py`](examples/session_demo.py): persistent session, iterate covers with different seeds.
- [`examples/realtime_cover.py`](examples/realtime_cover.py): a full real-time cover workflow with dual prompts, dual LoRAs, timbre / hint references, temporal masking, and engine-exclusive per-frame curves.
- [`examples/covers/`](examples/covers/): one standalone script per feature.

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

## Building TensorRT engines

DEMON targets TensorRT 10.16.x. Plans are version- and GPU-architecture-specific by default, so rebuild after changing TensorRT, CUDA, driver, or the GPU used for inference.

```bash
# Full matrix (decoder refit + VAE for 60s / 120s / 240s).
uv run python -m acestep.engine.trt.build --all

# 60s only (recommended starting point).
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

ONNX intermediates are duration-agnostic and auto-reused across builds; the model is only loaded when an export is actually needed.

```
trt_engines/
  _onnx/                          # shared, auto-reused across durations
    vae_encode/vae_encode.onnx
    vae_decode/vae_decode.onnx
    decoder/decoder.onnx          # + external data shards
    decoder_refit/decoder_refit.onnx
  decoder_mixed_refit_b8_60s/
    decoder_mixed_refit_b8_60s.engine
  vae_decode_fp16_3to30s/
    vae_decode_fp16_3to30s.engine
  ...
```

Pass engine paths to `Session` when using the API directly:

```python
session = Session(
    decoder_backend="tensorrt",
    vae_backend="tensorrt",
    vae_window=3.0,
    trt_engines={
        "decoder": "trt_engines/decoder_mixed_refit_b8_60s/decoder_mixed_refit_b8_60s.engine",
        "vae_encode": "trt_engines/vae_encode_fp16_60s/vae_encode_fp16_60s.engine",
        "vae_decode": "trt_engines/vae_decode_fp16_3to30s/vae_decode_fp16_3to30s.engine",
    },
)
```

## Demo applications

The engine is meant to be driven. The repository ships a flagship reference application plus a handful of focused entry points.

### realtime_motion_graph_web (the headline demo)

A Python backend plus a Next.js front-end in a single launcher. Feed it audio and a prompt, then twist knobs, draw automation curves, blend prompts, hot-swap timbre / structure references, and toggle LoRAs while the model generates and plays back continuously. Most of the engine surface above is exposed as a live control.

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
# then open http://localhost:6660
```

The launcher starts the backend on `:1318` and the Next.js dev server on `:6660`. Forward backend flags after `--`:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel tensorrt
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint xl
```

Highlights:

- **Prompt A ↔ B blending.** Two text fields plus a blend slider. One encoder pass per submission; the slider lerps per tick.
- **LoRA library.** Browse genre-grouped LoRAs, click to enable, drag faders for strength. Optional auto-prepend of trigger words to keep prompts honest.
- **Timbre and structure references.** Independent fixtures, uploaded clips, or short mic recordings bias instrument character and section / rhythm / dynamics. Mix freely.
- **Source-audio swap.** Library, upload, or record a 60 s snippet from your mic.
- **Schedule curves.** Draw automation over the timeline for denoise, hint strength, feedback, shift, and any LoRA strength. Smooth / linear / step interpolation.
- **MIDI learn.** Right-click any slider, wiggle a physical control, done. Mappings persist per option-profile.
- **Audio-reactive video.** WebGL2 shader pipeline with saturation-driven color parallax and bloom-on-kick.
- **Recording.** Capture audio (Opus/WebM, AAC/M4A fallback) or the live graph canvas as video with audio muxed in.
- **Config import / export.** Snapshot full live session state (knobs, prompts, LoRAs, curves) to JSON.
- **Onboard MCP server.** Every user-facing action exposed as an MCP tool. Drive the demo from Claude Code or any MCP client.

All defaults (knob positions, MIDI map seed, walk-window behavior, idle reset, LUFS matcher, audio-reactive shader params, XL-checkpoint overrides) live in [`demos/realtime_motion_graph_web/web/public/config.json`](demos/realtime_motion_graph_web/web/public/config.json). Edit, refresh, done.

See [`demos/realtime_motion_graph_web/README.md`](demos/realtime_motion_graph_web/README.md) for backend args, wire protocol, onboard MCP setup, and the front-end architecture.

### Other entry points

- [`examples/session_demo.py`](examples/session_demo.py): one-shot generation, persistent session.
- [`examples/realtime_cover.py`](examples/realtime_cover.py): real-time cover workflow exercising dual prompts, dual LoRAs, timbre / hint references, temporal masking, and engine-exclusive per-frame curves.
- [`examples/covers/`](examples/covers/): standalone per-feature scripts (see table above).
- [`demos/test_stream_cover_graph.py`](demos/test_stream_cover_graph.py): a streaming cover graph driven from Python.

## Tests

```bash
uv run pytest tests/ -v
```

## Research

The DEMON paper and two companion technical notes are forthcoming:

- DEMON paper (main)
- FastOobleckDecoder (VAE distillation)
- Latent Channel Semantics (64-channel VAE characterization)

Links land here as artifacts are released.

## Acknowledgments

DEMON is built on top of [ACE-Step](https://github.com/ace-step/ACE-Step). The base diffusion model, VAE, text encoder, and 5 Hz LM are all ACE-Step's work; without them, none of this exists. Huge thanks to the ACE-Step team for releasing the v1.5 weights and code under MIT.

If you use DEMON in your work, please also cite ACE-Step.

## Authors

DEMON originally created by Ryan Fosdick ([@RyanOnTheInside](https://ryanontheinside.com)). Maintained by [Daydream Live](https://daydream.live) and contributors.

## License

DEMON is distributed under the **GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`); see [`LICENSE`](LICENSE) for the full text. Among other things, this means modified versions made available to users over a network must offer those users the corresponding source code (AGPL §13).

Portions of DEMON are derived from [ACE-Step](https://github.com/ace-step/ACE-Step), originally released under the MIT license. The original MIT notice is preserved in [`LICENSE-MIT`](LICENSE-MIT) as required by that license; the ACE-Step portions remain available under MIT on their own terms, while the combined work is offered under AGPL-3.0-or-later.
