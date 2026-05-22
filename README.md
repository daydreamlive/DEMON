# DEMON

**Diffusion Engine for Musical Orchestrated Noise**

A real-time composable diffusion engine for ACE-Step v1.5. Feed it audio and a prompt, then twist knobs, draw automation curves, blend prompts, hot-swap timbre/structure references, and toggle LoRAs while the model generates and plays back continuously. The whole thing runs in your browser against a local GPU.

> Don't have a GPU, or just want to play first? Try the hosted instance at **[music.daydream.live](https://music.daydream.live)**.

## What it does

- Composable multi-condition diffusion with per-frame modulation curves (velocity scaling, SDE denoise, guidance, noise injection, x0 target blending)
- StreamDiffusion-style ring buffer pipeline adapted for audio, with per-slot denoise, source latents, and SDE curves
- Live prompt blending (Tags A ↔ Tags B) with a single text-encoder pass per submission
- LoRA-conditioned generation with a browsable library and optional auto-prepend of trigger words
- Live timbre and structure references; swap to any fixture, uploaded clip, or recorded mic snippet on the fly
- Per-frame draw-on automation curves (denoise, hint strength, feedback, shift, LoRA strengths) with preset templates
- TensorRT acceleration for the DiT decoder, VAE encode, and VAE decode (with a windowed-decode variant for low-latency streaming)
- Mix-and-match backends (TRT / torch.compile / eager) per component
- Fused Triton kernels for Euler/SDE integration
- Residual CFG with per-tick mode selection (off / self / initialize / full)
- Typed node graph system (40+ nodes) for composable generation workflows
- Hardware MIDI control with per-knob MIDI learn; audio-reactive WebGL2 video
- Onboard MCP server to drive every user-facing action from an LLM
- Output recording: audio, or the live graph canvas as a video file with audio muxed in

## Tested on

NVIDIA RTX 3090, 4090, and 5090. The headline performance numbers below are from a 5090. See [Tuning VRAM, latency, and throughput](#tuning-vram-latency-and-throughput) for the knobs that let you fit DEMON to your card.

## Live demo: realtime_motion_graph_web

A Python backend + Next.js front end in a single launcher. Audio in, live ACE-Step stream out, with every control surface in the browser.

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
# then open http://localhost:6660
```

The launcher starts the backend on `:1318` and the Next.js dev server on `:6660`. First run installs `web/node_modules` automatically (Node.js 20+). The browser tab is the entry point for everything below.

Forward backend flags after `--`:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel tensorrt
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint xl
```

See `demos/realtime_motion_graph_web/README.md` for backend args, the wire protocol, and onboard MCP setup.

### Demo highlights

- **Prompt A ↔ B blending.** Two text fields plus a blend slider. Hit "Send Tags" once; the slider lerps conditioning per tick with no encoder re-run.
- **LoRA library.** Browse genre-grouped LoRAs, click to enable, drag faders for strength. If `auto_prepend_lora_triggers` is on (default), enabling a LoRA prepends its primary trigger word to both prompts so what you see is what the encoder sees.
- **Timbre and structure references.** Independent dropdowns that bias instrument character (timbre) and section/rhythm/dynamics (structure) toward any fixture, uploaded clip, or short mic recording. Mix them freely.
- **Source-audio swap.** Library, upload, or record a 60-second snippet from your mic. Swaps tear down and rebuild the session on the new audio.
- **Schedule curves.** Draw automation over the track timeline for denoise, hint strength, feedback, shift, and any LoRA strength. Smooth / linear / step interpolation; presets via right-click.
- **Walk mode.** On by default. Long sources (>60s) route through a 60-second sliding-window engine instead of loading a longer-duration engine, so parameter-update latency stays low on multi-minute tracks. Configured via `engine.walk_window` in [`config.json`](#startup-configuration).
- **MIDI learn.** Right-click any slider, wiggle a physical control, done. Mappings persist per option-profile in localStorage. Endless-encoder semantics built in.
- **Audio-reactive video.** Drop any `.mp4` / `.webm` / `.mov` into `demos/realtime_motion_graph_web/videos/`. WebGL2 shader pipeline does saturation-driven color parallax plus bloom-on-kick. CSS `--bloom-amount` lockstep with the perimeter HUD.
- **Recording.** Capture audio (Opus/WebM, AAC/M4A fallback) or the live graph canvas as video with the same audio muxed in. Soft caps at 60 minutes or ~150 MB.
- **Config import/export.** Snapshot the full live session state (knobs, prompts, LoRA states, curves) to JSON and restore it later.
- **MCP server.** `mcp_server.py` exposes every user-facing action as a stdio MCP tool. Drive the demo from Claude Code or any MCP client for automated testing or scripted performances.

### Startup configuration

All defaults (knob positions, MIDI map seed, enabled LoRAs, walk-window behavior, idle reset, LUFS matcher, audio-reactive shader params, channel slider ranges, XL-checkpoint overrides) live in:

```
demos/realtime_motion_graph_web/web/public/config.json
```

Edit, refresh the browser, done. No rebuild. Field-level help is inline in the file. Schema and defaults are in `web/lib/config.ts`.

## Tuning VRAM, latency, and throughput

Three knobs trade off against each other. Picking the right point on the curve is what makes DEMON run well on a given card.

**Ring buffer depth (`pipeline_depth`, 1–8).** The pipeline keeps `depth` in-flight generations at different denoise stages. After warmup, every tick finishes one of them.

- Higher depth: more concurrent slots, so finished-generation throughput climbs and parameter sweeps sound smoother. But it takes longer for a new knob value to reach the front of the queue (parameter convergence latency goes up), and per-tick VRAM grows.
- Lower depth: a knob change shows up almost immediately and VRAM is lower, but throughput drops and sweeps step rather than glide.
- The web demo's `web/public/config.json` ships with `depth=4` for the 2B checkpoint and `depth=2` for 5B/XL.

**Song duration.** TRT engines are profile-specific. Each engine reserves workspace sized to its profile, so a 240s engine costs more VRAM than a 60s engine even when the workload is only 60 seconds. Per-engine peak workspace, each measured in isolation on a 5090:

| Component       | 60s engine | 240s engine | Δ          |
|-----------------|-----------:|------------:|-----------:|
| Decoder (refit) |  13,511 MB |   15,911 MB |  +2,400 MB |
| VAE decode      |  10,547 MB |   10,814 MB |    +267 MB |
| VAE encode      |   4,178 MB |   10,614 MB |  +6,436 MB |

These are per-engine peaks in separate subprocesses, not a live-runtime sum: at inference time the decoder peak dominates and the VAE workspaces don't peak alongside it, which is why the live demo fits on a 24 GB card. The comparison is what matters: switching three engines from 240s to 60s frees about 9 GB. Source: `scripts/benchmarks/vram_60s_vs_240s_results.md`. Longer engines also pay more per-tick latency since the diffusion sequence length scales with duration. Build only the durations you actually need.

**VAE windowing.** Optional. When `vae_window > 0`, decode happens in overlapped time windows (range 3–30s) instead of full-length. Enables ultra-low-latency streaming updates: the demo's `vae_window=3` default decodes only the slice around the playhead each tick. Disable (set to 0) to fall back to full-length decode.

### Performance

RTX 5090, acestep-v15-turbo (2B), all-TRT, `depth=4`, `steps=8`, `vae_window=3s`, 60 s source.

| Metric | Value |
|---|---|
| Tick (decoder forward, depth=4) | ~43 ms |
| Decode (windowed VAE, 3 s) | 4.5 ms |
| Throughput | 11.3 generations/second |
| Parameter convergence | ~248 ms |
| Per-frame control resolution | 25 Hz (40 ms latent steps) |
| Streaming vs batch quality | bit-identical output (infinite SNR) |

To reproduce on your own hardware:

```bash
uv run python -u -m demos.realtime_motion_graph_web.benchmark \
    --accel tensorrt --depth 4 --vae-window 3 \
    --config demos/realtime_motion_graph_web/web/public/config.json \
    --json runs/depth4-w3.json
```

## Acceleration backends

Both the DiT decoder and the VAE pick a backend independently. Three values each: `tensorrt`, `compile`, `eager`.

| Component        | Backend     | Notes |
|------------------|-------------|-------|
| Decoder          | `tensorrt`  | Fastest. Requires a built TRT engine for the target duration and checkpoint. Refit-enabled engines support LoRA swaps. |
| Decoder          | `compile`   | `torch.compile`. Long warmup, no engine to build, good fallback. |
| Decoder          | `eager`     | Plain PyTorch. Useful for debugging. |
| VAE encode/decode| `tensorrt`  | Fastest. Windowed-decode engine (`vae_decode_fp16_3to30s`) is built once and reused across all durations. |
| VAE encode/decode| `compile`   | `torch.compile`. |
| VAE encode/decode| `eager`     | Plain PyTorch. |

Pass `--accel {tensorrt|compile|eager}` to set both at once. Pass `--decoder-accel` or `--vae-accel` to override one component at a time:

```bash
# All-TRT (recommended).
uv run python -u -m demos.realtime_motion_graph_web.run -- --accel tensorrt

# TRT decoder, eager VAE (e.g. for debugging the decode path).
uv run python -u -m demos.realtime_motion_graph_web.run -- \
    --accel tensorrt --vae-accel eager
```

**Recommended baseline: TRT windowed VAE decoder at minimum.** It's the cheapest TRT engine to build, it's checkpoint- and duration-agnostic, and it unlocks the demo's low-latency streaming behavior. Pair it with `--decoder-accel compile` if you don't want to build the decoder engine yet.

## Requirements

- Python 3.11
- NVIDIA GPU. Tested on 3090, 4090, and 5090 (see [Tuning VRAM, latency, and throughput](#tuning-vram-latency-and-throughput) for the knobs that let you fit DEMON to your card).
- Node.js 20+ (for the web demo; first run installs `web/node_modules` automatically)
- ACE-Step v1.5 checkpoints in `checkpoints/` (auto-downloaded on first run)

## Setup

```bash
uv sync
```

That's it for Python. Audio fixtures pull on first use from the [`daydreamlive/demon-fixtures`](https://huggingface.co/datasets/daydreamlive/demon-fixtures) Hugging Face dataset and cache under `~/.cache/huggingface/`. See `acestep/fixtures.py` for the canonical set.

LoRAs are not auto-downloaded. Drop a `.safetensors` file into `$ACESTEP_MODELS_DIR/loras/` (defaults to `~/.daydream-scope/models/demon/loras/`) and it'll show up in the demo's LoRA library on next refresh. See `acestep/paths.py::loras_dir`.

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

## Programmatic use

The Session API is the simplest path to generating audio in code:

```bash
uv run python examples/session_demo.py
```

Loads the model once, then generates covers in ~310ms per iteration after warmup.

The `examples/covers/` directory has standalone scripts demonstrating individual features. Each one loads the model, runs a single workflow, and saves output audio. A few highlights:

| Workflow | Feature |
|---|---|
| `cover_basic.py` | Standard cover pipeline (encode, condition, generate, decode) |
| `prompt_blend.py` | Two prompts blended with a temporal curve |
| `sde_denoise_curve.py` | Per-frame SDE re-noise modulation |
| `velocity_scaling.py` | Per-frame transformation rate control |
| `lora_generation.py` | LoRA-conditioned generation |
| `x0_target_blend.py` | Two-pass morphing toward a target latent |

See `examples/covers/` for the full set (conditioning average, guidance curve, latent noise mask, initial noise curve, ODE noise injection, semantic blend, x0 target from reference).

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

DEMON is built on top of [ACE-Step](https://github.com/ace-step/ACE-Step). The base diffusion model, VAE, text encoder, and 5Hz LM are all ACE-Step's work; without them, none of this exists. Huge thanks to the ACE-Step team for releasing the v1.5 weights and code under MIT.

If you use DEMON in your work, please also cite ACE-Step.

## Authors

DEMON originally created by Ryan Fosdick ([@RyanOnTheInside](https://ryanontheinside.com)). Maintained by [Daydream Live](https://daydream.live) and contributors.
