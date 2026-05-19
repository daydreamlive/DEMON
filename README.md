# DEMON

**Diffusion Engine for Musical Orchestrated Noise**

Real-time composable diffusion engine for interactive music generation.

> Built on [ACE-Step](https://github.com/ace-step/ACE-Step). DEMON wraps the ACE-Step v1.5 model with a streaming pipeline, TensorRT acceleration, a typed node graph, and realtime MIDI/web demos. The underlying generative model and weights are entirely the work of the ACE-Step team — DEMON is the engine around them.

## Research

Companion technical notes (forthcoming):

- **VAE Distillation** — FastOobleckDecoder
- **Latent Channel Semantics** — 64-channel VAE characterization

Links will be added here as the artifacts are released.

---

## Performance

On an RTX 5090 (32GB), 60-second generations:

| Metric | Value |
|---|---|
| Per-tick latency (depth=8) | ~102ms (decoder 81ms + windowed VAE 21ms) |
| Throughput (depth=8) | 12.3 generations/second |
| Per-frame control resolution | 25 Hz (40ms steps) |
| VAE decode speedup (windowed vs full) | ~3x (63ms → 21ms, bit-identical interior) |
| Streaming vs batch quality | bit-identical output (infinite SNR) |

Pipeline depth trades latency-to-first-effect for throughput:

| Depth | Tick (ms) | Gens/sec | Prompt convergence |
|---|---|---|---|
| 1 | 14 | 8.9 | 14ms |
| 4 | 43 | 11.3 | ~248ms |
| 8 | 81 | 12.3 | ~649ms |

`depth=4` is a good middle ground: 92% of max throughput at ~2.6x faster control response.

## Live demo: realtime_motion_graph_web

Browser front-end + GPU server in a single package. Feed it source audio and a text prompt; twist on-screen knobs (or a connected MIDI controller via Web MIDI) to control denoise strength, SDE curve shape, latent feedback, and diffusion shift while the engine generates and plays back audio continuously. Optional audio-reactive video: drop any `.mp4` into `demos/realtime_motion_graph_web/videos/`.

```bash
uv run python -u -m demos.realtime_motion_graph_web --port 8765
# then open http://<server-host>:8765/
```

MIDI knobs use CC 70-77 across three banks (Core / Groups / Keystones); see `demos/realtime_motion_graph_web/README.md` for the full mapping and the rest of the setup.

## Offline benchmark: stream pipeline stress test

Runs the streaming pipeline end-to-end without interactive I/O, sweeping denoise over many ticks and splicing the output into a single WAV. Use this to validate performance and listen to what the stream pipeline produces across a range of denoise values.

```bash
uv run python demos/test_stream_cover_graph.py --vae-window 15
```

Each tick produces a finished 60-second generation. The output file splices consecutive generations at advancing playback positions so you hear the song progress while the denoise character shifts.

## What it does

- Composable multi-condition diffusion with per-frame modulation curves (velocity scaling, SDE denoise, guidance, noise injection, x0 target blending)
- Automatic execution path selection (fast/switch/batched/sequential) based on active conditions per step
- StreamDiffusion-style ring buffer pipeline adapted for audio, with per-slot denoise, source latents, and SDE curves
- TensorRT acceleration for both the DiT decoder and VAE
- Fused Triton kernels for Euler/SDE integration
- Windowed VAE decode with empirically-sized overlap for streaming
- Typed node graph system (40+ nodes) for composable generation workflows
- Real-time interactive control via MIDI CC or webcam motion

## Requirements

- Python 3.11
- CUDA GPU (tested on RTX 5090, works on 8GB+ VRAM)
- ACE-Step v1.5 checkpoints in `checkpoints/` (auto-downloaded on first run)

## Setup

```bash
uv sync
```

That's it. Audio fixtures used by demos, workflows, and tests are pulled on first use from the [`daydreamlive/demon-fixtures`](https://huggingface.co/datasets/daydreamlive/demon-fixtures) Hugging Face dataset and cached under `~/.cache/huggingface/`. See `acestep/fixtures.py` to add or list the canonical set.

LoRAs are not auto-downloaded yet. If you want to use LoRA-conditioned generation (`examples/covers/lora_generation.py` or the web demo's LoRA picker), drop a `.safetensors` file into `$ACESTEP_MODELS_DIR/loras/` (defaults to `~/.daydream-scope/models/demon/loras/`). See `acestep/paths.py::loras_dir`.

## Quick start

The Session API is the simplest path to generating audio programmatically:

```bash
uv run python examples/session_demo.py
```

Loads the model once, then generates covers in ~310ms per iteration after warmup.

## Demos

| Script | What it does |
|---|---|
| `demos/realtime_motion_graph_web/` | Real-time generation with browser front-end + GPU server (single port) |
| `demos/test_stream_cover_graph.py` | StreamPipeline stress test with denoise sweep |
| `examples/session_demo.py` | Session API basics: load once, generate many |
| `examples/realtime_cover.py` | Interactive cover generation with live parameter control |
| `examples/session_test_all.py` | Exercises all node system features end-to-end |

## Workflow examples

The `examples/covers/` directory contains standalone scripts demonstrating individual features. Each loads the model, runs one workflow, and saves output audio.

| Workflow | Feature |
|---|---|
| `cover_basic.py` | Standard cover pipeline (encode, condition, generate, decode) |
| `sde_denoise_curve.py` | Per-frame SDE re-noise modulation |
| `velocity_scaling.py` | Per-frame transformation rate control |
| `prompt_blend.py` | Two prompts blended with a temporal curve |
| `x0_target_blend.py` | Two-pass morphing toward a target latent |
| `guidance_curve.py` | Per-frame CFG scale via positive + zeroed-out negative |
| `conditioning_average.py` | Weighted average of two text conditionings |
| `cover_semantic_blend.py` | Blend structural hints from two source audios |
| `latent_noise_mask.py` | Temporal inpainting mask on source latent |
| `initial_noise_curve.py` | Per-frame source/noise mixing in initial latent |
| `ode_noise_injection.py` | Per-frame ODE solver noise injection |
| `lora_generation.py` | LoRA-conditioned generation |
| `x0_target_from_reference.py` | Reference audio as x0 target for blending |

## Running with TensorRT

DEMON targets TensorRT 10.16.x for optional acceleration. TensorRT plans are
version- and GPU-architecture-specific by default, so rebuild engines after
changing TensorRT, CUDA, driver, or the GPU used for inference.

Build TRT engines:

```bash
# Build all engines (60s + 120s + 240s, VAE + refit decoder)
uv run python -m acestep.engine.trt.build --all

# Build 60s engines only (recommended starting point)
uv run python -m acestep.engine.trt.build --all --duration 60

# Preview what will be built
uv run python -m acestep.engine.trt.build --all --dry-run

# Force rebuild even if engines already exist (skipped by default)
uv run python -m acestep.engine.trt.build --all --force-rebuild

# Force ONNX re-export as well as engine rebuild
uv run python -m acestep.engine.trt.build --all --duration 60 --force-rebuild --force-onnx

# Only decoders or only VAE
uv run python -m acestep.engine.trt.build --all --decoder-only
uv run python -m acestep.engine.trt.build --all --vae-only
```

ONNX intermediates are duration-agnostic and auto-detected on subsequent
builds (the model is only loaded when an ONNX export is actually needed):

```
trt_engines/
  _onnx/                          # shared, auto-reused across durations
    vae_encode/vae_encode.onnx
    vae_decode/vae_decode.onnx
    decoder/decoder.onnx          # + external data shards
    decoder_refit/decoder_refit.onnx
  decoder_mixed_refit_b8_60s/
    decoder_mixed_refit_b8_60s.engine
  vae_decode_fp16_60s/
    vae_decode_fp16_60s.engine
  ...
```

Pass engine paths to Session:

```python
session = Session(
    trt_engines={
        "decoder": "trt_engines/decoder_mixed_refit_b8_60s/decoder_mixed_refit_b8_60s.engine",
        "vae_encode": "trt_engines/vae_encode_fp16_60s/vae_encode_fp16_60s.engine",
        "vae_decode": "trt_engines/vae_decode_fp16_60s/vae_decode_fp16_60s.engine",
    },
)
```

For single engine builds (fine-grained control):

```bash
# VAE only, 60s profile
uv run python -m acestep.engine.trt.build --max-duration 60

# Decoder only, refit-enabled for LoRA
uv run python -m acestep.engine.trt.build --skip-vae --decoder --decoder-mixed --decoder-refit --max-duration 60
```

## Tests

```bash
uv run pytest tests/ -v
```

## Acknowledgments

DEMON is built on top of [ACE-Step](https://github.com/ace-step/ACE-Step). The base diffusion model, VAE, text encoder, and 5Hz LM are all ACE-Step's work — without them, none of this exists. Huge thanks to the ACE-Step team for releasing the v1.5 weights and code under MIT.

If you use DEMON in your work, please also cite ACE-Step.

## Authors

DEMON originally created by Ryan Fosdick ([@RyanOnTheInside](https://ryanontheinside.com)). Maintained by [Daydream Live](https://daydream.live) and contributors.
