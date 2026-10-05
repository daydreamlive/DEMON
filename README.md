# DEMON

<p align="center"><strong>Diffusion Engine for Musical Orchestrated Noise</strong></p>
<p align="center"><sub>A platform for real-time music generation</sub></p>

<p align="center">
  <a href="LICENSE"><img alt="License: AGPL-3.0-or-later + MIT" src="https://img.shields.io/badge/license-AGPL--3.0--or--later%20%2B%20MIT-blue.svg"></a>
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue.svg">
  <a href="#model-families"><img alt="Model families: 5" src="https://img.shields.io/badge/model%20families-5-yellow.svg"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.28657"><img alt="DEMON paper on arXiv" src="https://img.shields.io/badge/paper-arXiv%3A2605.28657-b31b1b.svg"></a>
  <a href="https://daydreamlive.github.io/DEMON/"><img alt="DEMON project page" src="https://img.shields.io/badge/project%20page-daydreamlive.github.io-111111.svg"></a>
  <a href="https://music.daydream.live"><img alt="Try the hosted instance" src="https://img.shields.io/badge/hosted%20instance-music.daydream.live-ff7a00.svg"></a>
</p>

<p align="center">
  <img src="docs/assets/img/poster-hero.jpg" width="820" alt="The DEMON web demo: a live control drawer with automation curves and audio-reactive WebGL visuals reacting to generated music">
</p>
<p align="center"><sub><em>The DEMON web demo: live control drawer, automation curves, audio-reactive visuals.</em></sub></p>

DEMON streams music from a generative model while you steer it. One streaming engine, one wire protocol and one client SDK sit in front of interchangeable model families. Stable Audio 3 and ACE-Step v1.5 are featured; Magenta RealTime 2, MiniMax-Music3 and YuE2 are experimental. The streaming diffusion engine behind the featured families is described in the [paper](https://arxiv.org/abs/2605.28657) and on the [project page](https://daydreamlive.github.io/DEMON/).

> No GPU, or just want to play first? Try the hosted instance at **[music.daydream.live](https://music.daydream.live)**.

## Contents

- [Model families](#model-families)
  - [Benchmarks (RTX 5090)](#benchmarks-rtx-5090)
  - [Limitations](#limitations)
- [Quickstart](#quickstart)
- [The streaming engine](#the-streaming-engine)
- [ACE-Step engine reference](#ace-step-engine-reference)
  - [Performance](#performance)
  - [Tuning](#tuning)
  - [Acceleration backends](#acceleration-backends)
  - [Programmatic use: the Session API](#programmatic-use-the-session-api)
  - [Building TensorRT engines](#building-tensorrt-engines)
- [Demo applications](#demo-applications)
- [Research & citation](#research--citation)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [Authors](#authors)
- [License](#license)

## Model families

A server boots one family (`--checkpoint <alias>`). Every client, demo and integration speaks the same protocol whichever family is loaded, and reads that family's capabilities and knobs from the handshake instead of assuming a model.

| Family | Status | Steer it with | Boot alias |
|---|---|---|---|
| [Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3-small-music) (small-music, medium) | Featured | Per-frame curves, prompt morphing, audio-to-audio from an uploaded source | `sa3-small`, `sa3-medium` |
| [ACE-Step v1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (turbo 2B, XL turbo 5B) | Featured, the default install | Per-frame curves on every solver knob, prompt A/B morphing, LoRA hot-swap, timbre and structure references, audio in | default; `xl` |
| Magenta RealTime 2 (`mrt2_small`, `mrt2_base`) | Experimental | Prompt A/B blend, sampling and guidance knobs; append-only | `mrt2-sidecar` |
| [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) | Experimental, needs a 32 GB card | Style prompt and lyrics; append-only | `minimax-music3` |
| [YuE2](https://github.com/multimodal-art-projection/YuE) (3B) | Experimental, weights CC BY-NC 4.0 | Style prompt and lyrics compose the song; denoise, step count, x0 target, feedback and seed reshape it live | `yue2-3b` |

Per-family setup, knobs, known gaps, and the contract for adding a family are in [docs/FAMILIES.md](docs/FAMILIES.md).

### Benchmarks (RTX 5090)

One figure per cell; the full measurements are in [docs/FAMILIES.md](docs/FAMILIES.md) and [docs/MINIMAX.md](docs/MINIMAX.md). Latency is measured differently per family (parameter convergence, delivery frontier, or decoded PCM), so read down a column loosely.

| Family | Throughput | Knob to ear | First audio, cold server | VRAM per session |
|---|---|---|---|---|
| Stable Audio 3 medium (TensorRT, 8 steps) | 6.3 generations/s, flat across ring depth | 0.2 s at depth 1, 1.2 s at depth 4 | 17-21 s | 8-11 GB |
| ACE-Step v1.5 turbo 2B (TensorRT, depth 4) | 11.3 generations/s | ~0.25 s | ~15 s | fits a 24 GB card |
| Magenta RealTime 2 | 1.7x realtime (small), 0.9x (base) | 0.75 s lead | ~30 s sidecar warmup, once | not measured |
| MiniMax-Music3 | 1.3x realtime | 3-4 s | ~30 s | 28.7 GB of 32 GB |
| YuE2 3B | 70 ms tick, 32 ticks per solve | 2.4 s; a prompt change re-composes in 20-60 s | ~34 s | 18-22 GB |

For the two diffusion families, ring depth trades control latency for smoother parameter glides; it does not raise throughput on Stable Audio 3.

### Limitations

- **Stable Audio 3.** Weights are a manual download; the model variant is fixed at boot; the session renders one fixed length (up to 120 s).
- **ACE-Step v1.5.** Fixed-length canvas set by the TensorRT profile (60 s by default); engines are specific to the TensorRT version and GPU.
- **Magenta RealTime 2.** Runs as a JAX sidecar in a Linux/WSL venv; append-only on a 60 s window; one session per sidecar; the base model runs below realtime.
- **MiniMax-Music3.** Uses 28.7 GB of a 32 GB card and does not fit 24 GB; 1.3x realtime; append-only, so changes land in seconds; about 30 s to first audio on a fresh server. Its licence requires "MiniMax-Music3" to be displayed in a product UI and written authorisation above US$20M yearly revenue ([docs/MINIMAX.md](docs/MINIMAX.md)).
- **YuE2.** One acoustic step per tick, so the step grid sets the latency floor (`yue2_steps`, default 32); a prompt change re-composes the song (20-60 s); lyrics and length are fixed per session; ring depth is capped at 1; weights are non-commercial (CC BY-NC 4.0).

## Quickstart

**You need:** an NVIDIA GPU (tested on RTX 3090 / 4090 / 5090; the ACE-Step demo fits a 24 GB card), [uv](https://docs.astral.sh/uv/), Node.js 20+ (web demo only), and about 40 GB of free disk. `uv sync` installs Python 3.11 for you.

```bash
git clone https://github.com/daydreamlive/DEMON.git
cd DEMON
uv sync
uv run demon-setup
```

`demon-setup` checks the environment, downloads the ACE-Step v1.5 checkpoints (~18 GB), fetches the pinned Stable Audio 3 source checkout, downloads a starter pack of genre LoRAs, and builds the minimal TensorRT engine set (a few minutes on a recent GPU). It is idempotent; re-run it any time.

### Stable Audio 3

The weights are a manual download:

```bash
# small-music (or stable-audio-3-medium for the medium checkpoint)
huggingface-cli download stabilityai/stable-audio-3-small-music \
  --local-dir ~/.daydream-scope/models/demon/sa3/checkpoints/stable-audio-3-small-music
```

Boot with an SA3 alias, then open the SA3 page:

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint sa3-small
# or: --checkpoint sa3-medium
# open http://localhost:6660/sa3/  (http://localhost:1318/sa3/ when running the backend directly)
```

The variant is resolved at boot; opening `/sa3/` against an ACE-Step server does not switch models.

### ACE-Step v1.5

```bash
uv run python -u -m demos.realtime_motion_graph_web.run
# open http://localhost:6660
```

The page loads with a fixture selected. Click **Play** (browsers gate audio behind a click). The first start takes ~15 s while the model and engines load; then audio streams continuously and every control in the drawer is live. The spectral-control sliders are in the drawer's **Experimental** tab; they steer generation itself, so changes land on upcoming audio after a moment.

> The bare launch runs all-TensorRT by default and needs the engines `demon-setup` built. If they are missing, the server exits at boot and prints the fix. After `demon-setup --skip-engines`, launch with `-- --accel compile` instead (long `torch.compile` warmup on the first tick).

**Where things live.** Everything downloads to `~/.daydream-scope/models/demon/` (override with `ACESTEP_MODELS_DIR`), not into the repository: checkpoints under `<models dir>/checkpoints/`, Stable Audio 3 source under `<models dir>/sa3/vendor/`, TensorRT engines under `<models dir>/trt_engines/`. Use the v1.5 weights fetched by `demon-setup` (equivalently `uv run acestep-download`); do not substitute other checkpoints. Directory tree, manual download, engine options, headless notes and troubleshooting are in [docs/INSTALL.md](docs/INSTALL.md).

**Fixtures and LoRAs.** Demo audio pulls on first use from the [`daydreamlive/demon-fixtures-v2`](https://huggingface.co/datasets/daydreamlive/demon-fixtures-v2) dataset into `<models dir>/fixtures/` ([`acestep/fixtures.py`](acestep/fixtures.py)). `demon-setup` downloads 16 genre LoRAs (skip with `--skip-loras`); drop your own `.safetensors` (optionally with a `<stem>.metadata.json` sidecar) under `<models dir>/loras/` and it appears on the next refresh.

### Experimental families

Each boots with its own alias; its demo page is served by the backend on `:1318`.

**Magenta RealTime 2.** Start the sidecar in a Linux/WSL venv with `magenta_rt`, JAX (CUDA) and numpy, no torch. It listens on `127.0.0.1:7531` after a ~30 s JIT warmup (override with `DEMON_MRT2_SIDECAR=host:port`), then boot the server:

```bash
python scripts/mrt2_sidecar.py --model mrt2_small
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint mrt2-sidecar
# open http://localhost:1318/mrt2/
```

**MiniMax-Music3.** Needs the diffusers layout of [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/MiniMaxAI/MiniMax-Music3) under `DEMON_MINIMAX_DIR`, `<models dir>/minimax/checkpoints/MiniMax-Music3`, or the Hugging Face cache, and a 32 GB card with nothing else on it. An optional TensorRT engine for the renderer is built by `acestep/engine/trt/minimax_build.py`.

```bash
uv run python -u -m demos.realtime_motion_graph_web.run -- --checkpoint minimax-music3
# open http://localhost:1318/minimax/
```

**YuE2.** Runs in the server process. Set the environment (details and the TensorRT build in [docs/FAMILIES.md](docs/FAMILIES.md)), then start the server:

```bash
export DEMON_YUE2_ROOT=/path/to/yue2              # weights (YuE2-3B + codec)
export DEMON_YUE2_YUE_SRC=/path/to/YuE/src        # upstream YuE source at the pinned revision
export DEMON_YUE2_EXTRA_PATH=/path/to/extra-deps  # optional extra import path
export DEMON_YUE2_TRT_DIR=/path/to/engines        # optional; eager without it
python -u -m demos.realtime_motion_graph_web.server --port 1318 --checkpoint yue2-3b
# open http://localhost:1318/yue2/
```

Lyrics and duration are fixed at Start; the page shows "composing" while the autoregressive stages run.

## The streaming engine

Stable Audio 3 and ACE-Step run through one streaming diffusion pipeline ([`acestep/engine/stream.py`](acestep/engine/stream.py)); YuE2's acoustic stage runs in the same ring. The idea, as in StreamDiffusion for images: a ring buffer holds several in-flight generations at different denoising stages and advances them together in one batched forward pass per tick. After warmup, finished latents stream out at `depth/steps` generations per tick, and anything you change is heard on the next output.

- **Per-frame steering.** Velocity, guidance, noise injection, x0 targets and source preservation each accept a scalar or a `[T]` curve at the latent's 25 Hz frame rate, hot-mutable mid-stream.
- **Shared curves.** A write to a shared curve reaches every in-flight slot on the next tick, independent of ring depth.
- **Heterogeneous slots.** Each slot carries its own seed, denoise strength, schedule, conditioning, CFG mode and mask, and a regeneration, a style transfer and an RCFG request batch together.
- **Hot ring depth.** `pipeline_depth` is resizable while streaming; slots drain naturally.
- **End-to-end TensorRT.** DiT and VAE encode/decode all run through TensorRT; the decoder is refit-enabled, so a LoRA swap never rebuilds an engine.
- **Bit-identical to batch.** Streaming and one-shot paths compose the same step primitives ([`acestep/engine/ode_steps.py`](acestep/engine/ode_steps.py)) and produce the same output.
- **One surface, many clients.** The Session API ([`acestep/engine/session.py`](acestep/engine/session.py)), the typed node graph ([`acestep/nodes/`](acestep/nodes/)), the WebSocket protocol, the client SDK ([`packages/demon-client`](packages/demon-client/)) and the onboard MCP server all drive the same primitives.

The paper covers the mechanism in full: the SDE derivation behind the source-preservation curve, the windowed-decode receptive-field analysis, and the TensorRT precision recipe.

## ACE-Step engine reference

The sections below document the ACE-Step path in detail. Stable Audio 3 shares the pipeline through a model adapter; the other families are documented in [docs/FAMILIES.md](docs/FAMILIES.md).

### Performance

RTX 5090, ACE-Step v1.5 turbo (2B), all-TRT, `depth=4`, `steps=8`, `vae_window=3s`, 60 s source.

| Metric | Value |
|---|---|
| Tick (decoder forward, depth=4) | ~43 ms |
| Decode (windowed VAE, 3 s) | 4.5 ms |
| Throughput | 11.3 generations/second |
| Parameter convergence | ~248 ms |
| Per-frame control resolution | 25 Hz (40 ms latent steps) |
| Streaming vs. batch quality | bit-identical output |

Tested on RTX 3090, 4090 and 5090. The demo fits a 24 GB card with the 60 s engines (see the VRAM breakdown under [Tuning](#tuning)).

### Tuning

Three settings trade off against each other.

- **Ring buffer depth (`pipeline_depth`, 1 to 8).** Higher depth makes parameter sweeps glide through finer intermediate states at the cost of per-tick compute, VRAM and control latency; lower depth is snappier and cheaper.
- **Song duration.** TensorRT engines are profile-specific and reserve workspace sized to the profile, so a 240 s engine costs more VRAM and tick time than a 60 s engine even for a 60 s workload. Build only the durations you need.
- **VAE windowing.** The demo's default. With `vae_window > 0`, every streaming decode runs through the fixed 1 s windowed engine and only the requested window is decoded per call, which is what unlocks low-latency updates. Set it to 0 for full-length decode through the `vae_decode` engine.

<details>
<summary><strong>Per-engine VRAM: 60 s vs 240 s profiles (5090)</strong></summary>

Per-engine peak workspace, each measured in isolation:

| Component       | 60s engine | 240s engine |          Δ |
|-----------------|-----------:|------------:|-----------:|
| Decoder (refit) |  13,511 MB |   15,911 MB |  +2,400 MB |
| VAE decode      |  10,547 MB |   10,814 MB |    +267 MB |
| VAE encode      |   4,178 MB |   10,614 MB |  +6,436 MB |

These are isolated peaks, not a live-runtime sum; at inference the decoder peak dominates, which is why the live demo fits a 24 GB card. Switching the three engines from 240 s to 60 s frees about 9 GB. Source: [`scripts/benchmarks/vram_60s_vs_240s_results.md`](scripts/benchmarks/vram_60s_vs_240s_results.md).

</details>

### Acceleration backends

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

### Programmatic use: the Session API

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

Scripts: [`examples/session_demo.py`](examples/session_demo.py) (persistent session, covers across seeds), [`examples/realtime_cover.py`](examples/realtime_cover.py) (dual prompts, dual LoRAs, timbre and hint references, temporal masking, per-frame curves), and one standalone script per feature in [`examples/covers/`](examples/covers/).

<details>
<summary><strong>Per-feature example scripts</strong></summary>

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

</details>

### Building TensorRT engines

DEMON targets TensorRT 10.16.x. Plans are version- and GPU-specific, so rebuild after changing TensorRT, CUDA, the driver or the GPU. The minimal set for the web demo (what `demon-setup` builds) is the 60 s profile plus the fixed 1 s windowed VAE decode:

```bash
uv run python -m acestep.engine.trt.build --preset minimal
```

ONNX intermediates are duration-agnostic and reused across builds. The full matrix, precision recipes, the XL/FP8 path and engine naming are in [docs/TRT.md](docs/TRT.md).

<details>
<summary><strong>All build commands and on-disk engine layout</strong></summary>

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

</details>

## Demo applications

### realtime_motion_graph_web

The headline demo: a Python backend and a Next.js front-end in one launcher. Feed it audio and a prompt, then twist knobs, draw automation curves, blend prompts, swap timbre and structure references and toggle LoRAs while the model plays.

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

Highlights: prompt A/B blending with a per-tick slider; a genre-grouped LoRA library with strength faders; timbre and structure references from fixtures, uploads or the mic; source-audio swap; drawn automation curves for denoise, hint strength, feedback, shift and LoRA strength; MIDI learn on any slider; audio-reactive WebGL visuals; audio and video recording; config import/export; and an onboard MCP server that exposes every action as a tool.

<p align="center">
  <img src="docs/assets/img/poster-mcp-agent.jpg" width="720" alt="An AI agent driving a live DEMON session through the onboard MCP server">
</p>

Defaults live in [`demos/realtime_motion_graph_web/web/public/config.json`](demos/realtime_motion_graph_web/web/public/config.json). Backend args, wire protocol, MCP setup and front-end architecture are in [`demos/realtime_motion_graph_web/README.md`](demos/realtime_motion_graph_web/README.md).

### Other entry points

- [`demos/sa3/`](demos/sa3/), [`demos/mrt2/`](demos/mrt2/), [`demos/minimax/`](demos/minimax/), [`demos/yue2/`](demos/yue2/): static per-family pages on the client SDK, served by the backend.
- [`examples/session_demo.py`](examples/session_demo.py), [`examples/realtime_cover.py`](examples/realtime_cover.py), [`examples/covers/`](examples/covers/): the Session API from Python.
- [`demos/test_stream_cover_graph.py`](demos/test_stream_cover_graph.py): a streaming cover graph driven from Python.

## Research & citation

- **[DEMON: Diffusion Engine for Musical Orchestrated Noise](https://arxiv.org/abs/2605.28657)**, arXiv:2605.28657. Audio examples and experiments on the [project page](https://daydreamlive.github.io/DEMON/).
- FastOobleckDecoder (VAE distillation) and Latent Channel Semantics (64-channel VAE characterization): companion notes, forthcoming.

If you use DEMON, please cite DEMON and the model of the family you ran. For ACE-Step:

```bibtex
@article{fosdick2026demon,
  title   = {DEMON: Diffusion Engine for Musical Orchestrated Noise},
  author  = {Fosdick, Ryan},
  journal = {arXiv preprint arXiv:2605.28657},
  year    = {2026}
}

@software{demon,
  author = {Fosdick, Ryan},
  title  = {DEMON: Diffusion Engine for Musical Orchestrated Noise},
  year   = {2026},
  url    = {https://github.com/daydreamlive/DEMON}
}

@article{acestep2026,
  title   = {ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
  author  = {Gong and others},
  journal = {arXiv preprint arXiv:2602.00744},
  year    = {2026}
}
```

## Contributing

Contributions are welcome. [AGENTS.md](AGENTS.md) is the maintained developer guide: dev setup, the contract-first control surface (knobs and the wire protocol each live in one registry), and how to regenerate the TypeScript types after a registry change. Run the tests with:

```bash
uv run pytest tests/
```

## Acknowledgments

DEMON does not train or own the models it runs. Each family is its upstream team's work:

- **ACE-Step v1.5**: the [ACE-Step team](https://github.com/ace-step/ACE-Step). DEMON began as a streaming engine for this model, and the base diffusion model, VAE, text encoder and 5 Hz LM are all theirs, released under MIT.
- **Stable Audio 3**: [Stability AI](https://huggingface.co/stabilityai).
- **Magenta RealTime 2**: Google Magenta (`magenta_rt`).
- **MiniMax-Music3**: [MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-Music3).
- **YuE2**: the m-a-p YuE team ([code](https://github.com/multimodal-art-projection/YuE), [weights](https://huggingface.co/m-a-p/YuE2-3B)). Weights CC BY-NC 4.0, code Apache-2.0; not cleared for hosted or commercial use.

The ring-buffer streaming pattern follows [StreamDiffusion](https://arxiv.org/abs/2312.12491) (Kodaira et al., 2023).

## Authors

DEMON was created by Ryan Fosdick ([@RyanOnTheInside](https://ryanontheinside.com)). Maintained by [Daydream Live](https://daydream.live) and contributors.

## License

DEMON is distributed under the **GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`); see [`LICENSE`](LICENSE). Modified versions made available over a network must offer users the corresponding source (AGPL §13).

Portions of DEMON derive from [ACE-Step](https://github.com/ace-step/ACE-Step), released under MIT. The MIT notice is preserved in [`LICENSE-MIT`](LICENSE-MIT); the ACE-Step portions remain available under MIT on their own terms, while the combined work is offered under AGPL-3.0-or-later.
